"""Queue allocation, immutable execution and process ownership without a GPU."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import robotwin_dev_queue as queue


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed(path):
    write(path/'run.json', {'status': 'outcomes_complete'})
    write(path/'counterexample_summary.json', {'complete': True, 'counts': {'planned': 26, 'received': 26}})


def plan(tmp_path, count=3, gpus=None):
    code = tmp_path/'code'; code.mkdir(exist_ok=True)
    script = code/'search_robotwin_constraint_counterexamples.py'
    script.write_text('# frozen test runner\n')
    frozen = {str(script): queue.file_hash(script)}
    inputs = {}
    for name in ('reference', 'reference-manifest', 'template', 'split-plan'):
        path = code/(name+'.json'); write(path, {'name': name})
        inputs[name] = str(path); frozen[str(path)] = queue.file_hash(path)
    jobs = []
    for index in range(count):
        output = tmp_path/'outputs'/f'job{index}'
        command = [sys.executable, str(script), '--split', 'dev', '--output', str(output)]
        for key, value in inputs.items(): command += ['--'+key, value]
        jobs.append(dict(job_id=f'job{index}', mode='queued', output=str(output), command=command,
                         cwd=str(code), frozen_files=frozen.copy(), environment={'TEST_MARKER': 'frozen'}))
    return dict(version=1, queue_root=str(tmp_path/'queue'), lease_root=str(tmp_path/'leases'),
                gpus=[0, 1] if gpus is None else gpus, poll_seconds=.01, jobs=jobs)


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid; self.returncode = None; self.stdin = io.BytesIO()

    def poll(self): return self.returncode


class FakeRegistry:
    def __init__(self): self.children = []

    def add(self, process, log_handle=None): self.children.append((process, log_handle))

    def terminate_all(self):
        for process, log in self.children:
            if process.poll() is None: process.returncode = -15
            if log: log.close()


def fake_runtime(monkeypatch):
    children, calls, busy, identities = [], [], {}, {}
    def identity(pid):
        if pid in identities: return identities[pid]
        child = next((c for c in children if c.pid == pid), None)
        if child and child.poll() is not None: return None
        return dict(pid=pid, boot_id='boot', start_ticks=pid*10, command_sha256='a'*64)
    def spawn(command, **kwargs):
        child = FakeProcess(900000+len(children)); children.append(child); calls.append((command, kwargs))
        return child
    monkeypatch.setattr(queue, 'process_identity', identity)
    monkeypatch.setattr(queue, 'gpu_occupancy', lambda gpus: {gpu: busy.get(gpu, []) for gpu in gpus})
    monkeypatch.setattr(queue, 'spawn_resource', spawn)
    monkeypatch.setattr(queue, 'ProcessRegistry', FakeRegistry)
    return children, calls, busy, identities


def test_fills_only_free_selected_cards_then_backfills_completed_slot(tmp_path, monkeypatch):
    children, calls, busy, _ = fake_runtime(monkeypatch)
    p = plan(tmp_path); busy[1] = [12345]; q = queue.Queue(p); q.prepare()
    try:
        assert q.step() and len(children) == 1
        assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '0'
        completed(Path(p['jobs'][0]['output'])); children[0].returncode = 0
        q.step()
        assert len(children) == 2 and calls[1][1]['env']['CUDA_VISIBLE_DEVICES'] == '0'
        busy.clear(); q.step()
        assert len(children) == 3 and calls[2][1]['env']['CUDA_VISIBLE_DEVICES'] == '1'
        assert q.statuses['job0']['phase'] == 'completed'
    finally: q.close()


def test_gpu_inventory_outage_keeps_owned_work_and_blocks_new_launch(tmp_path, monkeypatch):
    children, _, busy, _ = fake_runtime(monkeypatch)
    p = plan(tmp_path); busy[1] = [123]; q = queue.Queue(p); q.prepare(); q.step()
    def unavailable(_): raise subprocess.TimeoutExpired('nvidia-smi', 20)
    monkeypatch.setattr(queue, 'gpu_occupancy', unavailable)
    try:
        assert q.step() and len(children) == 1 and children[0].poll() is None
        assert 'TimeoutExpired' in queue.read(q.root/'state.json')['gpu_inventory_error']
    finally: q.close()


def test_changed_code_and_existing_failed_output_are_never_retried(tmp_path, monkeypatch):
    children, _, _, _ = fake_runtime(monkeypatch); p = plan(tmp_path, count=2)
    Path(p['jobs'][0]['output']).mkdir(parents=True)
    Path(p['jobs'][1]['command'][1]).write_text('# changed after review\n')
    q = queue.Queue(p); q.prepare()
    try:
        assert not q.step() and not children
        assert q.statuses['job0']['phase'] == 'blocked_existing'
        assert q.statuses['job1']['phase'] == 'blocked_preflight'
        assert not q.step() and not children
        assert queue.read(q.root/'state.json')['phase'] == 'queue_exhausted'
    finally: q.close()


def test_existing_process_is_observed_never_killed_or_restarted(tmp_path, monkeypatch):
    children, _, _, identities = fake_runtime(monkeypatch); p = plan(tmp_path, count=1, gpus=[0])
    identity = dict(pid=12345, boot_id='boot', start_ticks=42, command_sha256='b'*64)
    identities[12345] = identity
    receipt = tmp_path/'standalone.json'; write(receipt, {'pid': 12345})
    observed = dict(job_id='existing', mode='observe', output=str(tmp_path/'existing'), gpu=0, process=identity,
                    launch_receipt=dict(path=str(receipt),sha256=queue.file_hash(receipt),pid_field='pid'))
    p['jobs'].insert(0, observed); q = queue.Queue(p); q.prepare()
    try:
        q.step(); assert not children and q.statuses['existing']['phase'] == 'observing_existing'
        # A recycled PID with the same command is not the observed old process.
        identities[12345] = dict(identity, start_ticks=43)
        q.step(); assert q.statuses['existing']['phase'] == 'blocked_existing'
        assert len(children) == 1
    finally: q.close()
    assert identities[12345]['start_ticks'] == 43


def test_resume_of_owned_live_guardian_observes_instead_of_spawning(tmp_path, monkeypatch):
    children, _, _, identities = fake_runtime(monkeypatch); p = plan(tmp_path, count=1, gpus=[0])
    q = queue.Queue(p); q.prepare()
    identity = dict(pid=33333, boot_id='boot', start_ticks=42, command_sha256='b'*64)
    identities[33333] = identity
    write(q.root/'jobs/job0/launch.json', dict(process=identity, gpu=0))
    try:
        assert q.step() and not children
        assert q.statuses['job0']['phase'] == 'observing_existing'
    finally: q.close()
    assert identities[33333] == identity


def test_shared_lease_prevents_two_queues_claiming_one_card(tmp_path, monkeypatch):
    children, _, _, _ = fake_runtime(monkeypatch)
    left = tmp_path/'left'; right = tmp_path/'right'; left.mkdir(); right.mkdir()
    p1=plan(left,count=1,gpus=[0]);p2=plan(right,count=1,gpus=[0]);p2['lease_root']=p1['lease_root']
    q1=queue.Queue(p1);q2=queue.Queue(p2);q1.prepare();q2.prepare()
    try:
        q1.step();q2.step();assert len(children)==1 and q2.statuses['job0']['phase']=='pending'
        completed(Path(p1['jobs'][0]['output']));children[0].returncode=0;q1.step();q2.step()
        assert len(children)==2
    finally:q1.close();q2.close()


def test_lease_receipt_keeps_card_reserved_while_orphan_guardian_cleans_up(tmp_path, monkeypatch):
    children, _, _, _ = fake_runtime(monkeypatch)
    left=tmp_path/'left';right=tmp_path/'right';left.mkdir();right.mkdir()
    p1=plan(left,count=1,gpus=[0]);p2=plan(right,count=1,gpus=[0]);p2['lease_root']=p1['lease_root']
    q1=queue.Queue(p1);q2=queue.Queue(p2);q1.prepare();q2.prepare()
    try:
        q1.step()
        # Parent death releases flock before its guardian necessarily exits.
        q1.running['job0']['lease'].close()
        q2.step();assert len(children)==1
        children[0].returncode=130
        q2.step();assert len(children)==2
    finally:q1.close();q2.close()


def test_frozen_queue_cannot_be_appended_in_place_and_empty_queue_is_honest_idle(tmp_path, monkeypatch):
    children, _, _, _ = fake_runtime(monkeypatch);p=plan(tmp_path,count=0)
    q=queue.Queue(p);q.prepare()
    try:
        assert not q.step() and not children
        state=queue.read(q.root/'state.json');assert state['phase']=='queue_exhausted' and 'No remaining' in state['idle_reason']
    finally:q.close()
    changed=copy.deepcopy(p);changed['gpus']=[0]
    other=queue.Queue(changed)
    try:
        with pytest.raises(ValueError,match='Frozen queue'):other.prepare()
    finally:other.close()


def test_development_only_and_all_command_inputs_must_be_frozen(tmp_path):
    p=plan(tmp_path,count=1);queue.validate_plan(p)
    bad=copy.deepcopy(p);bad['jobs'][0]['command'][3]='holdout'
    with pytest.raises(ValueError,match='development'):queue.validate_plan(bad)
    bad=copy.deepcopy(p);bad['jobs'][0]['frozen_files'].pop(bad['jobs'][0]['command'][1])
    with pytest.raises(ValueError,match='explicitly hashed'):queue.validate_plan(bad)


def test_failed_job_is_terminal_and_next_pre_reviewed_job_can_run(tmp_path, monkeypatch):
    children, _, _, _=fake_runtime(monkeypatch);p=plan(tmp_path,count=2,gpus=[0]);q=queue.Queue(p);q.prepare()
    try:
        q.step();children[0].returncode=1;q.step()
        assert q.statuses['job0']['phase']=='failed' and len(children)==2
        assert queue.read(q.root/'jobs/job0/launch_intent.json')['command']==p['jobs'][0]['command']
    finally:q.close()


def test_real_guardian_cleans_only_owned_child_when_queue_stops(tmp_path, monkeypatch):
    p=plan(tmp_path,count=1,gpus=[0]);job=p['jobs'][0];script=Path(job['command'][1])
    script.write_text('import os,sys,time\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--output")+1]);(p/"native.pid").write_text(str(os.getpid()))\ntime.sleep(120)\n')
    job['frozen_files'][str(script)]=queue.file_hash(script)
    monkeypatch.setattr(queue,'gpu_occupancy',lambda gpus:{gpu:[] for gpu in gpus})
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
    q=queue.Queue(p);q.prepare()
    try:
        q.step();pid_file=Path(job['output'])/'native.pid';deadline=time.monotonic()+5
        while not pid_file.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert pid_file.exists();native_pid=int(pid_file.read_text());identity=queue.process_identity(native_pid)
        assert identity is not None
        q.close()
        assert not queue.same_process(identity) and unrelated.poll() is None
        assert q.statuses['job0']['phase']=='interrupted'
    finally:
        q.close();unrelated.terminate();unrelated.wait(timeout=5)


def test_capture_existing_identity_requires_matching_receipt_and_command_output(tmp_path):
    output=tmp_path/'existing';receipt=tmp_path/'receipt.json'
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)',str(output)])
    try:
        write(receipt,dict(pid=child.pid))
        observed=queue.capture_observation('external',output,2,child.pid,receipt)
        assert queue.same_process(observed['process']) and observed['gpu']==2
        with pytest.raises(ValueError,match='output'):
            queue.capture_observation('external',tmp_path/'different',2,child.pid,receipt)
    finally:child.terminate();child.wait(timeout=5)
