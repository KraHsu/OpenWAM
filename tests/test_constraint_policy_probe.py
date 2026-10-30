"""Shared-card probes retain resource ownership without using a GPU in tests."""
import copy
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource
from scripts import run_constraint_policy_probe as probe


def plan(tmp_path, shared=True):
    model=tmp_path/'model';model.mkdir();weights=model/'checkpoint_step_42970.safetensors';weights.write_bytes(b'fake CPU-test weights')
    receipt=tmp_path/'checkpoint.json';receipt.write_text(json.dumps(dict(size_bytes=weights.stat().st_size,sha256=probe.sha(weights))))
    config=tmp_path/'deploy.yml';config.write_text('checkpoint: fixed\n')
    return dict(run_root=str(tmp_path/'run'),python=sys.executable,robotwin_python=sys.executable,
                openwam_root=str(tmp_path),rax_root=str(tmp_path/'rax'),robotwin_root=str(tmp_path/'native'),vulkan_icd='/fixed/icd.json',
                server_gpu=3,client_gpu=3 if shared else 2,allow_shared_gpu=shared,lease_root=str(tmp_path/'leases'),port=18882,
                checkpoint_receipt=str(receipt),model_dir=str(model),checkpoint_name=weights.name,server_config=str(config),
                jobs=[dict(task='lift_pot',job_id=f'lift_seed{seed}',reference=f'/refs/seed{seed}.json',source_manifest='/refs/manifest.json') for seed in (10008,10010)])


class FakeChild:
    def __init__(self,pid,code):self.pid=pid;self.returncode=code;self.stdin=io.BytesIO()
    def poll(self):return self.returncode


def fake_runtime(monkeypatch, p):
    children=[];calls=[];registries=[]
    class Registry:
        def __init__(self):self.children=[];self.terminated=False;registries.append(self)
        def add(self,child,stream):self.children.append((child,stream))
        def terminate_all(self,timeout=15):
            self.terminated=True
            for child,stream in self.children:
                if child.poll() is None:child.returncode=-15
                stream.close()
    def spawn(command,**kwargs):
        server='--readiness-token' in command
        child=FakeChild(800000+len(children),None if server else 0)
        children.append(child);calls.append((command,kwargs))
        if not server:
            output=Path(command[command.index('--output')+1]);output.mkdir(parents=True)
            (output/'result.json').write_text(json.dumps(dict(complete=True,native_success=False,commanded_constraints={'passed':False})))
            (output/'model_candidates.json').write_text('{}')
        return child
    class Client:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def ping(self):return json.loads((Path(p['run_root'])/'model_server_receipt.json').read_text())
    monkeypatch.setattr(probe,'ProcessRegistry',Registry)
    monkeypatch.setattr(probe,'spawn_resource',spawn)
    monkeypatch.setattr(probe,'WSPolicyClient',Client)
    monkeypatch.setattr(probe,'gpu_processes',lambda *_:[])
    monkeypatch.setattr(probe,'gpu_occupancy',lambda gpus:{gpu:[] for gpu in gpus})
    monkeypatch.setattr(probe,'gpu_memory',lambda gpu:dict(gpu=gpu,total_mib=140000,free_mib=120000))
    return children,calls,registries


def test_default_still_requires_separate_cards_and_shared_needs_explicit_lease_and_budget(tmp_path):
    p=plan(tmp_path);p.pop('allow_shared_gpu')
    with pytest.raises(ValueError,match='allow_shared_gpu'):probe.validate_plan(p)
    p['allow_shared_gpu']=True;p['lease_root']='relative'
    with pytest.raises(ValueError,match='lease_root'):probe.validate_plan(p)
    p['lease_root']=str(tmp_path/'leases');p['shared_gpu_native_headroom_mib']=1
    with pytest.raises(ValueError,match='at least'):probe.validate_plan(p)


def test_same_task_two_scenes_use_distinct_outputs_and_same_card_owned_fds(tmp_path,monkeypatch):
    p=plan(tmp_path);children,calls,registries=fake_runtime(monkeypatch,p)
    probe.run(p)
    assert len(children)==3 and registries[0].terminated
    assert {kw['env']['CUDA_VISIBLE_DEVICES'] for _,kw in calls}=={'3'}
    assert all(len(kw['pass_fds'])==1 for _,kw in calls)
    state=json.loads((Path(p['run_root'])/'policy_probe_state.json').read_text())
    assert state['phase']=='complete' and len(state['memory_checks'])==3
    assert [x['job_id'] for x in state['completed']]==['lift_seed10008','lift_seed10010']
    assert all(x['native_success'] is False and x['commanded_constraint_pass'] is False for x in state['completed'])
    assert len({x['candidates'] for x in state['completed']})==2
    with pytest.raises(ValueError,match='Existing policy trace'):probe.run(p)
    assert len(children)==3


def test_split_mode_retains_distinct_gpu_launch_and_skips_shared_memory_checks(tmp_path,monkeypatch):
    p=plan(tmp_path,shared=False);p.pop('allow_shared_gpu');p.pop('lease_root');p['jobs']=p['jobs'][:1]
    _,calls,_=fake_runtime(monkeypatch,p)
    def forbidden(*a,**kw):pytest.fail('Legacy split mode must not use shared-card memory checks')
    monkeypatch.setattr(probe,'gpu_memory',forbidden)
    probe.run(p)
    assert [kw['env']['CUDA_VISIBLE_DEVICES'] for _,kw in calls]==['3','2']
    assert all(kw['pass_fds']==() for _,kw in calls)


def test_foreign_gpu_process_blocks_before_server_without_killing_it(tmp_path,monkeypatch):
    p=plan(tmp_path);children,_,registries=fake_runtime(monkeypatch,p)
    monkeypatch.setattr(probe,'gpu_occupancy',lambda gpus:{3:[77777]})
    with pytest.raises(RuntimeError,match='foreign processes'):probe.run(p)
    assert not children and registries[0].children==[]
    assert 'failed'==json.loads((Path(p['run_root'])/'policy_probe_state.json').read_text())['phase']


@pytest.mark.parametrize('total,free',[(70000,70000),(140000,60000)])
def test_insufficient_total_or_initial_free_memory_never_starts_model(tmp_path,monkeypatch,total,free):
    p=plan(tmp_path);children,_,_=fake_runtime(monkeypatch,p)
    monkeypatch.setattr(probe,'gpu_memory',lambda gpu:dict(gpu=gpu,total_mib=total,free_mib=free))
    with pytest.raises(RuntimeError,match='Insufficient'):probe.run(p)
    assert not children


def test_insufficient_headroom_after_model_load_cleans_owned_server_no_native_start(tmp_path,monkeypatch):
    p=plan(tmp_path);children,_,registries=fake_runtime(monkeypatch,p)
    memories=iter([120000,10000]);monkeypatch.setattr(probe,'gpu_memory',lambda gpu:dict(gpu=gpu,total_mib=140000,free_mib=next(memories)))
    with pytest.raises(RuntimeError,match='Insufficient'):probe.run(p)
    assert len(children)==1 and children[0].returncode==-15 and registries[0].terminated
    lease=probe.SharedGpuLease(p);lease.acquire();lease.close()


def test_shared_queue_lease_collision_blocks_probe_without_model_start(tmp_path,monkeypatch):
    p=plan(tmp_path);children,_,_=fake_runtime(monkeypatch,p)
    directory=Path(p['lease_root']);directory.mkdir()
    with (directory/'gpu_3.lock').open('a+') as other:
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):probe.run(p)
    assert not children


def test_frozen_inputs_checked_before_resource_allocation_and_duplicate_jobs_rejected(tmp_path,monkeypatch):
    p=plan(tmp_path);children,_,_=fake_runtime(monkeypatch,p)
    p['frozen_inputs']={p['server_config']:'0'*64}
    with pytest.raises(ValueError,match='Frozen model-probe'):probe.run(p)
    assert not children
    p=copy.deepcopy(p);p['jobs'][1]['job_id']=p['jobs'][0]['job_id']
    with pytest.raises(ValueError,match='distinct job_id'):probe.validate_plan(p)


def test_guardian_inherits_lease_until_cleanup_and_ancestry_excludes_foreign_process(tmp_path,monkeypatch):
    p=plan(tmp_path);lease=probe.SharedGpuLease(p);lease.acquire()
    pidfile=tmp_path/'native.pid'
    command=[sys.executable,'-c','import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(120)']
    registry=ProcessRegistry();child=spawn_resource(command,pass_fds=(lease.handle.fileno(),));registry.add(child)
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert pidfile.exists();native_pid=int(pidfile.read_text());identity=probe.process_identity(child.pid)
        assert probe.owned_gpu_pid(native_pid,[identity])
        assert not probe.owned_gpu_pid(unrelated.pid,[identity])
        monkeypatch.setattr(probe,'gpu_occupancy',lambda _:{3:[native_pid]})
        monkeypatch.setattr(probe,'gpu_memory',lambda gpu:dict(gpu=gpu,total_mib=140000,free_mib=50000))
        assert probe.check_shared_capacity(p,guardians=[identity],native_ready=True)['stage']=='before_native_client'
        monkeypatch.setattr(probe,'gpu_occupancy',lambda _:{3:[native_pid,unrelated.pid]})
        with pytest.raises(RuntimeError,match='foreign processes'):probe.check_shared_capacity(p,guardians=[identity],native_ready=True)
        # Parent FD close must not unlock a still-live guardian's lease.
        lease.close();other=probe.SharedGpuLease(p)
        with pytest.raises(BlockingIOError):other.acquire()
        registry.terminate_all();assert unrelated.poll() is None
        other.acquire();other.close()
    finally:
        registry.terminate_all();lease.close();unrelated.terminate();unrelated.wait(timeout=5)
