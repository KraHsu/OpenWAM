#!/usr/bin/env python3
"""Fill explicitly selected idle GPUs from a frozen, pre-reviewed dev queue.

Existing processes are observed, never adopted, stopped or restarted. New jobs
use the resource guardian, a shared GPU lease and an immutable output claim.
No retries, candidate generation, held-out evaluation or training are scheduled.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def process_identity(pid):
    """PID alone is insufficient: Linux may reuse it after a process exits."""
    try:
        root = Path('/proc') / str(pid)
        stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X'):
            return None
        command = (root/'cmdline').read_bytes()
        return dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=int(stat[19]), command_sha256=hashlib.sha256(command).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    return isinstance(identity, dict) and process_identity(identity['pid']) == identity


def receipt_pid(receipt, field):
    for component in field.split('.'):
        receipt = receipt[component]
    return receipt


def capture_observation(job_id, output, gpu, pid, receipt, pid_field='pid'):
    identity = process_identity(pid)
    receipt = Path(receipt).resolve()
    if identity is None or receipt_pid(read(receipt), pid_field) != pid:
        raise ValueError('Observation needs a live exact PID and its existing launch receipt')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if str(Path(output).resolve()).encode() not in command:
        raise ValueError('Observed process command does not identify this output')
    return dict(job_id=job_id, mode='observe', output=str(Path(output).resolve()), gpu=gpu,
                process=identity, launch_receipt=dict(path=str(receipt), sha256=file_hash(receipt), pid_field=pid_field))


def gpu_occupancy(gpus):
    options = dict(text=True, timeout=20)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], **options)
    by_uuid = {row.split(',')[1].strip(): int(row.split(',')[0]) for row in rows.splitlines() if row.strip()}
    if not set(gpus).issubset(by_uuid.values()):
        raise RuntimeError('A selected GPU is not present; no jobs launched')
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], **options)
    busy = {gpu: [] for gpu in gpus}
    for row in rows.splitlines():
        if row.strip():
            uuid, pid = (part.strip() for part in row.split(','))
            if uuid not in by_uuid:
                raise RuntimeError('Unknown GPU/MIG identifier; inspect inventory before scheduling')
            if by_uuid[uuid] in busy:
                busy[by_uuid[uuid]].append(int(pid))
    return {gpu: sorted(set(pids)) for gpu, pids in busy.items()}


def option(command, flag, required=True):
    if command.count(flag) != 1:
        if not required and flag not in command:
            return None
        raise ValueError('Expected exactly one explicit '+flag)
    index = command.index(flag)
    if index + 1 == len(command):
        raise ValueError('Missing argument for '+flag)
    return command[index+1]


def validate_plan(plan):
    if plan.get('version') != 1 or not isinstance(plan.get('jobs'), list):
        raise ValueError('Explicit versioned job queue required')
    gpus = plan.get('gpus')
    if (not isinstance(gpus, list) or not gpus or len(gpus) != len(set(gpus))
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('Explicit distinct physical GPU indices required')
    if type(plan.get('poll_seconds')) not in (int, float) or not .01 <= plan['poll_seconds'] <= 60:
        raise ValueError('Polling interval must be positive and at most 60 seconds')
    for key in ('queue_root', 'lease_root'):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise ValueError('Absolute '+key+' required')
    ids, outputs = set(), []
    for job in plan['jobs']:
        jid = job.get('job_id')
        if (not isinstance(jid, str) or not jid or not all(c.isalnum() or c in '_-' for c in jid) or jid in ids):
            raise ValueError('Unique filesystem-safe job IDs required')
        ids.add(jid)
        if not isinstance(job.get('output'), str) or not Path(job['output']).is_absolute():
            raise ValueError('Absolute output path required')
        output = Path(job['output']).resolve()
        for other in outputs + [Path(plan['queue_root']).resolve(), Path(plan['lease_root']).resolve()]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError('Job outputs, queue and lease directories must be separate')
        outputs.append(output)
        if job.get('mode') == 'observe':
            identity = job.get('process', {})
            receipt = job.get('launch_receipt', {})
            if (job.get('gpu') not in gpus or type(identity.get('pid')) is not int or identity['pid'] <= 0
                    or type(identity.get('start_ticks')) is not int or not identity.get('boot_id')
                    or not is_sha(identity.get('command_sha256')) or not is_sha(receipt.get('sha256'))
                    or not isinstance(receipt.get('path'), str) or not Path(receipt['path']).is_absolute()
                    or not isinstance(receipt.get('pid_field'), str)):
                raise ValueError('Observed job needs PID identity, selected GPU and a hashed launch receipt')
        elif job.get('mode') == 'queued':
            command = job.get('command')
            if (not isinstance(command, list) or len(command) < 2 or not all(isinstance(v, str) and v for v in command)
                    or not Path(command[0]).is_absolute() or not Path(command[1]).is_absolute()
                    or Path(command[1]).name != 'search_robotwin_constraint_counterexamples.py'
                    or not Path(job.get('cwd', '')).is_absolute()
                    or option(command, '--split') != 'dev' or '--holdout-release' in command
                    or '--check-only' in command or Path(option(command, '--output')).resolve() != output):
                raise ValueError('Only explicit native development-discovery commands are allowed')
            frozen = job.get('frozen_files')
            if (not isinstance(frozen, dict) or not frozen or
                    any(not Path(path).is_absolute() or not is_sha(value) for path, value in frozen.items())):
                raise ValueError('Queued jobs require immutable input/code file hashes')
            required = [command[1]] + [option(command, flag) for flag in
                         ('--reference', '--reference-manifest', '--template', '--split-plan')]
            required += [value for flag in ('--model-candidates', '--feedback')
                         if (value := option(command, flag, required=False)) is not None]
            if any(not Path(path).is_absolute() or path not in frozen for path in required):
                raise ValueError('Every command input and runner source must be explicitly hashed')
            env = job.get('environment', {})
            if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise ValueError('Environment must contain string keys and values')
            allowed = job.get('eligible_gpus', gpus)
            if not allowed or len(allowed) != len(set(allowed)) or not set(allowed).issubset(gpus):
                raise ValueError('Job GPUs must be a nonempty subset of selected GPUs')
        else:
            raise ValueError('Job mode must be queued or observe')
    return plan


def check_inputs(job):
    if job['mode'] == 'observe':
        spec = job['launch_receipt']
        if file_hash(spec['path']) != spec['sha256'] or receipt_pid(read(spec['path']), spec['pid_field']) != job['process']['pid']:
            raise ValueError('Observed launch receipt changed')
    else:
        for path, expected in job['frozen_files'].items():
            if file_hash(path) != expected:
                raise ValueError('Frozen input/source changed: '+path)


def reported_completion(output):
    """Scheduling status only; acceptance still requires independent verification."""
    output = Path(output)
    try:
        run, summary = read(output/'run.json'), read(output/'counterexample_summary.json')
    except FileNotFoundError:
        return False
    counts = summary.get('counts', {})
    return (run.get('status') == 'outcomes_complete' and summary.get('complete') is True
            and type(counts.get('planned')) is int and counts['planned'] > 0
            and counts.get('received') == counts['planned'])


class Queue:
    def __init__(self, plan):
        self.plan = validate_plan(plan)
        self.root = Path(plan['queue_root'])
        self.stop = threading.Event()
        self.registry = ProcessRegistry()
        self.running = {}
        self.statuses = {}

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner_lock = (self.root/'owner.lock').open('a')
        fcntl.flock(self.owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {str(path): file_hash(path) for path in (Path(__file__), ROOT/'benchmarks/utils/owned_processes.py',
                                                          ROOT/'benchmarks/utils/resource_process.py')}
        frozen = dict(plan=self.plan, scheduler_sources=sources)
        path = self.root/'frozen_queue.json'
        if path.exists():
            if read(path) != frozen:
                raise ValueError('Frozen queue or scheduler source changed; use a new queue directory')
        else:
            atomic(path, frozen)
        self.queue_sha256 = digest(frozen)
        Path(self.plan['lease_root']).mkdir(parents=True, exist_ok=True)
        for job in self.plan['jobs']:
            directory = self.root/'jobs'/job['job_id']
            directory.mkdir(parents=True, exist_ok=True)
            state = directory/'state.json'
            self.statuses[job['job_id']] = read(state) if state.exists() else dict(phase='pending')

    def state(self, job, phase, **extra):
        value = dict(phase=phase, updated_unix=time.time(), **extra)
        self.statuses[job['job_id']] = value
        atomic(self.root/'jobs'/job['job_id']/'state.json', value)

    def observe(self, job, identity, gpu):
        if same_process(identity):
            self.state(job, 'observing_existing', process=identity, gpu=gpu)
            return True
        complete = reported_completion(job['output'])
        self.state(job, 'completed_existing' if complete else 'blocked_existing',
                   reason='Process exited or identity changed; never restart', process=identity, gpu=gpu)
        return False

    def lease(self, gpu):
        path = Path(self.plan['lease_root'])/f'gpu_{gpu}.lock'
        handle = path.open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        handle.seek(0)
        old = handle.read()
        if old and same_process(json.loads(old).get('process')):
            handle.close()
            return None
        return handle

    def launch(self, job, gpu, lease):
        check_inputs(job)
        output = Path(job['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()  # Atomic exclusion: an existing/failed run is never resampled.
        directory = self.root/'jobs'/job['job_id']
        env = dict(os.environ, **job.get('environment', {}))
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        intent = dict(queue_sha256=self.queue_sha256, job_id=job['job_id'], gpu=gpu,
                      owner=process_identity(os.getpid()), command=job['command'], cwd=job['cwd'],
                      environment=dict(job.get('environment', {}), CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1'),
                      created_unix=time.time(), frozen_files=job['frozen_files'])
        atomic(output/'.dev_queue_claim.json', intent)
        atomic(directory/'launch_intent.json', intent)
        log = (directory/'native.log').open('x')
        try:
            child = spawn_resource(job['command'], cwd=job['cwd'], env=env, stdout=log)
        except BaseException:
            log.close()
            raise
        self.registry.add(child, log)
        identity = process_identity(child.pid)
        self.running[job['job_id']] = dict(child=child, log=log, lease=lease, gpu=gpu, process=identity)
        receipt = dict(intent, process=identity, guardian_pid=child.pid, scheduler_sources=read(self.root/'frozen_queue.json')['scheduler_sources'])
        atomic(directory/'launch.json', receipt)
        lease.seek(0); lease.truncate()
        lease.write(json.dumps(dict(process=identity, queue_sha256=self.queue_sha256, job_id=job['job_id'])))
        lease.flush()
        self.state(job, 'running_owned', gpu=gpu, process=identity)

    def step(self):
        jobs = self.plan['jobs']
        occupied = set()
        for job in jobs:
            jid = job['job_id']
            if jid in self.running:
                row = self.running[jid]
                code = row['child'].poll()
                if code is None:
                    occupied.add(row['gpu'])
                    continue
                if row['child'].stdin is not None:
                    row['child'].stdin.close()
                row['log'].close(); row['lease'].close()
                complete = code == 0 and reported_completion(job['output'])
                self.state(job, 'completed' if complete else 'failed', returncode=code, gpu=row['gpu'])
                del self.running[jid]
                continue
            phase = self.statuses[jid]['phase']
            if phase in ('completed', 'completed_existing', 'failed', 'blocked_existing', 'blocked_preflight', 'interrupted'):
                continue
            try:
                if job['mode'] == 'observe':
                    check_inputs(job)
                    if self.observe(job, job['process'], job['gpu']):
                        occupied.add(job['gpu'])
                elif (self.root/'jobs'/jid/'launch.json').exists():
                    receipt = read(self.root/'jobs'/jid/'launch.json')
                    if self.observe(job, receipt['process'], receipt['gpu']):
                        occupied.add(receipt['gpu'])
                elif Path(job['output']).exists() or (self.root/'jobs'/jid/'launch_intent.json').exists():
                    self.state(job, 'completed_existing' if reported_completion(job['output']) else 'blocked_existing',
                               reason='Output/launch intent exists; no retries or ownership transfer')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.state(job, 'blocked_preflight', reason=f'{type(exc).__name__}: {exc}')
        pending = [job for job in jobs if job['mode'] == 'queued' and self.statuses[job['job_id']]['phase'] == 'pending']
        inventory, inventory_error = {}, None
        if pending:
            try:
                inventory = gpu_occupancy(self.plan['gpus'])
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # A monitoring outage blocks new allocation, not already
                # running simulations. Their guardian ownership stays intact.
                inventory_error = f'{type(exc).__name__}: {exc}'
        busy = occupied | {gpu for gpu, pids in inventory.items() if pids}
        for job in pending if inventory_error is None else []:
            for gpu in job.get('eligible_gpus', self.plan['gpus']):
                if gpu in busy:
                    continue
                lease = self.lease(gpu)
                if lease is None:
                    continue
                try:
                    # Recheck after obtaining the shared lease; never kill occupants.
                    try:
                        occupants = gpu_occupancy([gpu])[gpu]
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        inventory_error = f'{type(exc).__name__}: {exc}'
                        lease.close(); busy.add(gpu)
                        continue
                    if occupants:
                        lease.close(); busy.add(gpu)
                        continue
                    self.launch(job, gpu, lease)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if job['job_id'] not in self.running:
                        lease.close()
                    self.state(job, 'blocked_preflight', reason=f'{type(exc).__name__}: {exc}')
                busy.add(gpu)
                break
        active = [s for s in self.statuses.values() if s['phase'] in ('pending', 'running_owned', 'observing_existing')]
        waiting = any(s['phase'] == 'pending' for s in active)
        phase = 'running' if self.running else 'waiting_for_free_selected_gpus' if waiting else 'observing' if active else 'queue_exhausted'
        atomic(self.root/'state.json', dict(phase=phase, pid=os.getpid(), queue_sha256=self.queue_sha256,
                                          updated_unix=time.time(), jobs=self.statuses, gpu_inventory=inventory,
                                          gpu_inventory_error=inventory_error,
                                          idle_reason=None if active else 'No remaining pre-reviewed jobs; no synthetic GPU load'))
        return bool(active)

    def close(self):
        # Only children launched by this Queue are in this registry. Observed
        # jobs and unrelated GPU processes are never signalled, even on Ctrl-C.
        self.registry.terminate_all()
        for jid, row in self.running.items():
            row['lease'].close()
            job = next(job for job in self.plan['jobs'] if job['job_id'] == jid)
            self.state(job, 'interrupted', reason='Owned guardian cleaned up; no automatic retry', gpu=row['gpu'])
        self.running.clear()
        if hasattr(self, 'owner_lock'):
            self.owner_lock.close()

    def run(self):
        try:
            self.prepare()
            while not self.stop.is_set() and self.step():
                self.stop.wait(self.plan['poll_seconds'])
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--check-only', action='store_true')
    observe = commands.add_parser('capture-observation')
    for name in ('job-id', 'output', 'receipt'):
        observe.add_argument('--'+name, required=True)
    observe.add_argument('--gpu', type=int, required=True)
    observe.add_argument('--pid', type=int, required=True)
    observe.add_argument('--pid-field', default='pid')
    args = parser.parse_args()
    if args.command == 'capture-observation':
        print(json.dumps(capture_observation(args.job_id, args.output, args.gpu, args.pid, args.receipt, args.pid_field), indent=2))
        return
    plan = validate_plan(read(args.plan))
    if args.check_only:
        for job in plan['jobs']:
            check_inputs(job)
        print(json.dumps(dict(valid=True, jobs=len(plan['jobs']), gpus=plan['gpus'], gpu_started=False, plan_sha256=digest(plan))))
        return
    queue = Queue(plan)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: queue.stop.set())
    queue.run()


if __name__ == '__main__':
    main()
