#!/usr/bin/env python3
"""Own policy traces, with explicit leased same-GPU server/native execution."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource
from benchmarks.utils.transport import WSPolicyClient
from scripts.robotwin_training_gate import gpu_processes
from scripts.robotwin_constraint_policy_candidates import put, sha
from scripts.robotwin_dev_queue import gpu_occupancy, process_identity, same_process

SHARED_MEMORY_DEFAULTS = dict(shared_gpu_min_total_mib=80*1024,
                              shared_gpu_min_free_mib=64*1024,
                              shared_gpu_native_headroom_mib=16*1024)


def shared_gpu(plan):
    return plan['server_gpu'] == plan['client_gpu']


def job_name(job):
    return job.get('job_id', job['task'])


def validate_plan(plan):
    for key in ('server_gpu', 'client_gpu'):
        if type(plan.get(key)) is not int or plan[key] < 0:
            raise ValueError('Explicit nonnegative GPU indices required')
    if type(plan.get('allow_shared_gpu', False)) is not bool:
        raise ValueError('allow_shared_gpu must be an explicit boolean')
    if shared_gpu(plan):
        if not plan.get('allow_shared_gpu', False):
            raise ValueError('Separate model and native GPU ownership required unless allow_shared_gpu=true')
        if not isinstance(plan.get('lease_root'), str) or not Path(plan['lease_root']).is_absolute():
            raise ValueError('Shared mode requires the same absolute lease_root as the dev queues')
        for key, minimum in SHARED_MEMORY_DEFAULTS.items():
            value = plan.get(key, minimum)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be at least {minimum} MiB')
        values = [plan.get(key, value) for key, value in SHARED_MEMORY_DEFAULTS.items()]
        if not values[0] >= values[1] >= values[2]:
            raise ValueError('Shared total/initial-free/native-headroom thresholds must be ordered')
    if not isinstance(plan.get('jobs'), list) or not plan['jobs']:
        raise ValueError('Explicit finite policy-probe job list required')
    names = []
    for job in plan['jobs']:
        name = job_name(job)
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in '_-' for c in name):
            raise ValueError('Filesystem-safe task/job_id required')
        if name in names:
            raise ValueError('Repeated tasks require distinct job_id values, such as task_seed10008')
        names.append(name)
    for path, expected in plan.get('frozen_inputs', {}).items():
        if not Path(path).is_absolute() or len(expected) != 64 or set(expected)-set('0123456789abcdef'):
            raise ValueError('Frozen inputs require absolute paths and SHA256 digests')
    return plan


def gpu_memory(gpu):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    for line in raw.splitlines():
        index, total, free = (part.strip() for part in line.split(','))
        if int(index) == gpu:
            return dict(gpu=gpu, total_mib=int(total), free_mib=int(free))
    raise ValueError('Selected shared GPU is absent')


def owned_gpu_pid(pid, guardians):
    """GPU contexts belong to command children, not usually guardian PIDs."""
    live = {identity['pid'] for identity in guardians if same_process(identity)}
    visited = set()
    while pid > 1 and pid not in visited:
        if pid in live:
            return True
        visited.add(pid)
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            pid = int(stat[1])
        except (FileNotFoundError, ProcessLookupError):
            return False
    return False


def check_shared_capacity(plan, *, guardians=(), native_ready=False):
    gpu = plan['server_gpu']
    foreign = [pid for pid in gpu_occupancy([gpu])[gpu] if not owned_gpu_pid(pid, guardians)]
    if foreign:
        raise RuntimeError(f'Shared GPU {gpu} has foreign processes: {foreign}; no foreign process is terminated')
    memory = gpu_memory(gpu)
    key = 'shared_gpu_native_headroom_mib' if native_ready else 'shared_gpu_min_free_mib'
    minimum_free = plan.get(key, SHARED_MEMORY_DEFAULTS[key])
    minimum_total = plan.get('shared_gpu_min_total_mib', SHARED_MEMORY_DEFAULTS['shared_gpu_min_total_mib'])
    if memory['total_mib'] < minimum_total or memory['free_mib'] < minimum_free:
        raise RuntimeError(f'Insufficient shared GPU memory: {memory}; require total>={minimum_total}, free>={minimum_free} MiB')
    return dict(memory, stage='before_native_client' if native_ready else 'before_model_server',
                minimum_total_mib=minimum_total, minimum_free_mib=minimum_free)


class SharedGpuLease:
    """Inherited FDs keep the shared queue lease until all guardians exit."""
    def __init__(self, plan):
        self.plan, self.handle, self.wrote_owner = plan, None, False

    def acquire(self):
        root = Path(self.plan['lease_root']); root.mkdir(parents=True, exist_ok=True)
        self.handle = (root/f"gpu_{self.plan['server_gpu']}.lock").open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.seek(0); old = self.handle.read()
            if old and same_process(json.loads(old).get('process')):
                raise RuntimeError('Shared GPU lease still belongs to a live process')
            owner = dict(kind='shared_gpu_policy_probe', process=process_identity(os.getpid()),
                         run_root=self.plan['run_root'], gpu=self.plan['server_gpu'], created_unix=time.time(),
                         inherited_by_guardians=True)
            self.handle.seek(0); self.handle.truncate(); self.handle.write(json.dumps(owner)); self.handle.flush()
            self.wrote_owner = True
            return owner
        except BaseException:
            self.close(); raise

    def close(self):
        if self.handle is not None:
            if self.wrote_owner:
                self.handle.seek(0); self.handle.truncate()
                self.handle.write(json.dumps(dict(process=None, released_unix=time.time(), run_root=self.plan['run_root'])))
                self.handle.flush()
                self.wrote_owner = False
            # Never LOCK_UN: inherited guardian descriptors must retain the
            # same open-file-description lock until their cleanup completes.
            self.handle.close(); self.handle = None


def run(plan):
    validate_plan(plan)
    root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'policy_probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=root/'policy_probe_plan.json'
        if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('Frozen policy probe plan changed')
        put(frozen,plan)
        for job in plan['jobs']:
            if (root/'model_traces'/job_name(job)).exists():
                raise ValueError('Existing policy trace requires inspection/new version; no silent resampling')
        stop=threading.Event(); registry=ProcessRegistry()
        old_signals={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        for sig in old_signals:signal.signal(sig,lambda *_:stop.set())
        lease=SharedGpuLease(plan) if shared_gpu(plan) else None
        guardians=[];memory_checks=[]
        def state(phase,**kw):put(root/'policy_probe_state.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),
            shared_gpu=shared_gpu(plan),memory_checks=memory_checks,**kw))
        def launch(command,gpu,log):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',USE_TF='0',USE_FLAX='0',
                     ROBOTWIN_PATH=plan['robotwin_root'],VK_ICD_FILENAMES=plan['vulkan_icd'],VK_DRIVER_FILES=plan['vulkan_icd'])
            stream=(root/log).open('a')
            try:
                child=spawn_resource(command,cwd=plan['openwam_root'],env=env,stdout=stream,
                                     pass_fds=(lease.handle.fileno(),) if lease is not None else ())
            except BaseException:
                stream.close();raise
            registry.add(child,stream)
            identity=process_identity(child.pid)
            if identity is not None:guardians.append(identity)
            return child
        try:
            for path,expected in plan.get('frozen_inputs',{}).items():
                if sha(path)!=expected:raise ValueError('Frozen model-probe input/source changed: '+path)
            if lease is not None:
                receipt=lease.acquire();memory_checks.append(check_shared_capacity(plan));put(root/'gpu_lease_receipt.json',receipt)
            elif gpu_processes([plan['server_gpu'],plan['client_gpu']]):
                raise RuntimeError('Selected policy probe GPUs are occupied')
            state('verifying_model')
            checkpoint=json.loads(Path(plan['checkpoint_receipt']).read_text())
            weights=Path(plan['model_dir'])/plan['checkpoint_name']
            if weights.stat().st_size!=checkpoint['size_bytes'] or sha(weights)!=checkpoint['sha256']:
                raise ValueError('Model weights differ from recorded final checkpoint')
            token=secrets.token_hex(24)
            cmd=[plan['python'],str(Path(plan['openwam_root'])/'scripts/deploy.py'),'--config',plan['server_config'],
                 '--ckpt-dir',plan['model_dir'],'--ckpt-name',plan['checkpoint_name'],'--device','cuda:0',
                 '--host','127.0.0.1','--port',str(plan['port']),'--readiness-token',token]
            server=launch(cmd,plan['server_gpu'],'model_server.log')
            receipt=dict(checkpoint_sha256=checkpoint['sha256'],config_sha256=sha(plan['server_config']),
                         port=plan['port'],readiness_token=token,guardian_pid=server.pid,
                         server_gpu=plan['server_gpu'],client_gpu=plan['client_gpu'],shared_gpu=shared_gpu(plan),started_unix=time.time())
            put(root/'model_server_receipt.json',receipt);state('loading_model',server_pid=server.pid)
            deadline=time.monotonic()+600
            while True:
                if stop.is_set() or server.poll() is not None:raise RuntimeError('Owned policy server exited/interrupted')
                try:
                    with WSPolicyClient(f"ws://127.0.0.1:{plan['port']}",timeout=3,open_timeout=2) as client:
                        ready=client.ping().get('readiness_token')==token
                    if ready:break
                except (OSError,TimeoutError,RuntimeError):pass
                if time.monotonic()>deadline:raise TimeoutError('Policy model startup timed out')
                stop.wait(2)
            completed=[]
            for job in plan['jobs']:
                if stop.is_set():raise RuntimeError('Policy trace collection interrupted')
                if lease is not None:memory_checks.append(check_shared_capacity(plan,guardians=guardians,native_ready=True))
                output=root/'model_traces'/job_name(job)
                if output.exists():raise ValueError('Existing policy trace requires inspection/new version; no silent resampling')
                cmd=[plan['robotwin_python'],str(Path(plan['openwam_root'])/'scripts/robotwin_constraint_policy_candidates.py'),
                     '--reference',job['reference'],'--source-manifest',job['source_manifest'],
                     '--rax-root',plan['rax_root'],'--checkpoint-receipt',plan['checkpoint_receipt'],
                     '--server-config',plan['server_config'],'--server-receipt',str(root/'model_server_receipt.json'),
                     '--port',str(plan['port']),'--output',str(output)]
                child=launch(cmd,plan['client_gpu'],job_name(job)+'_model_client.log')
                state('collecting_model_trajectory',task=job['task'],job_id=job_name(job),client_pid=child.pid,completed=completed)
                deadline=time.monotonic()+1800
                while child.poll() is None:
                    if stop.wait(1) or time.monotonic()>deadline:raise TimeoutError('Policy trace interrupted/timed out')
                    if server.poll() is not None:raise RuntimeError('Owned model server exited during policy collection')
                if child.returncode:raise RuntimeError(f"Policy trace failed for {job_name(job)}; see client log")
                result=json.loads((output/'result.json').read_text())
                if not result['complete']:raise ValueError('Policy collector returned incomplete trace')
                completed.append(dict(task=job['task'],job_id=job_name(job),candidates=str(output/'model_candidates.json'),
                                      native_success=result['native_success'],
                                      commanded_constraint_pass=result['commanded_constraints']['passed']))
            state('complete',completed=completed)
        except BaseException as exc:
            state('failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            try:registry.terminate_all(timeout=15)
            finally:
                if lease is not None:lease.close()
                for sig,handler in old_signals.items():signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    run(json.loads(p.parse_args().plan.read_text()))
