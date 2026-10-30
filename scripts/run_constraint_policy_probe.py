#!/usr/bin/env python3
"""Own one policy server and serial genuine policy traces for constraint search."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource
from benchmarks.utils.transport import WSPolicyClient
from scripts.robotwin_training_gate import gpu_processes
from scripts.robotwin_constraint_policy_candidates import put, sha


def run(plan):
    root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'policy_probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=root/'policy_probe_plan.json'
        if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('Frozen policy probe plan changed')
        put(frozen,plan)
        if plan['server_gpu']==plan['client_gpu']:raise ValueError('Separate model and native GPU ownership required')
        busy=gpu_processes([plan['server_gpu'],plan['client_gpu']])
        if busy:raise RuntimeError('Selected policy probe GPUs are occupied')
        stop=threading.Event(); registry=ProcessRegistry()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
        def state(phase,**kw):put(root/'policy_probe_state.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))
        def launch(command,gpu,log):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',USE_TF='0',USE_FLAX='0',
                     ROBOTWIN_PATH=plan['robotwin_root'],VK_ICD_FILENAMES=plan['vulkan_icd'],VK_DRIVER_FILES=plan['vulkan_icd'])
            stream=(root/log).open('a'); child=spawn_resource(command,cwd=plan['openwam_root'],env=env,stdout=stream)
            registry.add(child,stream);return child
        try:
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
                         server_gpu=plan['server_gpu'],started_unix=time.time())
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
                output=root/'model_traces'/job['task']
                if output.exists():raise ValueError('Existing policy trace requires inspection/new version; no silent resampling')
                cmd=[plan['robotwin_python'],str(Path(plan['openwam_root'])/'scripts/robotwin_constraint_policy_candidates.py'),
                     '--reference',job['reference'],'--source-manifest',job['source_manifest'],
                     '--rax-root',plan['rax_root'],'--checkpoint-receipt',plan['checkpoint_receipt'],
                     '--server-config',plan['server_config'],'--server-receipt',str(root/'model_server_receipt.json'),
                     '--port',str(plan['port']),'--output',str(output)]
                child=launch(cmd,plan['client_gpu'],job['task']+'_model_client.log')
                state('collecting_model_trajectory',task=job['task'],client_pid=child.pid,completed=completed)
                deadline=time.monotonic()+1800
                while child.poll() is None:
                    if stop.wait(1) or time.monotonic()>deadline:raise TimeoutError('Policy trace interrupted/timed out')
                if child.returncode:raise RuntimeError(f"Policy trace failed for {job['task']}; see client log")
                result=json.loads((output/'result.json').read_text())
                if not result['complete']:raise ValueError('Policy collector returned incomplete trace')
                completed.append(dict(task=job['task'],candidates=str(output/'model_candidates.json'),
                                      native_success=result['native_success'],
                                      commanded_constraint_pass=result['commanded_constraints']['passed']))
            state('complete',completed=completed)
        except BaseException as exc:
            state('failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:registry.terminate_all(timeout=15)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    run(json.loads(p.parse_args().plan.read_text()))
