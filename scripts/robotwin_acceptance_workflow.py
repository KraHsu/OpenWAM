#!/usr/bin/env python3
"""Wait for C2R completion, validate held-out clean variations, publish, train.

The current benchmark and source dataset are read-only dependencies. A failed
acceptance pilot stops this new workflow; it never weakens the acceptance rules
or launches training on the old labels. All children belong to this workflow.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.utils.owned_processes import spawn_resource  # noqa: E402
from scripts.robotwin_training_gate import gpu_processes, validate_plan as validate_training_plan  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(threading.get_ident()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bind(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen workflow input changed: {path}')
    else:
        write(path, value)


def check_publication_hold(root):
    """An explicit research hold prevents old pilot evidence releasing training."""
    path = Path(root) / 'publication_hold.json'
    if not path.exists():
        return
    hold = read(path)
    if hold.get('version') != 1 or type(hold.get('active')) is not bool:
        raise ValueError('Invalid publication hold; inspect before publishing')
    if hold['active']:
        raise RuntimeError('Publication/training held: ' + str(hold.get('reason', 'additional validation required')))


def benchmark_complete(root, tasks):
    root = Path(root)
    if not (root / 'state.json').exists():
        return False
    state = read(root / 'state.json')
    if state.get('phase') == 'failed':
        raise RuntimeError('Previous benchmark failed; its incomplete results cannot release this workflow')
    if state.get('phase') != 'complete':
        return False
    summary = read(root / 'summary.json')
    rows = summary.get('tasks', [])
    if (summary.get('complete') is not True or summary.get('completed_episodes') != 500
            or summary.get('completed_tasks') != 50 or len(rows) != 50
            or {r['task'] for r in rows} != set(tasks)
            or any(r.get('completed_episodes') != 10 or r.get('complete') is not True for r in rows)):
        raise ValueError('Benchmark claims completion without all 50 × 10 outcomes')
    return True


def benchmark_stop_release(plan, receipt_path=None):
    """Validate an explicit user stop, or its previously frozen authorization.

    A stopped benchmark remains incomplete. This changes only the dependency
    release condition, with a durable receipt separate from the frozen plan.
    Reading/check-only never creates a release file or changes benchmark state.
    """
    frozen_path = Path(plan['run_root']) / 'benchmark_release.json'
    frozen = read(frozen_path) if frozen_path.exists() else None
    if receipt_path is None:
        if frozen is None:
            return None
        receipt_path = frozen['receipt_path']
    receipt_path = Path(receipt_path).resolve()
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get('version') != 1 or receipt.get('kind') != 'user_requested_benchmark_stop':
        raise ValueError('Expected an explicit user-requested benchmark stop receipt')
    for key in ('benchmark_root', 'run_root'):
        if not isinstance(receipt.get(key), str) or Path(receipt[key]).resolve() != Path(plan[key]).resolve():
            raise ValueError(f'Benchmark stop receipt has mismatched {key}')
    stopped = receipt.get('stopped_unix')
    if type(stopped) not in (int, float) or not math.isfinite(stopped) or stopped <= 0:
        raise ValueError('Benchmark stop receipt has invalid stop time')
    benchmark = Path(plan['benchmark_root'])
    if read(benchmark / 'state.json').get('phase') != 'stopped_by_user':
        raise ValueError('Benchmark stop receipt requires state stopped_by_user')
    summary_hash = hashlib.sha256((benchmark / 'summary.json').read_bytes()).hexdigest()
    if receipt.get('summary_sha256') != summary_hash:
        raise ValueError('Benchmark summary changed after the user stop')
    stop_raw = (benchmark / 'user_stop.json').read_bytes()
    if receipt.get('benchmark_stop_receipt_sha256') != hashlib.sha256(stop_raw).hexdigest():
        raise ValueError('Benchmark user-stop evidence hash changed')
    stop_evidence = json.loads(stop_raw)
    for key in ('version', 'kind', 'benchmark_root', 'run_root', 'stopped_unix', 'summary_sha256'):
        if stop_evidence.get(key) != receipt[key]:
            raise ValueError(f'Benchmark user-stop evidence differs in {key}')
    release = dict(version=1, receipt_path=str(receipt_path),
                   receipt_sha256=hashlib.sha256(raw).hexdigest(), receipt=receipt)
    if frozen is not None and frozen != release:
        raise ValueError('Frozen benchmark release changed')
    return release


def validate_plan(plan):
    if plan.get('version') != 1 or len(set(plan['tasks'])) != 50:
        raise ValueError('Expected a versioned 50-task workflow plan')
    validate_training_plan(plan['training'])
    if plan['gpus'] != plan['training']['gpus']:
        raise ValueError('Acceptance and subsequent training must use the same reserved GPU selection')
    root = Path(plan['run_root']).resolve()
    for key in ('benchmark_root', 'source_dataset_root'):
        other = Path(plan[key]).resolve()
        if root == other or root in other.parents or other in root.parents:
            raise ValueError('New run must be separate from the benchmark and existing data')
    for key in ('openwam_root', 'rax_root'):
        if root not in Path(plan[key]).resolve().parents:
            raise ValueError('New workflow requires isolated code snapshots under its own run root')
    if plan['training']['run_root'] != str(root / 'training_run'):
        raise ValueError('Training state must be isolated from workflow state')
    if Path(plan['training']['dataset_root']).resolve() != root / 'dataset':
        raise ValueError('New training dataset must be isolated at run_root/dataset')
    for key in ('openwam_root', 'rax_root', 'python'):
        if Path(plan['training'][key]).resolve() != Path(plan[key]).resolve():
            raise ValueError(f'Training {key} must match the frozen workflow')
    if plan['training'].get('require_acceptance_v2') is not True:
        raise ValueError('New training must require acceptance-v2 publication proof')
    if plan['training'].get('published_dataset_is_snapshot') is not True:
        raise ValueError('New training must use the accepted publication as its frozen snapshot')
    if plan.get('require_constraint_refinement') and not plan['training'].get('require_constraint_refinement'):
        raise ValueError('Constraint refinement evidence must also be required by training')
    if not set(plan['pilot_tasks']).issubset(plan['tasks']) or not plan['pilot_tasks']:
        raise ValueError('Explicit pilot tasks required')


def selected_gpu_jobs(gpus):
    return gpu_processes(gpus)


class Workflow:
    def __init__(self, plan, benchmark_stop_receipt=None):
        validate_plan(plan)
        self.plan, self.root = plan, Path(plan['run_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.registry = self.root / 'registry'
        self.benchmark_stop_receipt = benchmark_stop_receipt
        sys.path.insert(0, plan['rax_root'])

    def state(self, phase, **extra):
        write(self.root / 'state.json', dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **extra))

    def resume_completed(self):
        """A completed training gate is terminal even if the owner died afterward."""
        path = Path(self.plan['training']['run_root']) / 'state.json'
        trained = read(path) if path.exists() else {}
        previous = read(self.root / 'state.json') if (self.root / 'state.json').exists() else {}
        if trained.get('phase') == 'complete':
            frozen_plan = self.root / 'training_plan.json'
            if not frozen_plan.exists() or read(frozen_plan) != self.plan['training']:
                raise ValueError('Completed training has no matching frozen training plan')
            self.state('complete', training=trained)
            return True
        if previous.get('phase') == 'complete':
            raise ValueError('Workflow claims completion without completed training evidence')
        return False

    def wait_idle(self):
        while not self.stop.is_set():
            busy = selected_gpu_jobs(self.plan['gpus'])
            if not busy:
                return
            self.state('waiting_for_gpus', processes=busy)
            self.stop.wait(self.plan['poll_seconds'])
        raise RuntimeError('Workflow interrupted')

    def execute(self, command, log_path, *, gpu=None, timeout=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                   WANDB_MODE='disabled', ROBOTWIN_PATH=self.plan['robotwin_root'])
        if gpu is not None:
            env.update(VK_ICD_FILENAMES=self.plan['vulkan_icd'], VK_DRIVER_FILES=self.plan['vulkan_icd'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            # The guardian owns its command group; parent EOF also cleans up
            # native/torchrun children if this workflow itself disappears.
            child = spawn_resource(command, cwd=self.plan['rax_root'], env=env, stdout=log)
            started = time.monotonic()
            try:
                while child.poll() is None:
                    if self.stop.wait(1) or (timeout and time.monotonic()-started > timeout):
                        raise TimeoutError(f'Owned child interrupted/timed out; see {log_path}')
                return child.returncode
            finally:
                if child.poll() is None:
                    # The group was created above for this exact child only.
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                if child.stdin is not None:
                    child.stdin.close()

    def prepare_templates(self):
        """Select candidate rules on all source episodes before spending GPU time."""
        import h5py
        from services.robotwin.annotation_templates import entries
        from services.robotwin.programmatic_annotation import OutsideTemplate, instantiate, template_hash

        source = Path(self.plan['source_dataset_root'])
        assignments, jobs = [], {}
        for task in self.plan['tasks']:
            files = sorted((source / task / 'aloha-agilex_clean_50/data').glob('episode*.hdf5'))
            if len(files) != 50:
                raise ValueError(f'Source task {task} must have exactly 50 clean episodes')
            candidates = [e for _, e in entries(source, task) if e['status'] in ('active', 'candidate')]
            candidates.sort(key=lambda e: (e['status'] != 'active', template_hash(e['template'])))
            for path in files:
                chosen = None
                with h5py.File(path, 'r') as handle:
                    for entry in candidates:
                        try:
                            instantiate(handle, entry['template'])
                            chosen = entry
                            break
                        except OutsideTemplate:
                            continue
                if chosen is None:
                    raise ValueError(f'No reviewed template matches {path}; RAX review required before acceptance')
                thash = template_hash(chosen['template'])
                key = task + '/' + thash
                template_path = self.registry / 'annotation_templates' / task / (thash + '.json')
                if not template_path.exists():
                    write(template_path, dict(template=chosen['template'], status='candidate',
                                              source_episode=str(path), agent_run='acceptance-v2-migration',
                                              validation=[], legacy_source_status=chosen['status']))
                elif read(template_path)['template'] != chosen['template']:
                    raise ValueError('Registry template changed under the same content hash')
                jobs[key] = dict(task=task, template_sha256=thash, template=str(template_path),
                                 output=str(self.root / 'acceptance' / task / thash))
                assignments.append(dict(episode=str(path.relative_to(source)), task=task, template_sha256=thash))
        manifest = dict(source=str(source), jobs=list(jobs.values()), assignments=assignments)
        bind(self.root / 'template_plan.json', manifest)
        return manifest['jobs']

    def validate_jobs(self, jobs):
        from services.robotwin.annotation_templates import attach_evidence

        pending, reports = queue.Queue(), []
        mutex = threading.Lock()
        for job in jobs:
            pending.put(job)

        def worker(gpu):
            while not self.stop.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                directory = Path(job['output'])
                write(self.root / f'worker_{gpu}.json', dict(job, gpu=gpu, updated_unix=time.time()))
                command = [self.plan['robotwin_python'],
                           str(Path(self.plan['rax_root']) / 'scripts/validate_robotwin_constraint_acceptance.py'),
                           '--template', job['template'], '--training-root', self.plan['source_dataset_root'],
                           '--output', str(directory), '--seed-start', str(self.plan['acceptance']['seed_start']),
                           '--seeds', str(self.plan['acceptance']['seeds']),
                           '--baseline-repeats', str(self.plan['acceptance']['baseline_repeats']),
                           '--max-seed-attempts', str(self.plan['acceptance']['max_seed_attempts'])]
                try:
                    code = self.execute(command, directory / 'launcher.log', gpu=gpu,
                                        timeout=self.plan['acceptance']['timeout_seconds'])
                    if code:
                        raise RuntimeError(f'Acceptance runner exited {code}; inspect {directory}')
                    report = attach_evidence(self.registry, job['task'], job['template_sha256'], directory,
                                             protocol=self.plan['acceptance']['protocol'],
                                             reference_root=self.plan['source_dataset_root'])
                    record = dict(job, passed=report['ready'], reasons=report['reasons'])
                except Exception as exc:
                    record = dict(job, passed=False, error=f'{type(exc).__name__}: {exc}')
                with mutex:
                    reports.append(record)
                    write(self.root / 'acceptance_progress.json', dict(completed=reports, queued=pending.qsize()))
                pending.task_done()

        with ThreadPoolExecutor(max_workers=len(self.plan['gpus'])) as pool:
            futures = [pool.submit(worker, gpu) for gpu in self.plan['gpus']]
            for future in futures:
                future.result()
        failed = [r for r in reports if not r['passed']]
        if failed or len(reports) != len(jobs):
            write(self.root / 'acceptance_blocked.json', dict(failed=failed, interrupted=self.stop.is_set()))
            raise RuntimeError('Acceptance did not pass; source data retained, publication and training withheld')

    def run(self):
        with (self.root / 'workflow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bind(self.root / 'resolved_plan.json', self.plan)
            if self.resume_completed():
                return
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: self.stop.set())
            try:
                release = benchmark_stop_release(self.plan, self.benchmark_stop_receipt)
                if release is not None:
                    bind(self.root / 'benchmark_release.json', release)
                    self.state('benchmark_released_by_user', receipt_sha256=release['receipt_sha256'])
                while release is None and not benchmark_complete(self.plan['benchmark_root'], self.plan['tasks']):
                    self.state('waiting_for_benchmark', benchmark_root=self.plan['benchmark_root'])
                    if self.stop.wait(self.plan['poll_seconds']):
                        raise RuntimeError('Workflow interrupted')
                self.wait_idle()
                self.state('planning_acceptance')
                jobs = self.prepare_templates()
                pilot = [job for job in jobs if job['task'] in self.plan['pilot_tasks']]
                self.state('acceptance_pilot', templates=len(pilot))
                self.validate_jobs(pilot)
                remaining = [job for job in jobs if job not in pilot]
                self.state('acceptance', templates=len(remaining))
                self.validate_jobs(remaining)
                check_publication_hold(self.root)
                self.state('publishing_new_dataset')
                command = [self.plan['python'], str(Path(self.plan['rax_root']) /
                           'scripts/publish_robotwin_constraint_dataset.py'), '--source-root',
                           self.plan['source_dataset_root'], '--registry-root', str(self.registry),
                           '--output', self.plan['training']['dataset_root']]
                if self.plan.get('require_constraint_refinement'):
                    command.append('--require-refinement')
                if self.execute(command, self.root / 'publication.log'):
                    raise RuntimeError('Dataset publication failed; training withheld')
                self.wait_idle()
                check_publication_hold(self.root)
                train_plan = self.root / 'training_plan.json'
                bind(train_plan, self.plan['training'])
                self.state('training_gate')
                # The training gate selects CUDA itself after checking GPUs.
                command = [self.plan['python'], str(Path(self.plan['openwam_root']) /
                           'scripts/robotwin_training_gate.py'), '--plan', str(train_plan)]
                if self.execute(command, self.root / 'training_gate.log'):
                    raise RuntimeError('New training gate failed; see training_run/state.json')
                trained = read(Path(self.plan['training']['run_root']) / 'state.json')
                if trained['phase'] != 'complete':
                    raise RuntimeError('Training did not reach confirmed completion')
                self.state('complete', training=trained)
            except BaseException as exc:
                self.state('blocked', error=f'{type(exc).__name__}: {exc}',
                           next_action='Inspect evidence and repair the failed stage; no acceptance thresholds are relaxed automatically.')
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--benchmark-stop-receipt', type=Path,
                        help='Explicit user-stop evidence; never marks an incomplete benchmark complete')
    args = parser.parse_args()
    plan = read(args.plan)
    validate_plan(plan)
    if args.check_only:
        release = benchmark_stop_release(plan, args.benchmark_stop_receipt)
        complete = False if release is not None else benchmark_complete(plan['benchmark_root'], plan['tasks'])
        print(json.dumps(dict(valid=True, benchmark_complete=complete, benchmark_released_by_user=release is not None)))
    else:
        Workflow(plan, benchmark_stop_receipt=args.benchmark_stop_receipt).run()


if __name__ == '__main__':
    main()
