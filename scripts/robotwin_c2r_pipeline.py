#!/usr/bin/env python3
"""Resume a live C2R run without a global manifest barrier.

The legacy coordinator is stopped during handoff, while its existing expert
children finish unchanged. Idle GPUs evaluate ready tasks immediately. This
runner publishes the inherited manifests and retires only the stopped parent
after all inherited resource guardians exit. Each model episode is committed
separately; native scenes, seeds, success predicates and inference stay fixed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.robotwin_c2r_gate import (  # noqa: E402
    Gate,
    PolicyServer,
    atomic_json,
    bind_json,
    preflight,
    read_plan,
    sha256,
    training_ready,
    validate_manifest,
    write_manifest,
)


def identity(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        return dict(
            pid=pid,
            start_ticks=int(fields[19]),
            status=fields[0],
            argv=(path / "cmdline").read_bytes().decode().split("\0")[:-1],
        )
    except FileNotFoundError:
        return None


def same_process(record, *, running=False):
    current = identity(record["pid"])
    return bool(
        current
        and current["start_ticks"] == record["start_ticks"]
        and (not running or current["status"] not in ("Z", "X"))
    )


def capture_handoff(plan, path):
    """Pause only this run's legacy parent, retaining its active native clients."""
    root = Path(plan["run_root"])
    if path.exists():
        return json.loads(path.read_text())
    state = json.loads((root / "state.json").read_text())
    if state["phase"] != "building_manifests":
        raise ValueError("Live handoff requires the legacy manifest-building phase")
    parent = identity(state["pid"])
    if not parent or str(ROOT / "scripts/robotwin_c2r_gate.py") not in parent["argv"]:
        raise ValueError("Legacy coordinator identity mismatch")
    os.kill(parent["pid"], signal.SIGSTOP)
    try:
        deadline = time.monotonic() + 3
        while identity(parent["pid"])["status"] not in ("T", "t"):
            if time.monotonic() > deadline:
                raise TimeoutError("Could not stop the legacy coordinator")
            time.sleep(0.05)
        # SIGSTOP freezes dispatch only. Resource guardians use their own
        # process groups and keep their lifelines open while the parent lives.
        thread_root = Path("/proc") / str(parent["pid"]) / "task"
        children = set()
        for task_dir in thread_root.iterdir():
            children.update(int(pid) for pid in (task_dir / "children").read_text().split())
        workers = [json.loads(p.read_text()) for p in root.glob("worker_*.json")]
        jobs = []
        for pid in children:
            record = identity(pid)
            if not record or record["status"] == "Z":
                continue
            argv = record["argv"]
            if not any(arg.endswith("/resource_process.py") for arg in argv):
                raise ValueError(f"Unexpected live child of legacy coordinator: {pid}")
            if "--operation" not in argv or argv[argv.index("--operation") + 1] != "build_manifest":
                raise ValueError("Only expert-manifest children may be inherited")
            task = argv[argv.index("--task") + 1]
            result = Path(argv[argv.index("--result-file") + 1])
            if result.parent.parent != root / "manifest_attempts" / task:
                raise ValueError("Inherited attempt belongs to another run")
            worker = next(w for w in workers if w["task"] == task)
            jobs.append(dict(task=task, gpu=worker["gpu"], guardian=record, directory=str(result.parent)))
        if len({job["gpu"] for job in jobs}) != len(jobs):
            raise ValueError("More than one inherited job owns a GPU")
        handoff = dict(created_unix=time.time(), legacy_parent=parent, legacy_state=state, jobs=jobs)
        atomic_json(path, handoff)
        return handoff
    except BaseException:
        os.kill(parent["pid"], signal.SIGCONT)
        raise


class Pipeline(Gate):
    def __init__(self, plan, handoff):
        super().__init__(plan)
        self.handoff = handoff
        self.condition = threading.Condition()
        self.inherited = {job["gpu"]: job for job in handoff["jobs"]}
        self.claimed = {job["task"] for job in handoff["jobs"]}
        self.inherited_pending = set(self.claimed)
        self.failed = {}
        self.done = set()

    def mark_worker(self, gpu, task, operation, **extra):
        atomic_json(
            self.root / f"worker_{gpu}.json",
            dict(task=task, gpu=gpu, operation=operation, active=True, updated_unix=time.time(), **extra),
        )

    def claim(self, prefer_eval=False):
        """Return one unowned task; caller holds condition. None means no free work."""
        ready, missing = [], []
        for task in self.plan["tasks"]:
            if task in self.claimed or task in self.failed or task in self.done:
                continue
            if (self.root / "manifests" / f"{task}.json").exists():
                if self.completed_prefix(task) == 10:
                    self.done.add(task)
                else:
                    ready.append(task)
            else:
                missing.append(task)
        if ready and (prefer_eval or not missing):
            task, operation = ready[0], "run_eval"
        elif missing:
            task, operation = missing[0], "build_manifest"
        else:
            return None
        self.claimed.add(task)
        return task, operation

    def inherit_manifest(self, job):
        task, gpu = job["task"], job["gpu"]
        directory = Path(job["directory"])
        self.mark_worker(gpu, task, "build_manifest", inherited=True)
        deadline = time.monotonic() + self.plan["manifest_timeout_seconds"]
        while same_process(job["guardian"], running=True):
            if self.stop.wait(0.5):
                os.kill(job["guardian"]["pid"], signal.SIGTERM)
                raise RuntimeError("Handoff interrupted")
            if time.monotonic() > deadline:
                # Stop only the inherited guardian; it cleans its own client.
                os.kill(job["guardian"]["pid"], signal.SIGTERM)
                raise TimeoutError(f"Inherited expert timed out: {task}")
        payload = validate_manifest(json.loads((directory / "result.json").read_text()), task, self.plan)
        write_manifest(self.root / "manifests" / f"{task}.json", payload)

    def retire_legacy_if_finished(self):
        if self.inherited_pending:
            return
        # All inherited guardians have exited; the stopped parent must never
        # dispatch its old static shards or write an outdated final state.
        parent = self.handoff["legacy_parent"]
        if same_process(parent):
            if identity(parent["pid"])["status"] not in ("T", "t", "Z"):
                raise RuntimeError("Legacy coordinator unexpectedly resumed")
            os.kill(parent["pid"], signal.SIGKILL)
        atomic_json(self.root / "legacy_retired.json", dict(pid=parent["pid"], retired_unix=time.time()))

    def finish_job(self, task, operation, error=None):
        with self.condition:
            self.claimed.discard(task)
            if error is not None:
                record = dict(task=task, operation=operation, error=str(error), updated_unix=time.time())
                self.failed[task] = record
                atomic_json(self.root / "errors" / f"{operation}_{task}.json", record)
            elif operation == "run_eval":
                self.done.add(task)
            self.condition.notify_all()

    def worker(self, index, gpu):
        server = PolicyServer(self, gpu, self.plan["base_port"] + index)
        try:
            job = self.inherited.get(gpu)
            if job:
                try:
                    self.inherit_manifest(job)
                    self.finish_job(job["task"], "build_manifest")
                except Exception as exc:
                    self.finish_job(job["task"], "build_manifest", exc)
                # Do not hold the queue lock while the guardian cleans up;
                # other GPUs must keep dispatching and committing episodes.
                while same_process(job["guardian"], running=True):
                    time.sleep(0.5)
                with self.condition:
                    self.inherited_pending.discard(job["task"])
                    self.retire_legacy_if_finished()
            prefer_eval = not bool(job)
            while not self.stop.is_set():
                with self.condition:
                    work = self.claim(prefer_eval)
                    if work is None:
                        atomic_json(
                            self.root / f"worker_{gpu}.json",
                            dict(gpu=gpu, operation="idle", active=False, updated_unix=time.time()),
                        )
                        if not self.claimed:
                            break
                        self.condition.wait(timeout=2)
                        continue
                task, operation = work
                self.mark_worker(gpu, task, operation)
                try:
                    if operation == "build_manifest":
                        self.build_manifest(task, gpu)
                    else:
                        start = self.completed_prefix(task)
                        for episode in range(start, 10):
                            if self.stop.is_set():
                                raise RuntimeError("Evaluation interrupted")
                            self.mark_worker(gpu, task, operation, episode=episode)
                            self.evaluate_range(task, server, episode, 1)
                    self.finish_job(task, operation)
                except Exception as exc:
                    self.finish_job(task, operation, exc)
                prefer_eval = False
        finally:
            server.close()

    def run_pipeline(self):
        with (self.root / "pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: self.stop.set())
            bind_json(self.root / "resolved_plan.json", self.plan)
            bind_json(self.root / "preflight.json", preflight(self.plan))
            checkpoint = training_ready(self.plan)
            if checkpoint is None or not os.path.samefile(checkpoint, self.model_dir / checkpoint.name):
                raise ValueError("Final checkpoint snapshot changed")
            if not (self.root / "smoke_passed.json").exists():
                raise ValueError("A completed real rollout is required before live handoff")
            atomic_json(
                self.root / "pipeline_provenance.json",
                dict(
                    started_unix=time.time(),
                    source_sha256=sha256(__file__),
                    scheduler="dynamic-ready-tasks",
                    episodes_per_commit=1,
                    inherited_jobs=self.handoff["jobs"],
                ),
            )
            self.state("pipelined_evaluation", inherited_expert_jobs=len(self.inherited), episode_batch=1)
            self.summarize()
            self.retire_legacy_if_finished()
            try:
                with ThreadPoolExecutor(max_workers=len(self.plan["gpus"])) as executor:
                    futures = [executor.submit(self.worker, i, gpu) for i, gpu in enumerate(self.plan["gpus"])]
                    for future in as_completed(futures):
                        future.result()
                self.retire_legacy_if_finished()
                summary = self.summarize()
                if self.stop.is_set() or self.failed or not summary["complete"]:
                    raise RuntimeError(f"Incomplete evaluation: {len(self.failed)} failed tasks; see errors/")
                self.state(
                    "complete", episodes=500, successes=summary["successes"], success_rate=summary["success_rate"]
                )
            except BaseException as exc:
                self.state("failed", error=f"{type(exc).__name__}: {exc}")
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-handoff", action="store_true")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    handoff_path = Path(plan["run_root"]) / "pipeline_handoff.json"
    if args.capture_handoff:
        # Validate unchanged protocol before touching any owned process.
        bind_json(Path(plan["run_root"]) / "preflight.json", preflight(plan))
        print(json.dumps(capture_handoff(plan, handoff_path), indent=2))
    else:
        Pipeline(plan, json.loads(handoff_path.read_text())).run_pipeline()


if __name__ == "__main__":
    main()
