"""Queue and handoff invariants without starting any real GPU process."""

import json
import signal

from benchmarks.utils.eval_manifest import seal_manifest, write_manifest
from scripts import robotwin_c2r_pipeline as pipeline


def runner(tmp_path, tasks, inherited=()):
    plan = dict(
        run_root=str(tmp_path),
        tasks=tasks,
        gpus=[0, 1],
        base_port=18848,
        mode="demo_randomized",
        instruction_type="unseen",
        episodes_per_task=10,
        manifest_timeout_seconds=7200,
    )
    handoff = dict(jobs=list(inherited), legacy_parent=dict(pid=999999, start_ticks=123))
    return pipeline.Pipeline(plan, handoff)


def ready(run, task):
    payload = seal_manifest(
        dict(
            benchmark="robotwin",
            task=task,
            mode="demo_randomized",
            entries=[
                dict(episode=i, seed=100000 + i, instructions=dict(seen="seen", unseen="unseen")) for i in range(10)
            ],
        )
    )
    write_manifest(run.root / "manifests" / f"{task}.json", payload)
    return payload


def test_idle_gpu_evaluates_ready_task_while_expert_is_still_running(tmp_path):
    inherited = [dict(task="slow_expert", gpu=1)]
    run = runner(tmp_path, ["slow_expert", "ready_a", "ready_b"], inherited)
    ready(run, "ready_a")
    ready(run, "ready_b")
    with run.condition:
        assert run.claim(prefer_eval=True) == ("ready_a", "run_eval")
        assert run.claim(prefer_eval=True) == ("ready_b", "run_eval")
        assert run.claim(prefer_eval=True) is None
    assert "slow_expert" in run.inherited_pending


def test_remaining_manifests_are_not_starved_after_first_ready_task(tmp_path):
    run = runner(tmp_path, ["missing", "ready_a"])
    ready(run, "ready_a")
    with run.condition:
        assert run.claim(prefer_eval=True) == ("ready_a", "run_eval")
        assert run.claim(prefer_eval=False) == ("missing", "build_manifest")


def test_model_episodes_are_committed_one_at_a_time(tmp_path, monkeypatch):
    run = runner(tmp_path, ["ready_a"])
    ready(run, "ready_a")
    calls = []

    class Server:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "PolicyServer", Server)
    monkeypatch.setattr(run, "evaluate_range", lambda task, server, start, count: calls.append((start, count)))
    run.worker(0, 0)
    assert calls == [(i, 1) for i in range(10)]
    assert run.done == {"ready_a"}
    assert json.loads((tmp_path / "worker_0.json").read_text())["active"] is False


def test_inherited_finished_manifest_is_published_without_resampling(tmp_path, monkeypatch):
    run = runner(tmp_path, ["task"])
    payload = ready(run, "task")
    (tmp_path / "manifests/task.json").unlink()
    attempt = tmp_path / "manifest_attempts/task/1000-1"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps(payload))
    monkeypatch.setattr(pipeline, "same_process", lambda *a, **k: False)
    run.inherit_manifest(dict(task="task", gpu=1, guardian={}, directory=str(attempt)))
    assert json.loads((tmp_path / "manifests/task.json").read_text()) == payload


def test_parent_is_retired_only_after_all_inherited_jobs_exit(tmp_path, monkeypatch):
    run = runner(tmp_path, ["slow"], [dict(task="slow", gpu=1)])
    signals = []
    monkeypatch.setattr(pipeline, "same_process", lambda *a, **k: True)
    monkeypatch.setattr(pipeline, "identity", lambda pid: dict(status="T"))
    monkeypatch.setattr(pipeline.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    run.retire_legacy_if_finished()
    assert not signals
    run.inherited_pending.clear()
    run.retire_legacy_if_finished()
    assert signals == [(999999, signal.SIGKILL)]


def test_pid_reuse_is_never_treated_as_owned_process(monkeypatch):
    monkeypatch.setattr(pipeline, "identity", lambda pid: dict(pid=pid, start_ticks=456, status="R"))
    assert not pipeline.same_process(dict(pid=99, start_ticks=123))
    monkeypatch.setattr(pipeline, "identity", lambda pid: dict(pid=pid, start_ticks=123, status="Z"))
    assert not pipeline.same_process(dict(pid=99, start_ticks=123), running=True)
