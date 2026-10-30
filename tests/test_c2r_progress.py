"""Progress semantics: preparation != evaluation; in-flight counts never duplicate."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.robotwin.progress_snapshot import collect, native_step
from scripts.serve_c2r_progress import Monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    plan = tmp_path / "plan.json"
    write(
        plan,
        dict(
            run_root=str(root),
            episodes_per_task=10,
            tasks=["task_a", "task_b"],
            gpus=[0, 1],
            mode="demo_randomized",
            instruction_type="unseen",
        ),
    )
    write(root / "state.json", dict(phase="building_manifests", pid=0))
    write(root / "summary.json", dict(tasks=[dict(task="task_a", completed_episodes=1, successes=0)]))
    write(root / "results/task_a/episode_000.json", dict(success=False))
    return root, plan


def test_scenes_are_not_counted_as_model_successes(run):
    root, plan = run
    write(root / "manifests/task_a.json", dict(entries=[{}] * 10))
    write(root / "worker_1.json", dict(task="task_b", gpu=1, operation="build_manifest"))
    write(root / "manifest_attempts/task_b/1000000000-1/progress.json", dict(completed=6, total=10))
    result = collect(plan)
    assert result["scene_episodes"] == 16
    assert result["scene_tasks"] == 1
    assert result["completed_episodes"] == 1
    assert result["successes"] == 0 and result["success_rate"] == 0


def active_batch(root, *, start=1, count=9, done=3, wins=2):
    write(root / "state.json", dict(phase="evaluating", pid=0))
    write(root / "worker_0.json", dict(task="task_a", gpu=0, operation="run_eval"))
    directory = root / f"eval_attempts/task_a/{start}-{count}-2000000000-1"
    write(directory / "progress.json", dict(completed=done, total=count, successes=wins))
    return directory


def test_live_completed_episodes_include_active_batch_once(run):
    root, plan = run
    active_batch(root)
    result = collect(plan)
    assert result["completed_episodes"] == 4
    assert result["confirmed_episodes"] == 1
    assert result["provisional_episodes"] == 3
    assert result["successes"] == 2 and result["failures"] == 2
    assert result["success_rate"] == 0.5
    assert result["final_success_rate"] is None


def test_committed_batch_is_never_added_to_old_live_progress(run):
    root, plan = run
    active_batch(root, done=9, wins=4)
    write(root / "summary.json", dict(tasks=[dict(task="task_a", completed_episodes=10, successes=4)]))
    result = collect(plan)
    assert result["completed_episodes"] == 10
    assert result["successes"] == 4 and result["provisional_episodes"] == 0


@pytest.mark.parametrize("failed", ["attempt", "gate"])
def test_infrastructure_failure_discards_only_provisional_results(run, failed):
    root, plan = run
    directory = active_batch(root)
    if failed == "attempt":
        write(directory / "error.json", dict(error="connection closed"))
    else:
        write(root / "state.json", dict(phase="failed", pid=0, error="client failed"))
    result = collect(plan)
    assert result["completed_episodes"] == 1
    assert result["successes"] == 0 and result["provisional_episodes"] == 0


def test_no_finished_episode_is_unknown_rate_not_zero(run):
    root, plan = run
    write(root / "summary.json", dict(tasks=[]))
    assert collect(plan)["success_rate"] is None


def test_native_step_parses_carriage_returns_and_ansi(run):
    root, _ = run
    root.mkdir(exist_ok=True)
    (root / "client.log").write_bytes(b"step: \x1b[92m99 / 400\x1b[0m\rstep: \x1b[92m100 / 400\x1b[0m\r")
    assert native_step(root) == dict(current=100, total=400)


def test_ssh_failure_preserves_last_successful_snapshot(monkeypatch):
    args = SimpleNamespace(interval=10, ssh_port=20347, host="test-host", remote_plan="/plan.json")
    monitor = Monitor(args)
    original = dict(completed_episodes=17, successes=5, collected_at=123)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(original)))
    monitor.refresh()

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 18)

    monkeypatch.setattr(subprocess, "run", fail)
    monitor.refresh()
    assert monitor.snapshot()["data"] == original
    assert monitor.snapshot()["connected"] is False
    monitor.socket_dir.cleanup()
