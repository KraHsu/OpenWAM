"""CPU regressions for final-checkpoint gating and honest 500-episode scoring."""

import copy
import json
import struct
from pathlib import Path

import pytest
import yaml

from benchmarks.utils.eval_manifest import seal_manifest, write_manifest
from scripts import robotwin_c2r_gate as gate


@pytest.fixture
def plan(tmp_path):
    return dict(
        train_root=str(tmp_path / "training"),
        train_output=str(tmp_path / "training/run"),
        run_root=str(tmp_path / "eval"),
        openwam_root=str(gate.ROOT),
        robotwin_root=str(tmp_path / "native"),
        rax_root=str(tmp_path / "rax"),
        server_python="/usr/bin/python3",
        robotwin_python="/usr/bin/python3",
        mode="demo_randomized",
        tasks=gate.official_tasks(),
        episodes_per_task=10,
        expected_steps=42970,
        instruction_type="unseen",
        gpus=[0, 1, 2, 3],
        base_port=18848,
    )


def checkpoint(path):
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    return path


def complete_training(plan):
    output = Path(plan["train_output"])
    ckpt = checkpoint(output / "checkpoint_step_42970.safetensors")
    gate.atomic_json(Path(plan["train_root"]) / "state.json", {"phase": "complete", "run": str(output)})
    row = dict(step=42970, epoch=4, loss_total=0.1, loss_video=0.09, loss_action=0.005, loss_constraint=0.005)
    (output / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    return ckpt


def manifest(plan, task=None):
    return seal_manifest(
        dict(
            benchmark="robotwin",
            task=task or plan["tasks"][0],
            mode=plan["mode"],
            entries=[
                dict(episode=i, seed=100000 + 2 * i, instructions={"seen": f"seen {i}", "unseen": f"unseen {i}"})
                for i in range(10)
            ],
        )
    )


def result(plan, source, start=0, count=10):
    rows = [
        dict(episode=e["episode"], seed=e["seed"], instruction=e["instructions"]["unseen"], success=False)
        for e in source["entries"][start : start + count]
    ]
    return dict(
        operation="run_eval",
        task=source["task"],
        mode=plan["mode"],
        manifest_hash=source["manifest_hash"],
        instruction_type="unseen",
        episode_start=start,
        num_episodes=count,
        total_episodes=10,
        episodes=rows,
        successes=0,
    )


def test_training_must_exit_even_if_final_file_exists(plan):
    complete_training(plan)
    gate.atomic_json(Path(plan["train_root"]) / "state.json", {"phase": "training"})
    assert gate.training_ready(plan) is None


def test_complete_weights_do_not_require_deleted_optimizer_state(plan):
    ckpt = complete_training(plan)
    assert not list(ckpt.parent.glob("accel_state*"))
    assert gate.training_ready(plan) == ckpt


@pytest.mark.parametrize("problem", ["failed", "wrong_run", "wrong_step", "wrong_epoch", "nan", "truncated"])
def test_reject_failed_or_incomplete_training(plan, problem):
    ckpt = complete_training(plan)
    state = Path(plan["train_root"]) / "state.json"
    if problem == "failed":
        gate.atomic_json(state, {"phase": "failed", "error": "OOM"})
    elif problem == "wrong_run":
        gate.atomic_json(state, {"phase": "complete", "run": "/wrong"})
    elif problem == "truncated":
        ckpt.write_bytes(ckpt.read_bytes()[:-1])
    else:
        metrics = ckpt.parent / "metrics.jsonl"
        row = json.loads(metrics.read_text())
        row[{"wrong_step": "step", "wrong_epoch": "epoch", "nan": "loss_total"}[problem]] = {
            "wrong_step": 42968,
            "wrong_epoch": 3,
            "nan": float("nan"),
        }[problem]
        metrics.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, RuntimeError)):
        gate.training_ready(plan)


@pytest.mark.parametrize("problem", ["duplicate_seed", "clean_mode", "too_few", "wrong_task", "tampered"])
def test_manifest_rejects_protocol_drift(plan, problem):
    source = manifest(plan)
    body = {k: copy.deepcopy(v) for k, v in source.items() if k != "manifest_hash"}
    if problem == "duplicate_seed":
        body["entries"][1]["seed"] = body["entries"][0]["seed"]
    elif problem == "clean_mode":
        body["mode"] = "demo_clean"
    elif problem == "too_few":
        body["entries"].pop()
    elif problem == "wrong_task":
        body["task"] = "wrong"
    else:
        source["entries"][0]["seed"] += 1
    candidate = source if problem == "tampered" else seal_manifest(body)
    with pytest.raises(ValueError):
        gate.validate_manifest(candidate, plan["tasks"][0], plan)


def test_failed_rollout_is_a_scored_episode_not_infrastructure_error(plan):
    source = manifest(plan)
    payload = result(plan, source, 0, 1)
    assert gate.validate_result(payload, source, 0, 1, plan)["successes"] == 0


@pytest.mark.parametrize("problem", ["seed", "instruction", "missing", "non_boolean", "success_count", "hash"])
def test_result_rejects_missing_or_mismatched_episodes(plan, problem):
    source = manifest(plan)
    payload = result(plan, source)
    if problem == "seed":
        payload["episodes"][0]["seed"] += 1
    elif problem == "instruction":
        payload["episodes"][0]["instruction"] = "seen 0"
    elif problem == "missing":
        payload["episodes"].pop()
    elif problem == "non_boolean":
        payload["episodes"][0]["success"] = 0
    elif problem == "success_count":
        payload["successes"] = 1
    else:
        payload["manifest_hash"] = "wrong"
    with pytest.raises(ValueError):
        gate.validate_result(payload, source, 0, 10, plan)


def test_partial_run_cannot_be_reported_as_500_episode_score(plan):
    runner = gate.Gate(plan)
    source = manifest(plan)
    write_manifest(runner.root / "manifests" / f"{source['task']}.json", source)
    row = result(plan, source, 0, 1)["episodes"][0]
    gate.atomic_json(
        runner.root / "results" / source["task"] / "episode_000.json", dict(row, manifest_hash=source["manifest_hash"])
    )
    assert runner.completed_prefix(source["task"]) == 1
    summary = runner.summarize()
    assert summary["expected_episodes"] == 500
    assert summary["completed_episodes"] == 1
    assert summary["success_rate"] is None
    assert summary["partial_success_rate"] == 0
    assert not summary["complete"]


def test_all_tasks_are_required_for_final_score(plan):
    runner = gate.Gate(plan)
    for task in plan["tasks"]:
        for i in range(10):
            gate.atomic_json(runner.root / "results" / task / f"episode_{i:03}.json", {"success": i < 3})
    summary = runner.summarize()
    assert summary["complete"] and summary["completed_tasks"] == 50
    assert summary["successes"] == 150 and summary["success_rate"] == 0.3


def test_private_runtime_does_not_change_native_randomization(tmp_path, monkeypatch):
    from benchmarks.robotwin.c2r_eval_client import prepare_private_runtime

    native = tmp_path / "native"
    runtime = tmp_path / "runtime"
    config = native / "task_config/demo_randomized.yml"
    config.parent.mkdir(parents=True)
    settings = dict(
        collect_data=True,
        save_data=True,
        eval_video_log=True,
        domain_randomization={"random_background": True, "random_table_height": 0.03},
    )
    original = yaml.safe_dump(settings)
    config.write_text(original)
    monkeypatch.setenv("ROBOTWIN_PATH", str(native))
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(runtime))
    monkeypatch.setattr("sys.argv", ["client", "--operation", "build_manifest"])
    prepare_private_runtime()
    private = yaml.safe_load((runtime / "task_config/demo_randomized.yml").read_text())
    assert config.read_text() == original
    assert private["domain_randomization"] == settings["domain_randomization"]
    assert private["collect_data"] is False and private["save_data"] is False
    assert private["eval_video_log"] is False


def test_changed_plan_cannot_reuse_old_results(tmp_path):
    path = tmp_path / "resolved_plan.json"
    gate.bind_json(path, {"instruction_type": "unseen"})
    gate.bind_json(path, {"instruction_type": "unseen"})
    with pytest.raises(ValueError):
        gate.bind_json(path, {"instruction_type": "seen"})


def test_exact_official_task_catalog(plan, tmp_path):
    assert len(plan["tasks"]) == len(set(plan["tasks"])) == 50
    path = tmp_path / "plan.json"
    gate.atomic_json(path, plan)
    assert gate.read_plan(path)["tasks"] == plan["tasks"]
    plan["tasks"].pop()
    gate.atomic_json(path, plan)
    with pytest.raises(ValueError):
        gate.read_plan(path)


def test_selected_gpus_must_be_idle(monkeypatch):
    def output(command, **kwargs):
        if "--query-gpu=index,uuid" in command:
            return "0, GPU-a\n1, GPU-b\n"
        return "GPU-a, 100, python\nGPU-b, 101, python\n"

    monkeypatch.setattr(gate.subprocess, "check_output", output)
    assert gate.gpu_processes([1]) == ["GPU-b, 101, python"]
    with pytest.raises(ValueError):
        gate.gpu_processes([2])


def test_vulkan_override_is_child_scoped(plan, tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("VK_ICD_FILENAMES", "/original/driver.json")
    plan["vulkan_icd"] = "/headless/nvidia.json"
    env = gate.client_environment(plan, 2, tmp_path)
    assert env["VK_ICD_FILENAMES"] == env["VK_DRIVER_FILES"] == plan["vulkan_icd"]
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["VK_ICD_FILENAMES"] == "/original/driver.json"
