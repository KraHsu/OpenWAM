#!/usr/bin/env python3
"""Wait for a successful final training artifact, then evaluate 50 random x 10.

No training mutation or simulator/model initialization while waiting. Uses the
existing RoboTwin sealed-manifest client without adding a scheduling service.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.utils.eval_manifest import load_manifest, verify_manifest, write_manifest  # noqa: E402
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource  # noqa: E402


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bind_json(path, data):
    """Refuse to mix a resumed run with changed settings or source files."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Existing evaluation provenance differs: {path}")
    else:
        atomic_json(path, data)


def official_tasks():
    tree = ast.parse((ROOT / "benchmarks/robotwin/labtasker_runtime.py").read_text())
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ROBOTWIN_TASKS" for t in node.targets)
    )


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    required_paths = (
        "train_root",
        "train_output",
        "run_root",
        "openwam_root",
        "robotwin_root",
        "rax_root",
        "server_python",
        "robotwin_python",
    )
    for key in required_paths:
        if not Path(plan[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    if Path(plan["openwam_root"]).resolve() != ROOT:
        raise ValueError("Run the gate from the specified OpenWAM checkout")
    if plan["mode"] != "demo_randomized" or plan["episodes_per_task"] != 10 or plan["tasks"] != official_tasks():
        raise ValueError("This run requires the ordered official 50 tasks, random x 10")
    if plan["expected_steps"] != 42970 or plan["instruction_type"] not in ("seen", "unseen"):
        raise ValueError("Invalid final training step or instruction type")
    gpus = plan["gpus"]
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Distinct nonnegative GPU indices are required")
    if not 1024 <= plan["base_port"] <= 65535 - len(gpus):
        raise ValueError("Invalid policy server port range")
    destination = Path(plan["run_root"]).resolve()
    for key in ("train_root", "train_output", "robotwin_root"):
        source = Path(plan[key]).resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Evaluation output must be separate from training and native source")
    return plan


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_metric(path):
    with Path(path).open("rb") as handle:
        handle.seek(max(0, Path(path).stat().st_size - 65536))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if "step" in row:
                return row
        except (ValueError, UnicodeDecodeError):
            pass
    raise ValueError("No complete training metric")


def validate_safetensors(path):
    path = Path(path)
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            raise ValueError("Incomplete checkpoint prefix")
        length = struct.unpack("<Q", head)[0]
        if not 2 <= length <= 100_000_000:
            raise ValueError("Invalid safetensors header length")
        metadata = json.loads(handle.read(length))
    tensors = [v for k, v in metadata.items() if k != "__metadata__"]
    if not tensors or max(v["data_offsets"][1] for v in tensors) + 8 + length != path.stat().st_size:
        raise ValueError("Checkpoint payload is incomplete")
    return {"size_bytes": path.stat().st_size, "tensor_count": len(tensors)}


def training_ready(plan):
    state = json.loads((Path(plan["train_root"]) / "state.json").read_text())
    if state["phase"] == "failed":
        raise RuntimeError(f"Training failed: {state.get('error')}")
    if state["phase"] != "complete":
        return None
    output = Path(plan["train_output"]).resolve()
    if Path(state["run"]).resolve() != output:
        raise ValueError("Completed training run does not match the requested run")
    metric = last_metric(output / "metrics.jsonl")
    if metric["step"] != plan["expected_steps"] or metric["epoch"] != 4:
        raise ValueError("Five-epoch completion metric is missing")
    if not all(math.isfinite(metric[k]) for k in ("loss_total", "loss_video", "loss_action", "loss_constraint")):
        raise ValueError("Non-finite final training metric")
    checkpoint = output / f"checkpoint_step_{plan['expected_steps']}.safetensors"
    validate_safetensors(checkpoint)
    return checkpoint


def gpu_processes(gpus):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    identifiers = {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines()}
    if not set(gpus).issubset(identifiers):
        raise ValueError("Requested GPU is not visible")
    selected = {identifiers[g] for g in gpus}
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True
    )
    return [line for line in raw.splitlines() if line.split(",")[0].strip() in selected]


def preflight(plan):
    import yaml

    native = Path(plan["robotwin_root"])
    output = Path(plan["train_output"])
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    arch = cfg["model"]["architecture"]
    if (arch["framework"], arch["variant"], arch["action_dim"]) != ("dual_system", "joint_self_attn", 80):
        raise ValueError("Unexpected checkpoint architecture")
    if cfg["dataloader"]["action_mode"] != "eef" or cfg["dataloader"]["variant"] != "clean_50":
        raise ValueError("C2R requires the clean-only EEF checkpoint")
    for path in [
        output / "normalization_stats.npy",
        output / "tokenizer",
        native / "script/eval_policy.py",
        native / "assets",
        native / "description",
        Path(plan["rax_root"]) / "services/robotwin/native_compat.py",
        ROOT / "benchmarks/robotwin/c2r_eval_client.py",
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    for key in ("server_python", "robotwin_python"):
        if not os.access(plan[key], os.X_OK):
            raise ValueError(f"Interpreter is not executable: {plan[key]}")
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is required for evaluation videos")
    random_cfg = yaml.safe_load((native / "task_config/demo_randomized.yml").read_text())
    dr = random_cfg["domain_randomization"]
    if not all(dr[k] is True for k in ("random_background", "cluttered_table", "random_light")):
        raise ValueError("Native randomized configuration is not enabled")
    for task in plan["tasks"]:
        if not (native / "envs" / f"{task}.py").is_file():
            raise FileNotFoundError(task)
    if yaml.safe_load((ROOT / "benchmarks/robotwin/step_limits.yml").read_text()):
        raise ValueError("Use native step limits; remove experiment overrides in an isolated checkout")
    sources = [
        "scripts/robotwin_c2r_gate.py",
        "scripts/deploy.py",
        "benchmarks/robotwin/c2r_eval_client.py",
        "benchmarks/robotwin/eval_policy_wrapper.py",
        "benchmarks/robotwin/openwam2robotwin_interface.py",
        "benchmarks/robotwin/eval_manifest.py",
        "benchmarks/robotwin/policy_config.yml",
        "benchmarks/robotwin/step_limits.yml",
        "configs/deploy.yaml",
    ]
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "openwam/deploy").rglob("*.py"))]
    return {
        "passed": True,
        "tasks": len(plan["tasks"]),
        "episodes_per_task": 10,
        "planned_episodes": 500,
        "domain_randomization": dr,
        "mode": plan["mode"],
        "instruction_type": plan["instruction_type"],
        "native_commit": subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip(),
        "native_eval_sha256": sha256(native / "script/eval_policy.py"),
        "random_config_sha256": sha256(native / "task_config/demo_randomized.yml"),
        "native_compat_sha256": sha256(Path(plan["rax_root"]) / "services/robotwin/native_compat.py"),
        "vulkan_icd_sha256": sha256(plan["vulkan_icd"]) if plan.get("vulkan_icd") else None,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
    }


def validate_manifest(payload, task, plan):
    payload = verify_manifest(payload)
    entries = payload["entries"]
    if (
        payload["benchmark"] != "robotwin"
        or payload["task"] != task
        or payload["mode"] != plan["mode"]
        or len(entries) != plan["episodes_per_task"]
    ):
        raise ValueError("Manifest task/mode/count mismatch")
    seeds = [e["seed"] for e in entries]
    if seeds != sorted(set(seeds)) or any(type(s) is not int or s < 100000 for s in seeds):
        raise ValueError("Expected distinct held-out native seeds >= 100000")
    for index, entry in enumerate(entries):
        if entry["episode"] != index or not all(
            isinstance(entry["instructions"].get(k), str) and entry["instructions"][k] for k in ("seen", "unseen")
        ):
            raise ValueError("Invalid episode identity or instructions")
    return payload


def validate_result(payload, manifest, start, count, plan):
    expected = manifest["entries"][start : start + count]
    if (
        payload.get("operation") != "run_eval"
        or payload.get("task") != manifest["task"]
        or payload.get("mode") != plan["mode"]
        or payload.get("manifest_hash") != manifest["manifest_hash"]
        or payload.get("instruction_type") != plan["instruction_type"]
        or payload.get("episode_start") != start
        or payload.get("num_episodes") != count
        or payload.get("total_episodes") != plan["episodes_per_task"]
        or len(payload.get("episodes", [])) != count
    ):
        raise ValueError("Result does not cover the exact requested manifest range")
    for row, entry in zip(payload["episodes"], expected):
        if (
            row.get("episode") != entry["episode"]
            or row.get("seed") != entry["seed"]
            or row.get("instruction") != entry["instructions"][plan["instruction_type"]]
            or type(row.get("success")) is not bool
        ):
            raise ValueError("Result episode/seed/instruction/outcome mismatch")
    if payload.get("successes") != sum(row["success"] for row in payload["episodes"]):
        raise ValueError("Result success count mismatch")
    return payload


def client_environment(plan, gpu, directory):
    native = Path(plan["robotwin_root"])
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        ROBOTWIN_PATH=str(native),
        RAX_ROOT=plan["rax_root"],
        ASSETS_PATH=str(native),
        ROBOTWIN_RUNTIME_ROOT=str(directory / "runtime"),
        MPLCONFIGDIR=str(directory / "matplotlib"),
        OMP_NUM_THREADS="4",
    )
    env["PYTHONPATH"] = ":".join(
        [str(native), str(ROOT / "benchmarks/robotwin"), str(ROOT), plan["rax_root"], env.get("PYTHONPATH", "")]
    )
    env.pop("ROBOTWIN_ENABLE_PLANNER_FALLBACK", None)
    env.pop("ROBOTWIN_SKIP_VERSION_CHECK", None)
    if plan.get("vulkan_icd"):
        # Select the existing headless NVIDIA EGL ICD for this child only.
        # Some compute images have GLX installed but lack libEGL.so.1, causing
        # its default Vulkan ICD to fail even when CUDA works correctly.
        env["VK_ICD_FILENAMES"] = plan["vulkan_icd"]
        env["VK_DRIVER_FILES"] = plan["vulkan_icd"]
    egl = (
        Path(plan["robotwin_python"]).parent.parent
        / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    )
    if egl.exists() and not env.get("__EGL_VENDOR_LIBRARY_FILENAMES") and not env.get("__EGL_VENDOR_LIBRARY_DIRS"):
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    return env


class Gate:
    def __init__(self, plan):
        self.plan = plan
        self.root = Path(plan["run_root"])
        self.stop = threading.Event()
        self.summary_lock = threading.Lock()
        self.policy_config = self.root / "policy.yml"
        self.deploy_config = self.root / "deploy.yml"
        self.model_dir = self.root / "model"

    def state(self, phase, **extra):
        atomic_json(self.root / "state.json", dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **extra))
        print(json.dumps(dict(phase=phase, **extra), ensure_ascii=False), flush=True)

    def execute(self, command, directory, env, timeout):
        directory.mkdir(parents=True, exist_ok=True)
        registry = ProcessRegistry()
        with (directory / "client.log").open("w") as log:
            process = spawn_resource(command, cwd=ROOT, env=env, stdout=log)
            registry.add(process)
            try:
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if self.stop.wait(0.5):
                        raise RuntimeError("Evaluation interrupted")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Client timeout: {directory}")
                if process.returncode:
                    raise RuntimeError(f"Client exit {process.returncode}: {directory}/client.log")
            finally:
                registry.terminate_all()

    def command(self, task, operation, directory):
        return [
            self.plan["robotwin_python"],
            str(ROOT / "benchmarks/robotwin/c2r_eval_client.py"),
            "labtasker",
            "--task",
            task,
            "--mode",
            self.plan["mode"],
            "--policy-config",
            str(self.policy_config),
            "--operation",
            operation,
            "--instruction-type",
            self.plan["instruction_type"],
            "--result-file",
            str(directory / "result.json"),
            "--progress-file",
            str(directory / "progress.json"),
            "--total-episodes",
            str(self.plan["episodes_per_task"]),
            "--render-device",
            "cuda:0",
        ]

    def probe_renderers(self):
        def probe(gpu):
            directory = self.root / "render_probes" / f"gpu{gpu}-{time.time_ns()}"
            command = [
                self.plan["robotwin_python"],
                str(ROOT / "benchmarks/robotwin/c2r_eval_client.py"),
                "--render-probe",
            ]
            self.execute(command, directory, client_environment(self.plan, gpu, directory), 120)

        with ThreadPoolExecutor(max_workers=len(self.plan["gpus"])) as executor:
            futures = [executor.submit(probe, gpu) for gpu in self.plan["gpus"]]
            for future in as_completed(futures):
                future.result()

    def build_manifest(self, task, gpu):
        path = self.root / "manifests" / f"{task}.json"
        if path.exists():
            return validate_manifest(load_manifest(path), task, self.plan)
        for attempt in range(1, 3):
            directory = self.root / "manifest_attempts" / task / f"{time.time_ns()}-{attempt}"
            try:
                self.execute(
                    self.command(task, "build_manifest", directory),
                    directory,
                    client_environment(self.plan, gpu, directory),
                    self.plan["manifest_timeout_seconds"],
                )
                payload = validate_manifest(json.loads((directory / "result.json").read_text()), task, self.plan)
                write_manifest(path, payload)
                return payload
            except Exception as exc:
                atomic_json(directory / "error.json", {"error": str(exc), "attempt": attempt})
                if self.stop.is_set() or attempt == 2:
                    raise

    def evaluate_range(self, task, server, start, count):
        manifest = validate_manifest(load_manifest(self.root / "manifests" / f"{task}.json"), task, self.plan)
        for attempt in range(1, 3):
            directory = self.root / "eval_attempts" / task / f"{start}-{count}-{time.time_ns()}-{attempt}"
            try:
                server.start()
                command = self.command(task, "run_eval", directory) + [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(server.port),
                    "--episode-start",
                    str(start),
                    "--num-episodes",
                    str(count),
                    "--manifest-file",
                    str(self.root / "manifests" / f"{task}.json"),
                    "--manifest-hash",
                    manifest["manifest_hash"],
                ]
                self.execute(
                    command,
                    directory,
                    client_environment(self.plan, server.gpu, directory),
                    self.plan["eval_timeout_seconds"],
                )
                payload = validate_result(
                    json.loads((directory / "result.json").read_text()), manifest, start, count, self.plan
                )
                for row in payload["episodes"]:
                    atomic_json(
                        self.root / "results" / task / f"episode_{row['episode']:03d}.json",
                        dict(row, task=task, manifest_hash=manifest["manifest_hash"], artifact_dir=str(directory)),
                    )
                self.summarize()
                return
            except Exception as exc:
                atomic_json(directory / "error.json", {"error": str(exc), "attempt": attempt})
                server.close()
                if self.stop.is_set() or attempt == 2:
                    raise

    def completed_prefix(self, task):
        manifest = load_manifest(self.root / "manifests" / f"{task}.json")
        paths = sorted((self.root / "results" / task).glob("episode_*.json"))
        for index, path in enumerate(paths):
            row = json.loads(path.read_text())
            entry = manifest["entries"][index]
            if (
                row["episode"] != index
                or row["seed"] != entry["seed"]
                or type(row["success"]) is not bool
                or row["instruction"] != entry["instructions"][self.plan["instruction_type"]]
                or row["manifest_hash"] != manifest["manifest_hash"]
            ):
                raise ValueError("Existing results do not form the same manifest prefix")
        return len(paths)

    def summarize(self):
        with self.summary_lock:
            tasks = []
            for task in self.plan["tasks"]:
                rows = [
                    json.loads(p.read_text()) for p in sorted((self.root / "results" / task).glob("episode_*.json"))
                ]
                count = len(rows)
                successes = sum(r["success"] for r in rows)
                tasks.append(
                    dict(
                        task=task,
                        completed_episodes=count,
                        expected_episodes=10,
                        successes=successes,
                        success_rate=successes / count if count else None,
                        complete=count == 10,
                    )
                )
            total = sum(t["completed_episodes"] for t in tasks)
            successes = sum(t["successes"] for t in tasks)
            complete = all(t["complete"] for t in tasks)
            summary = dict(
                updated_unix=time.time(),
                mode=self.plan["mode"],
                instruction_type=self.plan["instruction_type"],
                completed_episodes=total,
                expected_episodes=500,
                completed_tasks=sum(t["complete"] for t in tasks),
                successes=successes,
                complete=complete,
                success_rate=successes / 500 if complete else None,
                partial_success_rate=successes / total if total else None,
                tasks=tasks,
            )
            atomic_json(self.root / "summary.json", summary)
            temporary = self.root / "task_summary.csv.tmp"
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
                writer.writeheader()
                writer.writerows(tasks)
            os.replace(temporary, self.root / "task_summary.csv")
            return summary

    def pool(self, operation, servers=None):
        # A worker owns one GPU and policy endpoint for its entire task shard.
        failures = []

        def worker(index, gpu):
            server = None if servers is None else servers[index]
            try:
                for task in self.plan["tasks"][index :: len(self.plan["gpus"])]:
                    if self.stop.is_set():
                        return
                    try:
                        atomic_json(
                            self.root / f"worker_{gpu}.json",
                            {"operation": operation, "task": task, "gpu": gpu, "updated_unix": time.time()},
                        )
                        if operation == "build_manifest":
                            self.build_manifest(task, gpu)
                        else:
                            start = self.completed_prefix(task)
                            if start < 10:
                                self.evaluate_range(task, server, start, 10 - start)
                    except Exception as exc:
                        record = {"operation": operation, "task": task, "error": str(exc)}
                        atomic_json(self.root / "errors" / f"{operation}_{task}.json", record)
                        failures.append(record)
            finally:
                if server is not None:
                    server.close()

        with ThreadPoolExecutor(max_workers=len(self.plan["gpus"])) as executor:
            futures = [executor.submit(worker, i, gpu) for i, gpu in enumerate(self.plan["gpus"])]
            for future in as_completed(futures):
                future.result()
        if failures or self.stop.is_set():
            raise RuntimeError(f"{operation}: {len(failures)} failed tasks; see errors/")

    def freeze_model(self, checkpoint):
        self.model_dir.mkdir(exist_ok=True)
        destination = self.model_dir / checkpoint.name
        if not destination.exists():
            os.link(checkpoint, destination)  # same shared volume; no duplicate 25 GB download/copy
        elif not os.path.samefile(destination, checkpoint):
            raise ValueError("Existing model snapshot is not the requested final checkpoint")
        source = Path(self.plan["train_output"])
        for name in ("config.yaml", "normalization_stats.npy"):
            shutil.copy2(source / name, self.model_dir / name)
        if not (self.model_dir / "tokenizer").exists():
            shutil.copytree(source / "tokenizer", self.model_dir / "tokenizer")
        atomic_json(
            self.root / "checkpoint.json",
            dict(
                source=str(checkpoint),
                snapshot=str(destination),
                sha256=sha256(destination),
                **validate_safetensors(destination),
            ),
        )

    def run(self):
        import yaml

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "gate.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: self.stop.set())
            bind_json(self.root / "resolved_plan.json", self.plan)
            bind_json(self.root / "preflight.json", preflight(self.plan))
            policy = yaml.safe_load((ROOT / "benchmarks/robotwin/policy_config.yml").read_text())
            policy.update(
                action_type="ee",
                state_dim=20,
                send_state=True,
                request_timeout=1800,
                debug=False,
                instruction_type=self.plan["instruction_type"],
            )
            self.policy_config.write_text(yaml.safe_dump(policy))
            shutil.copy2(ROOT / "configs/deploy.yaml", self.deploy_config)
            self.summarize()
            try:
                while True:
                    checkpoint = training_ready(self.plan)
                    busy = gpu_processes(self.plan["gpus"]) if checkpoint else []
                    if checkpoint and not busy:
                        break
                    self.state("waiting_for_gpus" if checkpoint else "waiting_for_training", gpu_processes=busy)
                    if self.stop.wait(self.plan.get("poll_seconds", 30)):
                        raise RuntimeError("Evaluation gate interrupted")
                # Recheck after the wait: edits made while training must not
                # silently change an already armed benchmark protocol.
                bind_json(self.root / "preflight.json", preflight(self.plan))
                self.state("probing_renderers")
                self.probe_renderers()
                self.state("freezing_checkpoint", checkpoint=str(checkpoint))
                self.freeze_model(checkpoint)
                first = self.plan["tasks"][0]
                self.state("building_smoke_manifest", task=first, episodes_per_task=10)
                self.build_manifest(first, self.plan["gpus"][0])
                servers = [
                    PolicyServer(self, gpu, self.plan["base_port"] + i) for i, gpu in enumerate(self.plan["gpus"])
                ]
                try:
                    self.state("smoke_eval", task=first, episodes=1)
                    if not self.completed_prefix(first):
                        self.evaluate_range(first, servers[0], 0, 1)
                    # A completed failed rollout is a valid benchmark outcome.
                    # Reuse its result as episode 0; it is never filtered out.
                    atomic_json(
                        self.root / "smoke_passed.json",
                        {
                            "passed": True,
                            "task": first,
                            "episode": 0,
                            "counts_toward_500": True,
                            "requires_task_success": False,
                        },
                    )
                    servers[0].close()
                    self.state("building_manifests", tasks=50, episodes_per_task=10)
                    self.pool("build_manifest")
                    self.state("evaluating", tasks=50, episodes_per_task=10)
                    self.pool("run_eval", servers)
                finally:
                    for server in servers:
                        server.close()
                summary = self.summarize()
                if not summary["complete"]:
                    raise RuntimeError("Evaluation returned without all 500 episode verdicts")
                self.state(
                    "complete", successes=summary["successes"], episodes=500, success_rate=summary["success_rate"]
                )
            except BaseException as exc:
                self.summarize()
                self.state("failed", error=f"{type(exc).__name__}: {exc}")
                raise


class PolicyServer:
    def __init__(self, gate, gpu, port):
        self.gate = gate
        self.gpu = gpu
        self.port = port
        self.registry = ProcessRegistry()
        self.process = None

    def close(self):
        self.registry.terminate_all()
        self.process = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        from benchmarks.utils.transport import WSPolicyClient

        token = secrets.token_hex(24)
        directory = self.gate.root / "policy" / f"gpu{self.gpu}-{time.time_ns()}"
        directory.mkdir(parents=True)
        command = [
            self.gate.plan["server_python"],
            str(ROOT / "scripts/deploy.py"),
            "--config",
            str(self.gate.deploy_config),
            "--ckpt-dir",
            str(self.gate.model_dir),
            "--ckpt-name",
            f"checkpoint_step_{self.gate.plan['expected_steps']}.safetensors",
            "--device",
            "cuda:0",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--readiness-token",
            token,
        ]
        atomic_json(directory / "command.json", {"argv": command[:-2], "gpu": self.gpu})
        log = (directory / "server.log").open("w")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(self.gpu),
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="8",
            TORCHINDUCTOR_CACHE_DIR=str(self.gate.root / "compile_cache" / f"gpu{self.gpu}"),
        )
        self.process = spawn_resource(command, cwd=ROOT, env=env, stdout=log)
        self.registry.add(self.process, log)
        try:
            deadline = time.monotonic() + self.gate.plan["server_start_timeout_seconds"]
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Policy server exited; see {directory}/server.log")
                if self.gate.stop.is_set():
                    raise RuntimeError("Evaluation interrupted")
                try:
                    with WSPolicyClient(f"ws://127.0.0.1:{self.port}", timeout=2, open_timeout=2) as client:
                        response = client.ping()
                    if response.get("type") != "pong" or response.get("readiness_token") != token:
                        raise ValueError("Policy endpoint identity mismatch")
                    return
                except (ConnectionRefusedError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Policy server startup timeout: {directory}")
                self.gate.stop.wait(1)
        except BaseException:
            self.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.check_only:
        print(json.dumps(preflight(plan), indent=2))
    elif args.summarize:
        print(json.dumps(Gate(plan).summarize(), indent=2))
    else:
        Gate(plan).run()


if __name__ == "__main__":
    main()
