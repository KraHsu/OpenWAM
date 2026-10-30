"""Read-only, stdlib-only snapshot of a C2R run; also executable through SSH stdin."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def latest_attempt(root, task):
    paths = [p for p in (root / task).glob("*") if p.is_dir()]
    return max(paths, key=lambda p: int(p.name.split("-")[-2]), default=None)


def native_step(directory):
    if directory is None:
        return None
    path = directory / "client.log"
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            tail = stream.read().decode(errors="replace")
    except FileNotFoundError:
        return None
    tail = re.sub(r"\x1b\[[0-9;]*m", "", tail)
    matches = re.findall(r"step:\s*(\d+)\s*/\s*(\d+)", tail)
    return {"current": int(matches[-1][0]), "total": int(matches[-1][1])} if matches else None


def collect(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan["run_root"])
    state = read_json(root / "state.json")
    summary = read_json(root / "summary.json")
    phase = state.get("phase", "unknown")
    expected = plan["episodes_per_task"]
    committed = {row["task"]: row for row in summary.get("tasks", [])}
    active = {}
    if phase in ("building_manifests", "evaluating"):
        operation = "build_manifest" if phase == "building_manifests" else "run_eval"
        for path in root.glob("worker_*.json"):
            worker = read_json(path)
            if worker.get("operation") == operation:
                active[worker["task"]] = worker
    elif phase in ("building_smoke_manifest", "smoke_eval"):
        task = state.get("task", plan["tasks"][0])
        active[task] = dict(
            task=task,
            gpu=plan["gpus"][0],
            operation="build_manifest" if phase == "building_smoke_manifest" else "run_eval",
        )

    rows, workers = [], []
    for task in plan["tasks"]:
        saved = committed.get(task, {})
        confirmed = int(saved.get("completed_episodes", 0))
        wins = int(saved.get("successes", 0))
        count = confirmed
        manifest = read_json(root / "manifests" / f"{task}.json")
        scenes = len(manifest.get("entries", []))
        manifest_attempt = latest_attempt(root / "manifest_attempts", task)
        if not scenes and manifest_attempt and not (manifest_attempt / "error.json").exists():
            scenes = int(read_json(manifest_attempt / "progress.json").get("completed", 0))
        scenes = min(scenes, expected)
        task_state = "complete" if confirmed == expected else "ready" if scenes == expected else "queued"
        worker = active.get(task)
        step = None
        if worker and confirmed < expected:
            building = worker["operation"] == "build_manifest"
            attempt = manifest_attempt if building else latest_attempt(root / "eval_attempts", task)
            failed = attempt is not None and (attempt / "error.json").exists()
            task_state = "retrying" if failed else "preparing" if building else "evaluating"
            if not building and attempt and not failed:
                start, batch_count = map(int, attempt.name.split("-")[:2])
                progress = read_json(attempt / "progress.json")
                done = progress.get("completed", 0)
                succeeded = progress.get("successes", 0)
                end = start + done
                if (
                    type(done) is int
                    and type(succeeded) is int
                    and 0 <= succeeded <= done <= batch_count
                    and end <= expected
                    and start <= confirmed < end
                ):
                    # Earlier committed prefix + this attempt's completed episodes.
                    # Never count the same episode again after a batch is committed.
                    prefix = [read_json(root / "results" / task / f"episode_{i:03d}.json") for i in range(start)]
                    if all(type(p.get("success")) is bool for p in prefix):
                        count = end
                        wins = sum(p["success"] for p in prefix) + succeeded
                step = native_step(attempt)
            workers.append(
                dict(
                    gpu=worker["gpu"],
                    task=task,
                    operation=worker["operation"],
                    status=task_state,
                    completed=scenes if building else count,
                    expected=expected,
                    step=step,
                )
            )
        error = read_json(root / "errors" / f"build_manifest_{task}.json") if scenes < expected else {}
        error = error or (read_json(root / "errors" / f"run_eval_{task}.json") if confirmed < expected else {})
        if error and not worker:
            task_state = "error"
        rows.append(
            dict(
                task=task,
                scenes=scenes,
                expected=expected,
                completed=count,
                confirmed=confirmed,
                provisional=count - confirmed,
                successes=wins,
                failures=count - wins,
                success_rate=wins / count if count else None,
                status=task_state,
                gpu=worker["gpu"] if worker else None,
                step=step,
            )
        )

    total = sum(row["completed"] for row in rows)
    wins = sum(row["successes"] for row in rows)
    confirmed = sum(row["confirmed"] for row in rows)
    pid = state.get("pid")
    alive = bool(pid and (Path("/proc") / str(pid)).exists())
    return dict(
        collected_at=time.time(),
        run=root.name,
        phase=phase,
        process_alive=alive,
        error=state.get("error"),
        mode=plan["mode"],
        instruction_type=plan["instruction_type"],
        expected_tasks=len(rows),
        completed_tasks=sum(row["completed"] == expected for row in rows),
        confirmed_tasks=sum(row["confirmed"] == expected for row in rows),
        expected_episodes=len(rows) * expected,
        completed_episodes=total,
        confirmed_episodes=confirmed,
        provisional_episodes=total - confirmed,
        successes=wins,
        failures=total - wins,
        success_rate=wins / total if total else None,
        final_success_rate=summary.get("success_rate") if phase == "complete" else None,
        scene_episodes=sum(row["scenes"] for row in rows),
        scene_tasks=sum(row["scenes"] == expected for row in rows),
        workers=sorted(workers, key=lambda row: row["gpu"]),
        tasks=rows,
    )


if __name__ == "__main__":
    print(json.dumps(collect(sys.argv[1]), ensure_ascii=False, allow_nan=False))
