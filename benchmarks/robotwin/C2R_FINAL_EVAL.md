# Final clean-to-random evaluation

`scripts/robotwin_c2r_gate.py` waits for the existing five-epoch training gate to
report successful completion, validates step 42970 and the complete final
safetensors payload, then waits for all configured GPUs to become idle. It does
not initialize CUDA, load a model, or run simulation while training is active.
The final optimizer state is deliberately not required: training removes it on
successful completion.

The benchmark covers all 50 tasks from `labtasker_runtime.ROBOTWIN_TASKS`, with
10 `demo_randomized` episodes per task and `unseen` instructions (500 outcomes).
The installed native RoboTwin commit and configuration hashes are recorded. Its
own scene sampling, expert eligibility check, success predicates, and step limits
are used. Starting at native test seed 100000, the first ten expert-eligible
seeds for each task are sealed before that task's model evaluation. Both instructions and
seeds remain fixed for retries and resumed runs.

After training, a rendered-image probe checks every GPU. The first task's
manifest and one real policy episode check the deployment and simulation path
before the remaining manifests are built on four workers. That episode counts
toward the 500 outcomes regardless of success. The original gate waits for every
manifest before starting full evaluation. The current run uses the pipeline
scheduler below, so ready tasks can be evaluated while other scenes are prepared.
Each model worker evaluates one task on its GPU. Each server uses
the same final weights, ten synchronous denoising steps, the full action chunk,
and the repository's deploy optimization defaults, copied to `deploy.yml`.

Native `collect_data` and `save_data` are disabled in a private runtime copy;
random rollouts never enter the training dataset. Policy videos remain enabled.
The installed revision's legacy `place_fan` reward constructor uses the same
narrow RAX compatibility adapter as clean data collection. No planner fallback
is enabled. Local and remote source trees and native task configuration stay
unchanged by evaluation.

The H200 image's default GLX Vulkan entry fails because `libEGL.so.1` is absent.
The plan explicitly selects its already installed NVIDIA EGL ICD with child-only
`VK_ICD_FILENAMES`/`VK_DRIVER_FILES`. No driver install or global environment edit
is needed. The original failed run remains at `c2r_random50x10_final`; the
corrected run uses `c2r_random50x10_final_v2` and retains the same 500-episode
protocol. GPU rendering probes run after training, before any expert scan.

## Launch

Use a plan based on `c2r_final_50x10.plan.json`; adjust absolute environment paths
before creating a new run directory. The current plan targets the existing
four-GPU H200 training run.

```bash
python scripts/robotwin_c2r_gate.py --plan benchmarks/robotwin/c2r_final_50x10.plan.json --check-only
python scripts/robotwin_c2r_gate.py --plan benchmarks/robotwin/c2r_final_50x10.plan.json
```

Launch with the model environment's Python, which provides YAML, Pillow, NumPy
and WebSockets. Native clients use the separate `robotwin_python` from the plan.
An OS lock prevents duplicate gates. Only processes spawned by this gate are
terminated on cancellation or infrastructure failure; unrelated jobs are never
stopped. No Hugging Face identity or global configuration is changed.

## Current run: pipeline handoff

`scripts/robotwin_c2r_pipeline.py` removes the global manifest barrier and static
task shards. Idle GPUs take ready tasks from a shared queue. Workers also claim
remaining scene-preparation tasks, so slow expert checks do not hold up unrelated
model evaluations. Each model episode is validated, committed, and summarized
immediately. Completed episodes are never rerun merely to finish a task batch.

The live `c2r_random50x10_final_v2` run was handed over on 2026-10-30 at 09:57 UTC.
Its existing expert process was retained, as were all sealed manifests and the
initial smoke outcome. The handoff pauses only the legacy coordinator; its
resource guardians and native clients continue. After all inherited guardians
exit, the pipeline retires the paused coordinator. PID start times and command
lines protect against acting on a reused or unrelated PID.

For a legacy run still in `building_manifests`, capture its handoff once, then
start the pipeline using the model environment's Python:

```bash
python scripts/robotwin_c2r_pipeline.py --plan benchmarks/robotwin/c2r_final_50x10.plan.json --capture-handoff
python scripts/robotwin_c2r_pipeline.py --plan benchmarks/robotwin/c2r_final_50x10.plan.json
```

For the current run, the handoff already exists and the pipeline is running.
Do not start the legacy gate or resume its paused coordinator. If the pipeline
exits, the second command resumes committed episode prefixes using the existing
handoff. `pipeline.lock` prevents duplicate pipeline instances. The scheduler
records its source hash in `pipeline_provenance.json`; benchmark settings, model
weights, sealed seeds, native eligibility and success criteria remain unchanged.

## Results and recovery

- `state.json`, `gate.log`, `pipeline.log`: phase and launcher status.
- `pipeline_handoff.json`, `pipeline_provenance.json`: scheduler handoff and source identity.
- `worker_*.json`, attempt `progress.json`/`client.log`: current task and rollout progress.
- `checkpoint.json`, `preflight.json`, `resolved_plan.json`: weights and protocol provenance.
- `manifests/*.json`: sealed seeds and both instruction variants.
- `results/<task>/episode_*.json`: validated episode outcomes.
- `eval_attempts/`: native videos, logs and raw results.
- `summary.json`, `task_summary.csv`: per-task scores and overall success rate.

The overall score is withheld until all 500 episode outcomes exist. Ordinary
task failures count as failures and are not retried. Infrastructure errors get
one retry with the same seeds; persistent errors leave the run incomplete and
are reported separately. The original gate reruns a partial task batch in full
after a client crash. The pipeline uses one-episode batches, so an infrastructure
retry affects only that episode, never previous committed outcomes.
Re-running the same scheduler resumes completed task prefixes. Changed settings
or relevant source hashes require a new evaluation directory.

CPU regression checks:

```bash
python -m pytest -q tests/test_robotwin_c2r_gate.py tests/test_c2r_pipeline.py tests/test_c2r_progress.py
```

The first actual GPU rollout intentionally runs only after training finishes;
CPU checks do not claim to validate final model inference or task success.

## Local live progress page

Run on the local development machine:

```bash
python scripts/serve_c2r_progress.py --port 8767 --interval 10
```

Open `http://127.0.0.1:8767`. The server reads the current run selected by the
remote plan over a dedicated SSH connection every ten seconds. It starts no
remote web service and does not write evaluation files or read videos. Only the
loopback HTTP listener is exposed. `--host`, `--ssh-port`, and `--remote-plan`
can select another already authorized experiment server.

The page separates expert scene preparation from model outcomes. Success rate
uses ended model episodes, never expert eligibility successes. Active batches'
ended episodes are shown as provisional until committed, without double counting
the initial smoke episode or completed batches. Infrastructure retries discard
only the failed attempt's provisional count. SSH outages retain the last good
snapshot with a visible disconnected notice and its timestamp.

The local launch record and log for the current session are under
`../artifacts/c2r_evaluation/progress_monitor/`. CPU checks for progress aggregation
and disconnect handling are in `tests/test_c2r_progress.py`.
