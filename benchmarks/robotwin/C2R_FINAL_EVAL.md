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

The current run was stopped at the user's request on 2026-10-30 after 64
committed episodes (19 successes). Its partial results, manifests and videos
remain in place; this is not a complete C2R score. All owned evaluation processes
were stopped. Do not restart this run or its legacy coordinator. For a different
interrupted run that is intended to continue, the second command resumes
committed episode prefixes using the existing handoff. `pipeline.lock` prevents duplicate pipeline instances. The scheduler
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


## Following run: counterfactual acceptance and new training

`robotwin_acceptance_workflow.py` uses `c2r_acceptance_v2.plan.json` to wait for
all 500 outcomes in the current benchmark. The next run has its own source
snapshots, registry, evidence, dataset and training directory. It reads the old
benchmark and 2,500 clean recordings without changing them. It waits for the
selected GPUs to become idle and never terminates unrelated work.

Each template actually needed by the 50 × 50 recordings must pass the new
acceptance protocol on held-out **clean** scenes. The initial pilot covers
`adjust_bottle`, `beat_block_hammer`, and `lift_pot`. Each template receives
three distinct scenes, two repeated baseline replays per scene, and position,
orientation, gripper-timing and coupled perturbations. Raw measured trajectories,
commands, native task outcomes and hashes remain available for independent
reassessment. Planning failures, incomplete trajectories and interventions not
realized by the robot do not count as task negatives.

The frozen pilot requires successful variations on every held-out scene, at
least six genuine task failures spanning at least two perturbation families,
zero accepted failures, and no more than 20% rejected successful variations.
This is evidence about these finite cases, not a proof that every constraint-
satisfying action succeeds. All planned outcomes are retained. A failed pilot
or template blocks publication and training; the workflow does not relax its
thresholds or silently revert to legacy labels. Repair requires reviewing the
recorded counterexamples and making a new version of the affected experiment.

When every used template passes, the publisher creates separate-inode copies
of all 2,500 recordings and embeds the accepted constraints and their proof.
The original actions, images, instructions and distinct clean initial states
are retained; random evaluation rollouts are never training data. The new
dataset appears atomically only after its labels, hashes and evidence validate.
Training revalidates the publication, audits the complete dataset, checks actual
action decoding and gradients, and runs one optimizer-step smoke test plus a
resume smoke test before starting five epochs with global batch 256 on four
H200s. It starts from the same Foundation checkpoint as the previous experiment.
The published dataset itself is the sealed training snapshot.

```bash
python scripts/robotwin_acceptance_workflow.py --plan benchmarks/robotwin/c2r_acceptance_v2.plan.json --check-only
python scripts/robotwin_acceptance_workflow.py --plan benchmarks/robotwin/c2r_acceptance_v2.plan.json
```

If the user explicitly stops the current benchmark before 500 outcomes, pass
`--benchmark-stop-receipt /absolute/new-run/benchmark_stop_receipt.json` to either
command. The workflow does not stop processes itself or mark the benchmark
complete. Its state must already be `stopped_by_user`. The receipt contains
`version: 1`, `kind: "user_requested_benchmark_stop"`, the exact `benchmark_root`
and `run_root`, `stopped_unix`, and SHA256 hashes of the benchmark's unchanged
`summary.json` and `user_stop.json` in `summary_sha256` and
`benchmark_stop_receipt_sha256`. The latter evidence file must repeat the same
identity, stop time, and summary hash. A failed/running benchmark or mismatched
hash is rejected. The workflow freezes the receipt content, source path and raw
hash in `benchmark_release.json`, which also permits a verified restart without
repeating the CLI flag. Check-only does not write this release. The frozen plan,
incomplete benchmark results and all acceptance/training gates remain intact.

The plan's code paths must point to isolated snapshots under the new run root.
`state.json`, `acceptance_progress.json`, `acceptance_blocked.json`,
`acceptance/<task>/<template>/`, `publication.log`, and
`training_run/state.json` identify the phase and any blocking evidence. A file
lock prevents duplicate workflows. Resource guardians close only this
workflow's children when its launcher exits. Re-running a completed workflow
does not start a second training; interrupted training requires inspection.

Code changes, rule checks, evidence reassessment and dataset publication use
CPU. Actual native counterfactual replay needs simulator GPU access. As of
2026-10-30, the prior same-size training took 141h 45m; the next training budget
is 142–156h. A provisional 24–48h acceptance budget must be calibrated by the
first native pilot. It is not measured throughput and excludes unknown template
repair cycles. A new model C2R benchmark is a separate step after training.


### Native EEF frame correction found during acceptance

The first acceptance baselines succeeded natively but exceeded the 5 mm tracking
ceiling. The installed native cuRobo adapter subtracts the arm-base translation
while omitting its fixed URDF rotation (left yaw 0.02 rad, right yaw 0.01 rad).
Zero-parameter prediction from that omitted transform explains 95–96% of the
active-left-arm position error on the three pilot baselines. A private complete
SE(3) conversion was verified on the same three frozen first-seed baselines:
maximum active-arm error fell from 7.9–11.4 mm to 2.1–3.1 mm, with native task
success retained. The installed native source and stopped evaluation remain
unchanged. Formal acceptance uses that adapter only during trial replay; expert
reference capture remains native.

Future baseline-versus-new-model C2R comparisons must record and use the same
controller/frame adapter for both models. The incomplete legacy result alone
cannot distinguish supervision effects from this controller error. Acceptance
attempts retain their own code, measured trajectories, native outcomes and
adapter identity so a correction cannot silently rewrite earlier evidence.
