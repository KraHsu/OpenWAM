# Collecting a real policy trace on one GPU

`scripts/run_constraint_policy_probe.py --plan PLAN.json` owns one model server
and serial RoboTwin clients. Separate server and client GPUs remain the default.
Explicit shared mode adds these plan fields:

```json
{
  "server_gpu": 3,
  "client_gpu": 3,
  "allow_shared_gpu": true,
  "lease_root": "/absolute/path/to/.robotwin_dev_gpu_leases",
  "shared_gpu_min_total_mib": 131072,
  "shared_gpu_min_free_mib": 98304,
  "shared_gpu_native_headroom_mib": 24576
}
```

Use the **same lease root as every development replay queue**. The probe takes
`gpu_3.lock` before checking occupancy or loading the model. It rejects foreign
GPU processes. After loading, only descendants of its exact owned guardian
processes may already occupy the card. The server and client guardians inherit
the lease descriptor, so owner death cannot release the card before guardian
cleanup. Cleanup signals only processes launched by this probe; it never stops
another queue or a foreign GPU process.

Thresholds are in MiB. Built-in minimums are 80 GiB total, 64 GiB initially free,
and 16 GiB still free before each native client. Plans can raise these values.
The initial and post-model checks are recorded in `policy_probe_state.json`;
they reserve practical headroom but cannot guarantee that every later allocation
will fit. A failed check stops and cleans the owned server without starting the
client. A concurrent process outside the cooperative lease protocol can still
arrive later, so this protocol requires other job owners to honor the same lease.

For multiple scenes of one task, use distinct `job_id` values:

```json
{
  "jobs": [
    {"task": "lift_pot", "job_id": "lift_pot_seed10008", "reference": "/frozen/seed10008.json", "source_manifest": "/frozen/experiment_manifest.json"},
    {"task": "lift_pot", "job_id": "lift_pot_seed10010", "reference": "/frozen/seed10010.json", "source_manifest": "/frozen/experiment_manifest.json"}
  ]
}
```

Other required fields retain the split-GPU plan format: `run_root`, `python`,
`robotwin_python`, `openwam_root`, `rax_root`, `robotwin_root`, `vulkan_icd`,
`port`, `checkpoint_receipt`, `model_dir`, `checkpoint_name`, `server_config`.
An optional `frozen_inputs` object maps absolute file paths to SHA256 digests;
all are checked before allocating a GPU resource. The checkpoint is always
checked against its existing receipt. This tool does not download weights or
change Hugging Face configuration.

Each job writes to `run_root/model_traces/<job_id>` (or `<task>` when `job_id`
is omitted). An existing output makes the whole invocation refuse to run: inspect
failed or partial evidence and explicitly choose a new version if another run is
needed. It never silently retries or overwrites a trajectory. The plan is frozen
in the run directory, with ownership and server receipts next to state/logs.

The collector uses a new observation at each reference-length control step.
Every commanded action comes from the model: no expert prefix/suffix, trajectory
projection, resampling, or DTW. It stores raw 20D outputs and records native-domain
gripper clipping separately. Report constraint scores on the effective native
commands as such; a clipped command's score does not prove the raw model output
was feasible. The fixed reference horizon is also distinct from the benchmark's
full native step-limit evaluation. These traces provide development behavior
evidence, not final holdout success-rate certification.
