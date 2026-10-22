# RoboTwin clean-only constraint supervision

This experimental training objective adds REACH, CLEARANCE and ORIENTATION
supervision to OpenWAM's existing action/video flow matching. It is disabled by
default. The first implementation supports `dual_system / joint_self_attn`,
RoboTwin `clean_50`, training split, and absolute EEF actions (20D or their
80D unified mapping). It uses existing demonstrations and requires no RAX
runtime or new trajectory collection.

The local development machine only needs code checks and small CPU tests.
Model weights, training dependencies, RoboTwin assets and simulation belong on
the GPU server. No download or installation is performed by this feature.

## Loss and scope

For the existing action flow convention,
`a_sigma = (1-sigma) * action + sigma * noise`, the model predicts velocity
`v = noise - action`. The differentiable clean estimate is
`a_hat = a_sigma - sigma * v_pred`. The callback runs after the forward pass,
outside the reader and `prepare_inputs()`'s no-grad context.

The decoder first undoes the unified action mapping, then applies the reader's
exact inverse affine normalization. Geometry is evaluated in float32:

- REACH: distance outside a world-coordinate target ball; zero anywhere inside.
- CLEARANCE: shortfall of the **lowest transformed object vertex** below a
  world z threshold. Requires object geometry and an independently verified
  rigid EEF-to-object transform valid throughout the annotated interval.
- ORIENTATION: angular error of one local axis against a world target axis,
  beyond a tolerance. Rotation about that axis remains free.

The constraint loss averages squared, scale-normalized residuals across valid
frames within each active constraint, then averages active constraints across
the distributed microbatch. The configured weight multiplies this mean. The
original action FM and video FM terms retain their weights. Annotation density
and sliding-window overlap therefore affect sampling; use the same windows and
manifest in comparative runs. Gradient accumulation averages microbatch losses.

These are geometric losses on a **single-step estimate from noised training
actions**. They do not differentiate through contact dynamics or guarantee
execution success. Object supervision assumes the verified rigid grasp persists;
exclude slip, grasp transitions and unknown attachment intervals. Position units,
world frame, tool origin and quaternion conventions must match the dataset.

## Annotation contract

Supply a JSONL manifest: one constraint per line. `episode_id` is the canonical
POSIX path relative to `dataloader.dataset_dir`, including task, embodiment,
variant and HDF5 filename. Intervals use raw observation indices `[start, end)`;
they are unrelated to the video subsampling stride. A sample beginning at `s`
predicts actions for frames `s+1, s+2, ...`. An endpoint constraint applies only
to `end-1`; it is never moved to the end of a shorter window. Padding and
unobserved action dimensions are excluded.

Illustrative **proposed** label (coordinates and report IDs must come from your
own annotations; this example is not verified training data):

```json
{
  "schema_version": "0.1",
  "episode_id": "adjust_bottle/aloha-agilex_clean_50/data/episode0.hdf5",
  "segment_id": "reach-0",
  "primitive": "REACH",
  "interval": {"start": 10, "end": 31},
  "constraint": {
    "type": "REACH",
    "version": "1",
    "subject": "robot_eef:left",
    "frame": "world",
    "scope": "endpoint",
    "parameters": {"center_m": [0.1, 0.2, 0.8], "radius_m": 0.03, "scale_m": 0.03}
  },
  "evidence": {"source": "manual", "verification_status": "proposed"},
  "provenance": {"split": "train", "variant": "clean_50"}
}
```

Accepted labels require a nonempty `evidence.report_id`. This is a reference to
verification evidence, not proof supplied by the parser. The parser validates
the contract; it does not verify the physical truth of annotations or generate
simulator ground truth from existing HDF5 files. Proposed, rejected and unknown
labels are counted but excluded from the loss. A manifest without any accepted
labels, or with no accepted labels matching selected episodes, is rejected.
`demonstration_proxy` labels require the explicit
`training.constraints.allow_demonstration_proxy=true` opt-in and must be reported
separately from simulator-ground-truth experiments.

| Type | Required parameters | Subjects |
| --- | --- | --- |
| REACH | `center_m[3]`, `radius_m >= 0`, `scale_m > 0` | `robot_eef:left`, `robot_eef:right`, `object:<id>` |
| CLEARANCE | `safe_height_m`, `scale_m > 0` | `object:<id>` only |
| ORIENTATION | unit `local_axis[3]`, unit `target_axis[3]`, `0 <= tolerance_rad < pi`, `scale_rad > 0` | EEF or object |

Object subjects additionally require `constraint.attachment`:

```json
{
  "arm": "left",
  "position_m": [0, 0, -0.1],
  "quaternion_wxyz": [1, 0, 0, 0],
  "vertices_m": [[-0.02, -0.02, -0.05], [0.02, 0.02, 0.05]],
  "verification_status": "accepted",
  "report_id": "replace-with-rigid-grasp-verification-reference"
}
```

This too is a schema example. Supply the full relevant object geometry in its
local frame, not just these illustrative points. The attachment pose maps the
object frame into the EEF frame. Its quaternion is **wxyz**; RoboTwin HDF5
EEF quaternions use **xyzw** and the reader converts them into rotation-6D
columns. All labels are training-only metadata and never enter policy inputs.

## GPU-server launch

Use the author's existing training environment and a local copy of the
`OpenWAM-Alpha-Sim-RoboTwin-Clean2Random` checkpoint for a rapid warm-start
experiment. This does not repeat foundation pretraining or clean fine-tuning.
Do not substitute the Full checkpoint: it includes randomized training data.

First run the integration tests in that environment (no weights needed):

```bash
python -m pytest tests/test_constraint_supervision.py tests/test_flow_match_loss.py tests/test_openwam_trainer.py -q
```

After preparing real, verified labels, run a 20-step smoke test. The paths below
are placeholders for files already on the server. Batch size 1 is a conservative
smoke-test setting, not a measured H200 throughput choice.

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/data/robotwin \
  dataloader.variant=clean_50 \
  dataloader.split=train \
  dataloader.action_mode=eef \
  model.architecture.variant=joint_self_attn \
  training.finetune_ckpt_path=/models/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random \
  training.constraints.manifest=/data/labels/verified_constraints.jsonl \
  training.constraints.weight=0.1 \
  training.batch_size=1 \
  training.debug=true \
  training.output_path=outputs/c2r_constraints_smoke
```

The weight `0.1` is an initial diagnostic setting, not a tuned recommendation.
Run the matched continued-FM control from the **same checkpoint**, seed, data,
batching and schedule with `training.constraints.weight=0`. Also retain the
untouched checkpoint's evaluation as a reference. A short smoke run checks
execution and gradient health; it is not evidence of improved benchmark scores.

Each constraint run archives the exact manifest, its SHA-256, source/status
counts, matched count, weight and gradient-entry convention. Resume refuses
changes to this objective; use a new fine-tuning run for a changed ablation.
Metrics include physical residual means/maxima and violation rates for each
observed type. Unobserved types report a count of zero and omit residual/rate
metrics; an absent label is not a satisfied constraint.

## Evaluation

Use the existing [RoboTwin deployment/evaluation workflow](../../benchmarks/robotwin/README.md).
Serve each resulting checkpoint separately. For a selected-task randomized
smoke test, with the policy server already running:

```bash
ROBOTWIN_PATH=/opt/RoboTwin \
ROBOTWIN_PYTHON=/opt/conda/envs/robotwin/bin/python \
ROBOTWIN_TEST_NUM=5 \
bash benchmarks/robotwin/multi_eval.sh \
  -m demo_randomized -n constraint_smoke \
  -d /path/to/resulting/checkpoint adjust_bottle
```

Evaluate control and constraint checkpoints with identical tasks, seeds,
episode counts and deployment settings. The full randomized C2R evaluation
uses `all` tasks and 100 episodes/task; also check clean performance for
regressions. Do not use held-out randomized evaluation episodes to construct
training constraints or tune labels.

RAX remains an optional later data/program generation source. This first stage
does not include automatic simulator annotation extraction, an agent verifier,
an E1/E2 sparse-action ablation, or a policy-feedback collection loop.
