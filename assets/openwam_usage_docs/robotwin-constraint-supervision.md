# Embedded geometric constraints

Constraints are part of each recorded HDF5 episode, at
`trajectory_constraints/json`. There are **no operation categories**. RAX writes
and revises base constraints during clean RoboTwin recording. Training reads
the annotations with the action data; it does not invoke RAX, a simulator, an
object dynamics model or a separate annotation manifest.

## Physical quantities and losses

Let the decoded EEF position/orientation/opening be `p, R, g`. A fixed reference
frame has world origin `o` and rotation `Q`; a unit annotation axis `n` becomes
`a = Q n` in world coordinates. For each scalar `u` and interval `[l,h]`:

```
v(u) = relu(l - u) + relu(u - h)
L_i = mean_over_valid_frames((v(u) / scale_i) ** power_i)
L_constraint = sum_i(weight_i * L_i) / sum_i(weight_i)
L_total = L_existing_FM + training.constraints.weight * L_constraint
```

An empty valid set contributes a connected zero, not a false success. Averaging
uses active weights globally across DDP ranks. `power=2` is `squared_hinge`;
`power=4` is `quartic_hinge`. Both give zero inside the allowed interval and
continuous gradients at its boundary. At violation/scale > 1, the fourth power
grows faster. Weight and scale have distinct roles: mixture importance versus
physical normalization. The original action/video FM coefficients remain active.
Consequently, this additive objective does **not** remove imitation penalties
inside the allowed set: zero constraint loss still leaves the original FM loss.
More varied observed actions alone cannot distinguish useful freedom from
competing gradients or tracking errors.

| Base type | Scalar u | Bounds / scale units |
| --- | --- | --- |
| `axis_position_interval` | `a · (p-o)` | metres |
| `radial_distance_interval` | `norm((I-aaᵀ)(p-o))` | metres |
| `point_distance_interval` | `norm(p-o)` | metres |
| `axis_angle_interval` | `atan2(norm((R b) × a), (R b) · a)`; b is the EEF local axis | radians |
| `axis_displacement_interval` | `a · (p_k-p_start)` | metres |
| `angular_displacement_interval` | rotation angle of `R_startᵀ R_k` | radians |
| `gripper_interval` | continuous opening g, 0 closed / 1 open | opening fraction |

For axial position, `dL/du = 2(u-h)/scale²` above h and
`2(u-l)/scale²` below l; it is zero inside. Quartic changes the outside
derivative magnitude to `4 |v|³ / scale⁴`. The position gradient follows the
world axis a. No Euler component interval is used for orientations. Aligning
one local axis leaves rotation around that axis free. Norm/angle constraints
can have symmetric stationary points (zero distance with a positive lower bound,
or exactly antipodal axes); the FM prior remains important.

Constraints compose by conjunction. A cylinder is radial distance bounds plus
an axial interval. Three axial intervals form a box, which is geometrically
different. Use separate IDs and weights; there are no hidden semantic groups,
category-to-loss switches or executable agent-authored losses.

## Activation, masks and action decoding

Indices are raw recorded observation frames `[start,end)`. A reader window at
frame s contains measured proprio at s and predicted targets at s+1, s+2, … .

- `interval`: evaluate valid future frames in the intersection with the chunk.
- `endpoint`: evaluate exactly end-1, never clip it to the chunk's last frame.
- `reference_event`: evaluate at the annotated frame, after checking the
  specified increasing/decreasing gripper threshold crossing on the **reference**
  frame pair. Generated gripper values cannot disable this constraint. Add a
  separate gripper interval if opening itself must be supervised.
- Displacement constraints with the default `parameters.anchor=predicted_start`
  need activation.start inside the chunk or at the
  measured initial state. Missing starts, event pairs, endpoints or required
  dimensions mask the corresponding constraint.

Schema 1.2 adds explicit `parameters.anchor=reference_frame` to displacement
types. Axial displacement then uses `a · (p-o)`; angular displacement uses the
rotation angle of `Qᵀ R`. These fixed measured anchors remain usable when the
original demonstration frame is outside the action chunk. They do **not** claim
to measure a change from an unobserved generated starting pose. Publication
policy v2 converts spans longer than 32 actions to this representation, preserving
the reference bounds and activation, and records the change in each rationale
and the HDF5 revision history. The training gate rejects unreachable endpoint
constraints instead of silently masking them throughout training.

`training.constraints.execution_horizon=null` uses the complete predicted
chunk. Set a positive integer for an executed-prefix ablation. No claim is made
that the full predicted tail is actually executed by a receding-horizon policy.

`training.constraints.require_full_coverage=true` (default) requires a
chunk-safe position/motion constraint for every action frame of both arms.
Gripper-only intervals and displacements requiring distant anchors do not count.
Orientation and gripper freedoms remain task-specific. Use `false` only for an
explicit sparse ablation, not the full-coverage experiment.
This is temporal coverage, not geometric completeness: a single axial position
interval can cover a frame while leaving the other axes unconstrained. It does
not certify object contact, collision avoidance, progress, or task success.

Schema 1.1 adds `motion_direction_cone`: for adjacent decoded positions, let
`v=p[t]-p[t-1]`, `a=dot(n,v)`, and `b=norm(v-a*n)`. Its feature is
`max(0,b-tan(half_angle_rad)*a,-a)`; a squared hinge penalizes values above the
annotated slack. A broad 60-degree cone allows different paths and step lengths,
with no absolute per-frame position targets. Reference turns/stops split the
direction intervals. The first predicted step uses the measured initial state,
so short chunks need no distant anchor. Zero motion is allowed: task arrival
still relies on interaction/endpoint constraints and the base loss. Direction
slack, cone width, and stationary hold margins are declared priors, not physical
success guarantees. Schema 1.0 remains readable for legacy data.

Schema 1.3 adds `window_direction_cone`. Its reference segment has a fixed axis,
an at-most-eight-step observation window, a minimum of two available steps, and
a reference net-motion threshold. The predicted feature uses the net displacement
over the same visible window instead of a first difference. Windows never cross
the reference segment or chunk boundary. When the reference has moved less than
5 mm, or fewer than two valid steps are available, a fixed segment bounding sphere
provides position supervision. It covers the entire reference segment plus a
20 mm prior, not a separate target around each reference frame. Reference turns
and stops split segments. All branch decisions use reference data and masks;
the prediction cannot disable the loss by pausing or keeping its gripper open.
Direction still allows pausing; it does not impose task progress by itself.
These values are declared motion priors, not verified physical tolerances.
RAX policy v3 emits this new type. Existing 1.0–1.2 annotations retain their old
semantics; no frozen dataset is rewritten automatically.

The current official RoboTwin EEF reader reads **absolute** world targets, not
deltas. Decoding reverses its exact 80D scatter (when used) and affine
normalization. Source quaternion order is stored in the episode action contract.
Native RoboTwin emits WXYZ but the current OpenWAM reader converts it as XYZW;
the loss decoder reverses that permutation in rotation-matrix space. The
checkpoint representation and deployment conversions are unchanged.

The reusable `action_trajectory.py` helper also distinguishes step increments
from chunk-start offsets, and world/body translation and rotation composition.
It is not silently enabled on the absolute RoboTwin reader. A different dataset
must supply its own verified inverse normalization and initial-state adapter.
Joint actions are rejected until an actual differentiable FK adapter exists.
No physical time rate is inferred from variable-duration native commands.

For flow matching, with `a_sigma=(1-sigma)*a+sigma*noise` and velocity
`v=noise-a`, constraints act on `a_hat=a_sigma-sigma*v_pred`. The callback runs
after model forward, in float32, retaining gradients to the action head. It
does not evaluate physical geometry directly on noise or velocity predictions.
Reference actions now stay FP32 from `prepare_inputs` onward, as do noise,
sigma, interpolation and FM targets. Only the model's forward input is cast to
its compute dtype. Reconstruction uses the retained FP32 noisy sample; casting
an already rounded BF16 sample back to FP32 is insufficient. Real BF16 model
outputs can still have quantization error.

## Calibration and issue #1 regression fixes

New objectives record version `meta_intervals_v4_fp32_window_calibration` and
explicit time/scale settings in provenance. Resume refuses mismatched settings;
use a separate output directory and initialization for comparisons.

- `timestep_strategy: clean` is the default and keeps the clean-action geometric
  objective's time weighting. `fm_matched` is an optional ablation multiplying
  each constraint by `FM_weight / max(sigma, sigma_floor)^2`; the denominator
  remains the sum of active annotation weights, not the sum of time factors.
  The default floor 0.05 is not tuned. This changes the objective and can greatly
  amplify some time ranges. It cancels sigma-squared curvature for a quadratic
  clean-coordinate error above the floor, not for arbitrary nonlinear, interval,
  or quartic penalties. It does not make constraint and action losses comparable.
- `penalty_scale_overrides: {kind: physical_scale}` changes outside-set curvature
  without changing allowed intervals or the embedded labels. Set values only
  after measuring loss and gradient behavior; no new tuned scales are claimed.
- Metrics include each type's weighted loss contribution and four sigma bins
  `[0,.25), [.25,.5), [.5,.75), [.75,1]`. Each collection of contributions sums
  to the globally reduced constraint loss when FM context is supplied. Type
  residual counts count elements; sigma-bin counts count active constraint
  instances, not independent samples. `raw_loss_contribution` omits the time
  factor but retains annotation weights, scales, and the global coefficient.

For an isolated, unsharded diagnostic forward, use the actual weighted terms:

```python
from openwam.train.constraint_calibration import measure_loss_gradients

result = architecture.compute_loss(
    **architecture.prepare_inputs(batch),
    action_loss_callback=objective.bind(batch),
    lambda_video=lambda_video, lambda_action=lambda_action,
    return_loss_terms=True,
)
report = measure_loss_gradients(result['loss_terms'], tuple(architecture.parameters()))
```

This retains the graph and reports norms, cosine similarities and norm ratios
on exactly the supplied parameters without accumulating `.grad`. Zero-gradient
cosines are undefined and reported as null. It does not aggregate DDP gradients
or reconstruct sharded/ZeRO parameters; label any parameter subset explicitly.
Run it outside an active distributed training job. The normal trainer does not
enable these extra backward passes. CPU regression tests establish arithmetic
and measurement behavior, not actual checkpoint gradient ratios or C2R gains.

The next causal comparison is A: FP32 + FM-only; B: FP32 + original direction
labels; C: FP32 + window-direction labels. Use identical initial weights, clean
recordings, training budgets, evaluation manifests and inference settings.
Keep the current historical run separate. Test time reweighting independently.
Success should be evaluated per task on matching seeds, alongside constraint
false acceptance on failed rollouts, not inferred from offline loss alone.

## Embedded data contract

The complete JSON payload contains `schema_version`, `action_contract`,
`provenance`, and a `constraints` list. Each base constraint contains:

```
id, type, subject,
reference_frame: {origin_m, quaternion_wxyz},
parameters: {bounds, ...axes required by the type},
activation: {mode, start, end, event?},
penalty: {kind, scale, weight},
evidence: {source, status, report_id, rationale}
```

The standard-library schema is `openwam/train/constraint_schema.py`; RAX ships
a byte-identical copy so its recording environment does not depend on OpenWAM
or Torch. `subject` is left_eef or right_eef. Reference transforms are fixed
within their annotated intervals. Units are metres/radians/open fraction;
axes and quaternions must be unit length. Bounds must have a justified source,
not be inferred from one successful action trajectory as guaranteed geometry.
Known geometry can be annotated offline; it is not queried during training.

RAX's `geometric-constraints` skill supplies a full JSON template and upsert
procedure. Accepted annotations enter the loss. Proposed, unknown and rejected
annotations remain in the data but contribute no loss. Observation-only EEF
constraints do not establish grasp stability, contact force or object support.

The recorder stores native EEF observations, joint states, camera JPEGs,
instruction, annotations and revision history in the same file. The current
clean50 collection uses the native expert's plan/replay and
`merge_pkl_to_hdf5_video` pipeline, retaining `save_freq` in provenance.
Earlier interactive RAX recordings used native command boundaries; inspect
`recording_method` and sampling metadata before mixing the two sources.
Successful expert recordings enter `raw/` and are published to `data/` after
constraint annotation and validation. Rejected seeds are logged separately;
they never count towards the fifty successful, distinct scenes per task.

## Training and validation

The feature is disabled by default. Once actual clean episodes contain accepted
constraints, add these options to the existing training command:

```bash
dataloader=robotwin dataloader.variant=clean_50 dataloader.action_mode=eef \
training.constraints.weight=0.1
```

The numerical weight is an initial experiment setting, not a tuned result.
Use the same checkpoint, clean data, seeds and schedule for an FM-only control.
The objective archives embedded annotations and inverse-map hashes with the
run; resume rejects changes to them, their weights or the execution horizon.
Independent RoboTwin evaluation is still required to measure task success.

Before a paired run, freeze the selected episodes, annotation revisions and
normalization statistics for both arms of the comparison. Use an independent
snapshot so later changes to the collection directory cannot alter the run.
The generic configuration has placeholder data/model paths and a zero constraint
weight, so it is not a fully configured experiment launch command.

Validation on 2026-10-22: 117 OpenWAM tests pass in the H200 training Python
environment on CPU. Model-level unit tests use small/fake components. A separate
audit of 2054 published episodes (409656 observation frames) passed full motion
coverage, native/reference decoding consistency, distinct scenes, all camera
JPEG decoding, and chunk observability checks. Actual reader tests used fresh
min-max statistics and the 80D mapping: 576 chunks across 45 available tasks
had exactly zero reference loss; all 45 tasks produced finite nonzero gradients
under a physical position perturbation. This is a partial-collection snapshot;
the unattended gate repeats the checks on the final 2500-episode snapshot.
A real eight-H200 smoke test with the official Foundation checkpoint passed:
microbatch 16 and accumulation 2 give global batch 256; one optimizer update
was saved, a fresh process restored its full state, and a second update completed
with finite losses/gradients and changed action-model weights. The recorded
microstep/optimizer-step pairs were (2, 1) and (4, 2). This used the audited
partial snapshot and a constant learning rate to exercise parameter updates;
all 438 frozen T5/VAE tensors were exactly equal before and after restart.
The formal run retains warmup/cosine. It is a chain validation, not evidence of
convergence or task success. Full five-epoch training and post-training C2R
evaluation remain to be completed.

Final-collection validation on 2026-10-24 passed on all 2500 episodes / 552498
frames. The actual reader checked 650 chunks across all 50 tasks: reference
loss was exactly zero, and every task had finite nonzero perturbation gradients.
A four-H200 run with microbatch 16 and accumulation 4 passed actual optimizer,
save, fresh-process restore and next-update checks. Its microstep/optimizer-step
pairs were (4, 1) and (8, 2), with finite losses and gradients and changed action
weights. Observed peak GPU memory during this test was about 71 GiB per device.
The gate then started a fresh five-epoch run in
`experiments/c2r_meta_e5_bs256_4gpu`. The earlier eight-GPU run stopped at
microstep 620 / optimizer step 310 before its first periodic checkpoint; no
saved formal weights were available to resume. Its logs remain intact. The
four-GPU run starts from the original Foundation weights and saves complete
state every 200 microsteps, retaining the latest two checkpoints. Training
completion and independent C2R success measurements are still pending.

`scripts/robotwin_training_gate.py --plan <plan.json>` provides the unattended
workflow; start from `configs/robotwin_clean50_gate.example.json`. It waits for
all 50 tasks × 50 published episodes and completed annotations, requires an
explicit compatible checkpoint, copies an independent dataset snapshot, audits
all frames/JPEGs/labels and distinct initial scenes, and builds fresh clean-only
normalization statistics. It then checks the actual normalized/unified reader,
waits for free GPUs, and runs an optimizer/save/restart/resume smoke test using
the GPU indices in the plan.
Only after those checks pass does it start a fresh 5-epoch run with effective
global batch 256 (8 × microbatch 16 × accumulation 2 in the example; the epoch tail
can be a smaller batch). A four-GPU plan uses `gpus: [0, 1, 2, 3]`, microbatch
16 and accumulation 4 for the same global batch. Launch process counts and
five-epoch completion checks follow the selected world size. Optional plan
`save_steps` controls checkpoint frequency in microsteps (default 2000); the
four-GPU restart uses 200 to reduce work lost to interruptions. Smoke updates
are discarded. A failure stops the gate
with `state.json` and logs; it never retries a failed full training run implicitly.
Checkpoint saves now wait for completed accumulation, and legacy checkpoints
saved mid-accumulation are rejected on resume. Fixed reference anchors are a
training observability repair, not additional physical-tolerance calibration.

For the official `dual_system / joint_self_attn` foundation initialization, run
`scripts/prepare_robotwin_initial_checkpoint.py --plan <plan.json> --model-root
<server-model-directory>` on the training host. It resolves the current `main`
of `OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model` once, downloads that coherent
revision, verifies file sizes and LFS SHA256 hashes, checks the 80D architecture
and safetensors header, then atomically fills `initial_checkpoint` in the plan.
The gate keeps waiting while the download is incomplete. Download status and
the exact resolved revision are recorded in `checkpoint_download.json`.

Tests need no model weights or simulation:

```bash
python -m pytest tests/test_constraint_supervision.py \
  tests/test_embedded_constraint_dataset.py tests/test_robotwin_dataloader.py \
  tests/test_flow_match_loss.py tests/test_openwam_trainer.py -q
```

The cross-repository test expects the RAX fork beside OpenWAM. Run these tests
on the existing H200 environment; local development needs only light checks.

## Fixed EEF-local points (schema 1.5)

The four position/distance types may optionally use a fixed local marker
`x = p + R r` in place of the EEF origin `p`: `axis_position_interval`,
`radial_distance_interval`, `point_distance_interval`, and
`axis_displacement_interval`. Put `r` in
`parameters.local_point={frame:"subject_eef",offset_m:[x,y,z],provenance:{...}}`.
The required provenance contains `source` (`known_geometry` or `manual`), a
nonempty `reference` and `rationale`, and a 64-character lowercase `sha256` of
the geometry source. The frame is the decoded physical EEF frame; metres are
explicit. These fields are immutable dataset geometry, not trainable offsets.
No other primitive accepts this parameter, and it requires embedded schema 1.5.

The existing interval loss applies to the transformed point, so gradients reach
both position and the physical rotation, after inverse normalization, unified
index selection and the legacy RoboTwin quaternion correction. No simulator or
object pose is required. For a nonzero offset, the point and any predicted-start
anchor need valid xyz and all six orientation dimensions. Invalid dimensions
mask the feature; valid nonfinite/degenerate rotations raise instead of falling
back to identity. An omitted or explicit zero offset preserves the old geometry
and position-only mask, including when orientation is unavailable.

The original displacement anchor may be row zero's measured state. It is
detached by the objective and is not a generated-action loss sample. Anchors
outside the chunk remain unavailable unless `anchor=reference_frame` is explicit;
there is no added requirement for both endpoints to be generated. RAX compiles
selection-based origins and long-span fixed anchors from the reference **marker**
using the same transform. Existing reference event and phase activation remains
fixed, and source schemas 1.0–1.4 retain their old meanings.

A rigid kinematic marker is not automatically a fingertip contact point or a
proof of object interaction. This implementation introduces no task-specific
offset/tolerance, activates no templates and rewrites no existing annotations.
The objective provenance version is `meta_intervals_v5_rigid_local_point`, so a
resume can distinguish it from the prior evaluator even for unchanged data.
