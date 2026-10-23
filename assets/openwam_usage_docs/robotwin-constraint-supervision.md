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
normalization statistics for both arms of the comparison. The live collection
directory is still growing; it is not itself a frozen experimental split.
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
Actual-checkpoint optimizer steps, checkpoint save/resume, 8-GPU training and
post-training C2R success evaluation remain to be validated.

`scripts/robotwin_training_gate.py --plan <plan.json>` provides the unattended
workflow; start from `configs/robotwin_clean50_gate.example.json`. It waits for
all 50 tasks × 50 published episodes and completed annotations, requires an
explicit compatible checkpoint, copies an independent dataset snapshot, audits
all frames/JPEGs/labels and distinct initial scenes, and builds fresh clean-only
normalization statistics. It then checks the actual normalized/unified reader,
waits for free GPUs, and runs an 8-GPU optimizer/save/restart/resume smoke test.
Only after those checks pass does it start a fresh 5-epoch run with effective
global batch 256 (8 × microbatch 2 × accumulation 16 by default; the epoch tail
can be a smaller batch). Smoke updates are discarded. A failure stops the gate
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
