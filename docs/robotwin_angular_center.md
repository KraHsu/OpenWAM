# Fixed reference hints for an exact angular center

An `axis_angle_interval` with positive lower bound has a nondifferentiable
center. For `a = R local_axis`, target `n`, and
`phi = atan2(norm(a × n), a · n)`, the lower penalty is locally radial:
`((lower - norm(x))/scale)^2`. At `x=0` there is no ordinary derivative.
PyTorch's zero selection for the norm can therefore give positive loss and
zero gradient. This extension selects a nonzero limiting generalized gradient
at that exact center. It does not change the allowed set or claim a convex
subgradient, convergence, a decrease of the total training loss, or task success.

The default training mode remains `legacy`. Schema 1.0–1.5 remains readable.
Schema 1.6 adds only `parameters.center_tangent_hint` to `axis_angle_interval`.
The template opt-in is a rule field:

```json
{"angle_center_hint": "selection_start_reference_projection_v1"}
```

The rule keeps its existing selector, bounds, scale, weight and reference frame.
The compiler takes the physical reference axis at the selected start (the event
frame for a reference event), expresses it in the constraint's reference frame,
and projects it onto the plane perpendicular to `reference_axis`. The normalized
projection is the fixed direction. It is never recomputed from a predicted
chunk or replaced by a guessed world axis. A projection norm at most `1e-8`
rejects annotation; that threshold is source validity, not a near-center region
where training derivatives are changed.

The emitted object contains `version=1`, `frame=reference`, a unit `direction`,
and `source={kind,recording_id,frame,quaternion_wxyz,sha256}`. The source stores
the measured physical WXYZ quaternion with canonical sign (retaining its measured
roundoff), and a hash of that source, subject, local/target axes and reference
frame. Portable schema validation recomputes the projection and hash. Full HDF5
reading independently verifies the stored source orientation against its actual
episode row, interpreting the explicitly declared WXYZ or XYZW order. Recording
completion and annotation publication repeat that check. An internally valid,
rehashed hint for a different source pose is rejected. As with existing reader
caches, the training dataset must remain immutable after initialization.

The source can be outside a 32-action training chunk. Its fixed hint travels
with the annotation; no generated pose is substituted for the missing source.
All old activation and validity rules remain: row zero is measured/detached,
all six orientation dimensions must be valid, padding is excluded before
geometry, and reference-event gripper masks depend on the data. Rotation
decoding still restores the physical orientation for both the physical-6D and
native WXYZ-read-as-XYZW conventions, after inverse normalization/unified indexing.
The NumPy scorer and viewer retain the same feature values and allowed regions;
the hint is a backward policy, not another geometric acceptance constraint.

Explicit training configuration:

```yaml
training:
  constraints:
    angle_center_backward: fixed_annotation_v1
    angle_center_log_per_constraint: true
```

This mode requires a valid hint on every accepted positive-lower axis-angle
constraint before training. Missing or invalid hints fail closed. The runtime
selection requires **every cross-product component exactly zero**, positive
dot product, and a positive lower bound. Nonzero values, including values whose
norm underflows, use the original derivative; antiparallel/pi cases and zero
lower bounds retain legacy behavior. The selected cross-norm gradient is the
unit `hint_direction_world × target`. A zero-valued forward correction adds this
first-order selection while retaining the original `vector_norm`/`atan2` graph
everywhere else. There is no epsilon neighborhood or changed forward value.
Higher-order derivatives at the singularity are not part of this contract.

The objective/checkpoint-root provenance freezes the policy, code fingerprints,
hint bundles and diagnostic catalog; resume rejects any difference. Legacy
provenance is unchanged. Metrics include exact-center hits, active hint frames,
and invalid-hint count (zero for admitted batches; invalid data aborts instead
of being counted and silently ignored). With per-constraint logging enabled,
dataset-wide episode/constraint hashes map to contribution metrics through the
stored catalog. Every rank uses the same keys, including inactive zeros; the
global constraint-weight denominator is unchanged. This can add many metric
columns on a large dataset. Turning that diagnostic off is explicit and bound
to resume provenance; aggregate counts and contributions remain available.

CPU acceptance must check unchanged forward/nonzero derivatives, one-sided
descent rather than a central derivative at the cusp, frame covariance, both
6D conventions, float32/autocast, masks/events/row zero, long chunks, source
tampering and real HDF5 reader roundtrips. An intrinsic reference-directed step
can preserve a paired X-axis constraint in a particular geometry, but this does
not guarantee joint feasibility after an arbitrary optimizer update or against
other losses. A separately authorized training smoke must measure actual hit
frequency and optimization behavior. This implementation neither activates a
template nor rewrites existing data or running experiments.
