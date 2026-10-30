"""Schema 1.3 net-motion supervision keeps chunk edges out of finite differences."""

import copy
import math

import pytest
import torch

from openwam.train.constraint_schema import validate_constraint, validate_bundle
from openwam.train.meta_constraints import residual, penalty


def fixture(length=65):
    reference = torch.zeros(length, 20)
    reference[:, [3, 7, 13, 17]] = 1
    reference[:, 0] = torch.arange(length) * 0.005
    c = dict(
        id="window",
        type="window_direction_cone",
        subject="left_eef",
        reference_frame=dict(origin_m=[0.16, 0, 0], quaternion_wxyz=[1, 0, 0, 0]),
        parameters=dict(
            axis=[1, 0, 0],
            half_angle_rad=math.pi / 3,
            bounds=[0, 0.0001],
            window_steps=8,
            minimum_window_steps=2,
            minimum_reference_progress_m=0.005,
            fallback_radius_m=0.18,
        ),
        activation=dict(mode="interval", start=1, end=length),
        penalty=dict(kind="squared_hinge", scale=0.005, weight=1),
        evidence=dict(
            source="demonstration_analysis",
            status="accepted",
            report_id="synthetic",
            rationale="Uncalibrated motion prior, no success claim.",
        ),
    )
    return reference, c


def loss(pred, ref, c, *, start=0, valid=None):
    validate_constraint(c)
    return residual(
        pred,
        ref,
        torch.ones_like(pred, dtype=torch.bool) if valid is None else valid,
        c,
        start_frame=start,
        rotation_convention="physical",
    )


@pytest.mark.parametrize("seed", range(5))
def test_net_motion_is_less_sensitive_to_independent_position_jitter(seed):
    ref, c = fixture()
    pred = ref.clone()
    pred[1:, :3] += torch.randn(64, 3, generator=torch.Generator().manual_seed(seed)) * 0.005
    old = copy.deepcopy(c)
    old["type"] = "motion_direction_cone"
    old["parameters"] = {k: v for k, v in c["parameters"].items() if k in ("axis", "half_angle_rad", "bounds")}
    assert penalty(loss(pred, ref, c), c["penalty"]) < 0.2 * penalty(loss(pred, ref, old), old["penalty"])


def test_reference_zero_loss_and_pauses_do_not_imply_required_progress():
    ref, c = fixture()
    assert loss(ref, ref, c).max() == 0
    pred = ref.clone()
    pred[:, :3] = 0
    assert loss(pred, ref, c).max() == 0


def test_chunk_first_action_uses_fixed_segment_region_not_single_step_direction():
    ref, c = fixture()
    ref = ref[30:32]
    pred = ref.clone()
    pred[1, 1] += 0.02
    assert loss(pred, ref, c, start=30).item() == 0
    pred[1, 1] = 0.3
    assert loss(pred, ref, c, start=30).item() > 0.1


def test_reference_low_motion_uses_region_even_when_predicted_motion_is_large():
    ref, c = fixture()
    ref[:, :3] = torch.tensor([0.16, 0, 0])
    pred = ref.clone()
    pred[1::2, 1] += 0.02
    assert loss(pred, ref, c).max() == 0


def test_turn_boundary_never_reads_previous_segment():
    ref, c = fixture()
    c["activation"]["start"] = 20
    c["parameters"]["axis"] = [0, 1, 0]
    ref[19:, :3] = 0
    ref[19:, 1] = torch.arange(len(ref) - 19) * 0.005
    ref[:19, :3] = float("nan")
    assert loss(ref, ref, c).max() == 0


def test_padding_and_missing_window_history_use_only_valid_positions():
    ref, c = fixture()
    pred = ref.clone()
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[5:8, :3] = False
    pred[5:8, :3] = float("nan")
    ref[5:8, :3] = float("nan")
    valid[60:] = False
    pred[60:] = float("nan")
    ref[60:] = float("nan")
    assert torch.isfinite(loss(pred, ref, c, valid=valid)).all()


def test_outside_direction_has_finite_nonzero_gradient():
    ref, c = fixture()
    pred = ref.clone()
    pred[:, 0] = -pred[:, 0]
    pred.requires_grad_()
    value = penalty(loss(pred, ref, c), c["penalty"])
    value.backward()
    assert value > 0 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_old_schema_keeps_old_kind_and_new_kind_requires_new_schema():
    _, c = fixture()
    b = dict(
        schema_version="1.2",
        action_contract=dict(
            position="absolute",
            rotation="absolute",
            quaternion_order="wxyz",
            gripper="absolute_open_fraction",
            frame="world",
        ),
        provenance=dict(split="train", variant="clean_50", recording_id="fixture", source="test"),
        constraints=[c],
    )
    with pytest.raises(ValueError, match="schema 1.3"):
        validate_bundle(b)
    b["schema_version"] = "1.3"
    validate_bundle(b)
    b["schema_version"] = "1.2"
    c["type"] = "motion_direction_cone"
    c["parameters"] = {k: v for k, v in c["parameters"].items() if k in ("axis", "half_angle_rad", "bounds")}
    validate_bundle(b)
