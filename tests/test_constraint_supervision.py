"""Geometry, gradients, annotation alignment and provenance; no model assets."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.train.constraint_supervision import (
    ActionDecoder,
    Constraint,
    ConstraintStore,
    RoboTwinConstraintObjective,
    build_constraint_objective,
    constraint_residual,
    reduce_constraint_metrics,
    rotation_6d_to_matrix,
    validate_constraint_resume,
)


def _row(kind="REACH", *, scope="interval", start=1, end=4):
    parameters = {
        "REACH": {"center_m": [0, 0, 0], "radius_m": 0.1, "scale_m": 0.1},
        "CLEARANCE": {"safe_height_m": 0.5, "scale_m": 0.1},
        "ORIENTATION": {"local_axis": [0, 0, 1], "target_axis": [0, 0, 1], "tolerance_rad": 0.1, "scale_rad": 0.1},
    }
    row = {
        "schema_version": "0.1",
        "episode_id": "task/aloha-agilex_clean_50/data/episode0.hdf5",
        "segment_id": "synthetic-test",
        "interval": {"start": start, "end": end},
        "constraint": {
            "type": kind,
            "version": "1",
            "subject": "robot_eef:left",
            "frame": "world",
            "scope": scope,
            "parameters": parameters[kind],
        },
        "evidence": {"source": "sim_gt", "verification_status": "accepted", "report_id": "test-only"},
        "provenance": {"split": "train", "variant": "clean_50"},
    }
    if kind == "CLEARANCE":
        row["constraint"]["subject"] = "object:synthetic-cube"
        row["constraint"]["attachment"] = {
            "arm": "left",
            "position_m": [0, 0, -0.1],
            "quaternion_wxyz": [1, 0, 0, 0],
            "verification_status": "accepted",
            "report_id": "synthetic-rigid-grasp",
            "vertices_m": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]],
        }
    return row


def _actions(n=3):
    actions = torch.zeros(n, 20)
    actions[:, [3, 7, 13, 17]] = 1  # Identity rotation, column convention.
    return actions


def _objective(tmp_path, rows=None, **kwargs):
    rows = rows if rows is not None else [_row()]
    path = tmp_path / "constraints.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    episode = tmp_path / _row()["episode_id"]
    reader = SimpleNamespace(
        action_mode="eef",
        normalize_mode=None,
        _normalizer=None,
        _unify_dst_index=None,
        variant="clean_50",
        split="train",
        _episode_files=[episode],
        _episode_lengths=[10],
    )
    objective = RoboTwinConstraintObjective(reader, path, tmp_path, weight=0.5, **kwargs)
    return objective, str(episode)


def _evaluate(objective, episode, prediction, *, start=0, mask=None):
    if mask is None:
        mask = torch.ones_like(prediction, dtype=torch.bool)
    sample = {"episode_path": episode, "start_frame": start, "action_mask": mask}
    return objective.bind([sample])(prediction.unsqueeze(0))


def test_reach_feasible_region_and_gradient():
    c = Constraint.parse(_row())
    actions = _actions()
    actions[:, 0] = torch.tensor([0, 0.05, 0.3])
    actions.requires_grad_()
    residual = constraint_residual(actions, c)
    torch.testing.assert_close(residual, torch.tensor([0.0, 0.0, 0.2]))
    residual.square().sum().backward()
    assert actions.grad[2, 0] > 0
    assert actions.grad[:2].count_nonzero() == 0
    improved = constraint_residual(actions.detach() - 0.1 * actions.grad, c)
    assert improved[2] < residual[2]


def test_clearance_uses_lowest_object_geometry_and_stops_above_threshold():
    c = Constraint.parse(_row("CLEARANCE"))
    actions = _actions()
    actions[:, 2] = torch.tensor([0.6, 0.8, 1.0])
    actions.requires_grad_()
    residual = constraint_residual(actions, c)
    # EEF is already above .5, but object bottom is .4 in the first frame.
    torch.testing.assert_close(residual, torch.tensor([0.1, 0.0, 0.0]))
    residual.square().sum().backward()
    assert actions.grad[0, 2] < 0
    assert actions.grad[1:].count_nonzero() == 0
    assert torch.isfinite(actions.grad).all()


def test_orientation_yaw_free_pitch_penalized_and_differentiable():
    c = Constraint.parse(_row("ORIENTATION"))
    actions = _actions(2)
    # +90 degree yaw: columns (0,1,0),(-1,0,0); local z stays upright.
    actions[0, 3:9] = torch.tensor([0, 1, 0, -1, 0, 0])
    theta = 0.5
    actions[1, 3:9] = torch.tensor([math.cos(theta), 0, -math.sin(theta), 0, 1, 0])
    actions.requires_grad_()
    residual = constraint_residual(actions, c)
    torch.testing.assert_close(residual, torch.tensor([0.0, 0.4]))
    residual.square().sum().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad[1, 3:9].abs().sum() > 0


def test_column_rotation_convention_and_degenerate_rejection():
    matrix = rotation_6d_to_matrix(torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]))
    torch.testing.assert_close(matrix @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="degenerate"):
        rotation_6d_to_matrix(torch.zeros(6))


def test_unify_affine_inverse_preserves_right_arm_gradient():
    indices = np.array(list(range(10)) + list(range(34, 44)))
    reader = SimpleNamespace(
        action_mode="eef",
        normalize_mode="min-max",
        _unify_dst_index=indices,
        _normalizer=SimpleNamespace(_scale=np.full(20, 2.0), _offset=np.full(20, 0.25)),
    )
    prediction = torch.zeros(1, 80, requires_grad=True)
    decoded = ActionDecoder.from_reader(reader).decode(prediction)
    torch.testing.assert_close(decoded, torch.full((1, 20), 0.25))
    row = _row()
    row["constraint"]["subject"] = "robot_eef:right"
    constraint_residual(decoded, Constraint.parse(row)).sum().backward()
    assert prediction.grad[0, 34:37].abs().sum() > 0
    assert prediction.grad[0, :34].count_nonzero() == 0
    assert prediction.grad[0, 37:].count_nonzero() == 0


def test_endpoint_alignment_does_not_clip_to_window_end(tmp_path):
    objective, episode = _objective(tmp_path, [_row(scope="endpoint", start=1, end=5)])
    prediction = _actions().requires_grad_()
    early = _evaluate(objective, episode, prediction)  # Frames 1,2,3: endpoint=4 absent.
    assert early["metrics"]["constraint_count"] == 0
    early["loss"].backward()
    assert prediction.grad.count_nonzero() == 0
    prediction = _actions()
    prediction[:, 0] = 0.3
    prediction.requires_grad_()
    result = _evaluate(objective, episode, prediction, start=1)  # Frames 2,3,4.
    assert result["metrics"]["reach_count"] == 1
    torch.testing.assert_close(result["loss"], torch.tensor(2.0))
    result["loss"].backward()
    assert prediction.grad[:2].count_nonzero() == 0
    assert prediction.grad[2, 0] > 0


def test_padding_and_unknown_dimensions_do_not_contribute(tmp_path):
    objective, episode = _objective(tmp_path)
    prediction = _actions()
    prediction[:, 0] = 0.3
    prediction[1:] = float("nan")
    prediction.requires_grad_()
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[1:, :3] = False
    result = _evaluate(objective, episode, prediction, mask=mask)
    assert result["metrics"]["reach_count"] == 1
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1:].count_nonzero() == 0


def test_empty_nan_window_has_connected_finite_zero(tmp_path):
    objective, episode = _objective(tmp_path)
    prediction = torch.full((3, 20), float("nan"), requires_grad=True)
    result = _evaluate(objective, episode, prediction, mask=torch.zeros_like(prediction, dtype=torch.bool))
    assert result["loss"] == 0
    result["loss"].backward()
    assert prediction.grad.count_nonzero() == 0


def test_eef_orientation_does_not_require_observed_translation(tmp_path):
    objective, episode = _objective(tmp_path, [_row("ORIENTATION")])
    prediction = _actions()
    prediction[:, :3] = float("nan")
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, :3] = False
    result = _evaluate(objective, episode, prediction, mask=mask)
    assert result["metrics"]["orientation_count"] == 3
    assert result["loss"] == 0


@pytest.mark.parametrize("status", ["proposed", "rejected", "unknown"])
def test_unverified_constraints_excluded(tmp_path, status):
    accepted, unknown = _row(), _row()
    unknown["segment_id"] = "excluded"
    unknown["evidence"]["verification_status"] = status
    objective, _ = _objective(tmp_path, [accepted, unknown])
    assert objective.matched_constraints == 1
    assert objective.store.status_counts[status] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["provenance"].update(split="val"),
        lambda r: r["provenance"].update(variant="randomized_500"),
        lambda r: r["constraint"].update(frame="camera"),
        lambda r: r["constraint"]["parameters"].update(scale_m=0),
        lambda r: r["constraint"]["parameters"].update(radius_m=float("nan")),
        lambda r: r["evidence"].update(report_id=True),
        lambda r: r.update(episode_id="../episode0.hdf5"),
        lambda r: r["interval"].update(end=True),
    ],
)
def test_schema_rejects_ambiguous_or_invalid_supervision(mutation):
    row = _row()
    mutation(row)
    with pytest.raises(ValueError):
        Constraint.parse(row)


def test_clearance_cannot_use_eef_height():
    row = _row("CLEARANCE")
    row["constraint"].pop("attachment")
    row["constraint"]["subject"] = "robot_eef:left"
    with pytest.raises(ValueError, match="object geometry"):
        Constraint.parse(row)


def test_manifest_rejects_duplicate_no_accepted_or_no_matching_episode(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _objective(tmp_path, [_row(), _row()])
    row = _row()
    row["evidence"]["verification_status"] = "unknown"
    with pytest.raises(ValueError, match="no accepted"):
        _objective(tmp_path, [row])
    row = _row()
    row["episode_id"] = "another/data/episode0.hdf5"
    with pytest.raises(ValueError, match="match"):
        _objective(tmp_path, [row])


def test_demo_proxy_needs_explicit_opt_in(tmp_path):
    row = _row()
    row["evidence"]["source"] = "demonstration_proxy"
    with pytest.raises(ValueError, match="explicit"):
        _objective(tmp_path, [row])
    assert _objective(tmp_path, [row], allow_demonstration_proxy=True)[0].matched_constraints == 1


@pytest.mark.parametrize("start,end", [(0, 1), (5, 11)])
def test_intervals_must_reach_valid_action_frames(tmp_path, start, end):
    with pytest.raises(ValueError, match="future action frame|exceeds"):
        _objective(tmp_path, [_row(start=start, end=end)])


def test_resume_cannot_change_manifest_weight_or_disable_objective(tmp_path):
    objective, _ = _objective(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    objective.save_provenance(output)
    validate_constraint_resume(output, objective)
    with pytest.raises(ValueError, match="disable"):
        validate_constraint_resume(output, None)
    objective.weight = 0.2
    with pytest.raises(ValueError, match="changed"):
        validate_constraint_resume(output, objective)
    objective.weight = 0.5
    row = _row()
    row["constraint"]["parameters"]["radius_m"] = 0.2
    path = tmp_path / "edited.jsonl"
    path.write_text(json.dumps(row) + "\n")
    objective.store = ConstraintStore(path)
    with pytest.raises(ValueError, match="changed"):
        validate_constraint_resume(output, objective)


def test_zero_weight_does_not_open_manifest_or_require_robotwin():
    cfg = SimpleNamespace(training={"constraints": {"weight": 0, "manifest": "/nonexistent"}})
    assert build_constraint_objective(cfg, None) is None


def test_pooled_metrics_omit_unknown_types_and_use_global_max(tmp_path):
    objective, episode = _objective(tmp_path)
    prediction = _actions()
    prediction[:, 0] = torch.tensor([0.3, 0.1, 0.1])
    result = _evaluate(objective, episode, prediction)
    local = {"constraint": result["loss"], "constraint_metrics": result["metrics"]}
    other = copy.deepcopy(result["metrics"])
    other.update(
        reach_count=torch.tensor(1.0),
        reach_residual_sum=torch.tensor(0.6),
        reach_residual_max=torch.tensor(0.6),
        reach_violation_count=torch.tensor(1.0),
    )

    class TwoRanks:
        num_processes = 2

        def gather(self, value):
            if value.ndim == 1:
                return torch.cat([value, value])
            return torch.cat([value, torch.stack([other[key] for key in sorted(other)])[None]])

    metrics = reduce_constraint_metrics(TwoRanks(), local)["constraint_metrics"]
    assert metrics["reach_residual_mean"] == pytest.approx(0.2)
    assert metrics["reach_residual_max"] == pytest.approx(0.6)
    assert metrics["reach_violation_rate"] == pytest.approx(0.5)
    assert metrics["clearance_count"] == 0
    assert "clearance_residual_mean" not in metrics
