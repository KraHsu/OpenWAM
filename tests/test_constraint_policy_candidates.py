import numpy as np
import pytest

from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d
from scripts.robotwin_constraint_policy_candidates import native_action, rows_to_trajectory


def test_checkpoint_legacy_rotation_roundtrip_preserves_native_wxyz():
    left = np.array([.12, -.24, .8, .5, .5, -.5, .5])
    right = np.array([-.1, .3, 1., 1., 0., 0., 0.])
    encoded = robotwin_endpose_to_eef20d(left, right, .5, 1.)
    row, clipped = native_action(encoded)
    np.testing.assert_allclose(row[:3], left[:3], atol=1e-7)
    np.testing.assert_allclose(row[8:11], right[:3], atol=1e-7)
    assert abs(row[3:7] @ left[3:]) == pytest.approx(1.)
    assert abs(row[11:15] @ right[3:]) == pytest.approx(1.)
    assert row[7] == .5 and row[15] == 1. and not clipped


def test_actuator_clipping_is_explicit_and_never_moves_positions():
    pose = np.array([.1, .2, .3, 1., 0., 0., 0.])
    row, clipped = native_action(robotwin_endpose_to_eef20d(pose, pose, -0.1, 1.1))
    assert clipped and row[7] == 0. and row[15] == 1.
    np.testing.assert_allclose(row[:3], pose[:3], atol=1e-7)
    trace = rows_to_trajectory([row, row])
    assert trace['left']['pose'][0][3:] == row[3:7].tolist()
    assert len(trace['left']['gripper']) == 2


def test_invalid_model_output_is_not_repaired():
    with pytest.raises(ValueError, match='finite physical'):
        native_action(np.full(20, np.nan))
    with pytest.raises(ValueError, match='Degenerate'):
        native_action(np.zeros(20))
