"""Base geometry, data-only activation, physical decoding and embedded labels."""
import copy
import json
import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from openwam.train.action_trajectory import EEFActionSpec, physical_rotation, restore_eef_trajectory
from openwam.train.constraint_schema import HDF5_KEY, canonical_json, read_embedded, validate_bundle, validate_constraint
from openwam.train.constraint_supervision import ActionDecoder, RoboTwinConstraintObjective, build_constraint_objective, validate_constraint_resume
from openwam.train.meta_constraints import penalty, residual


def actions(n=5):
    x = torch.zeros(n, 20)
    x[:, [3, 7, 13, 17]] = 1
    return x


def constraint(kind='axis_position_interval', *, start=0, end=5, mode='endpoint'):
    parameters = {'bounds': [-0.03, 0.04]}
    if kind in ('axis_position_interval', 'radial_distance_interval', 'axis_displacement_interval'):
        parameters['axis'] = [0, 0, 1]
    if kind in ('point_distance_interval', 'radial_distance_interval', 'angular_displacement_interval'):
        parameters['bounds'] = [0, 0.1]
    if kind == 'gripper_interval':
        parameters['bounds'] = [0, 0.2]
    if kind == 'axis_angle_interval':
        parameters.update(local_axis=[0, 0, 1], reference_axis=[0, 0, 1], bounds=[0, 0.1])
    return {'id': 'c0', 'type': kind, 'subject': 'left_eef',
            'reference_frame': {'origin_m': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]},
            'parameters': parameters, 'activation': {'start': start, 'end': end, 'mode': mode},
            'penalty': {'kind': 'squared_hinge', 'scale': 0.1, 'weight': 1},
            'evidence': {'source': 'manual', 'status': 'accepted', 'report_id': 'test-only', 'rationale': 'synthetic'}}


def bundle(cs):
    return {'schema_version': '1.0', 'action_contract': {'position': 'absolute', 'rotation': 'absolute',
            'quaternion_order': 'xyzw', 'gripper': 'absolute_open_fraction', 'frame': 'world'},
            'provenance': {'source': 'test', 'split': 'train', 'variant': 'clean_50', 'recording_id': 'test'}, 'constraints': cs}


def evaluate(pred, c, ref=None, mask=None, start=0):
    validate_constraint(c)
    return residual(pred, actions(len(pred)) if ref is None else ref.detach(),
                    torch.ones_like(pred, dtype=torch.bool) if mask is None else mask,
                    c, start_frame=start, rotation_convention='physical')


def objective(tmp_path, cs, **kwargs):
    episode = str(tmp_path/'episode0.hdf5')
    b = bundle(cs)
    reader = SimpleNamespace(action_mode='eef', normalize_mode=None, _normalizer=None, _unify_dst_index=None,
                             variant='clean_50', split='train', _episode_files=[episode], _episode_lengths=[10],
                             _constraint_bundles={episode: b})
    obj = RoboTwinConstraintObjective(reader, tmp_path, weight=0.5, **kwargs)
    ref = actions()
    sample = {'episode_path': episode, 'start_frame': 0, 'proprio': ref[:1], 'action': ref[1:],
              'proprio_mask': torch.ones_like(ref[:1], dtype=torch.bool), 'action_mask': torch.ones_like(ref[1:], dtype=torch.bool),
              'trajectory_constraints': b}
    return obj, sample


def test_axial_interval_zero_inside_and_continuous_growing_penalty():
    x = actions(); x[:, 2] = torch.tensor([0, -0.04, -0.02, 0.03, 0.14]); x.requires_grad_()
    c = constraint(mode='interval')
    v = evaluate(x, c)
    torch.testing.assert_close(v, torch.tensor([0.01, 0, 0, 0.1]))
    penalty(v, c['penalty']).backward()
    assert x.grad[1, 2] < 0 and x.grad[-1, 2] > 0
    assert x.grad[2:4].count_nonzero() == 0
    large = torch.tensor([0.2], requires_grad=True)
    squared = penalty(large, c['penalty'])
    quartic = penalty(large, dict(c['penalty'], kind='quartic_hinge'))
    assert quartic > squared
    assert torch.autograd.grad(quartic, large)[0] > 0


def test_fixed_task_frame_rotation_and_translation():
    c = constraint(); c['parameters'].update(axis=[1, 0, 0], bounds=[0.9, 1.1])
    c['reference_frame'] = {'origin_m': [2, 3, 0], 'quaternion_wxyz': [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]}
    x = actions(); x[-1, :3] = torch.tensor([9, 4, 8])
    assert evaluate(x, c).item() == 0  # frame x is world y; other coordinates free


def test_cylinder_couples_horizontal_axes_without_restricting_axial_motion():
    radial = constraint('radial_distance_interval'); axial = constraint()
    axial['parameters']['bounds'] = [-1, 1]
    x = actions(); x[-1, :3] = torch.tensor([0.08, 0.08, 0.7])
    assert evaluate(x, axial).item() == 0
    assert evaluate(x, radial).item() > 0  # a box would incorrectly accept (.08,.08)
    x[-1, :2] = torch.tensor([0.03, 0.04]); assert evaluate(x, radial).item() == 0


def test_axis_angle_keeps_yaw_free_and_penalizes_tilt():
    c = constraint('axis_angle_interval')
    x = actions(); x[-1, 3:9] = torch.tensor([0., 1, 0, -1, 0, 0])
    assert evaluate(x, c).item() == 0
    x[-1, 3:9] = torch.tensor([math.cos(0.4), 0, -math.sin(0.4), 0, 1, 0]); x.requires_grad_()
    v = evaluate(x, c); assert v.item() == pytest.approx(0.3)
    penalty(v, c['penalty']).backward()
    assert torch.isfinite(x.grad).all() and x.grad[-1, 3:9].abs().sum() > 0


@pytest.mark.parametrize('kind', ['axis_displacement_interval', 'angular_displacement_interval'])
def test_displacements_require_visible_stage_origin(kind):
    c = constraint(kind, start=0, end=6)
    assert evaluate(actions(), c, start=1) is None


def test_axis_displacement_uses_two_sparse_endpoints():
    c = constraint('axis_displacement_interval'); c['parameters']['bounds'] = [0.2, 0.4]
    x = actions(); x[:, 2] = 1; x[-1, 2] = 1.1; x.requires_grad_()
    v = evaluate(x, c); v.square().sum().backward()
    assert x.grad[-1, 2] < 0 and x.grad[0, 2] > 0
    assert x.grad[1:-1].count_nonzero() == 0


def test_reference_event_activation_cannot_be_evaded_by_not_closing():
    c = constraint(mode='reference_event')
    c['activation']['event'] = {'frame': 3, 'direction': 'decreasing', 'threshold': 0.5}
    ref = actions(); ref[:, 9] = torch.tensor([1, 1, 1, 0, 0])
    pred = actions(); pred[:, 9] = 1; pred[:, 2] = 1; pred.requires_grad_()
    v = evaluate(pred, c, ref=ref)
    assert v.numel() == 1 and v.item() > 0
    v.square().sum().backward()
    assert pred.grad[3, 2] > 0 and pred.grad[:, 9].count_nonzero() == 0
    ref[:, 9] = 1
    assert evaluate(pred, c, ref=ref) is None


def test_endpoint_never_clips_to_chunk_and_padding_nan_never_touched(tmp_path):
    obj, sample = objective(tmp_path, [constraint(end=6)])
    pred = torch.full((1, 4, 20), float('nan'), requires_grad=True)
    result = obj.bind([sample])(pred)
    assert result['loss'] == 0
    result['loss'].backward(); assert pred.grad.count_nonzero() == 0


def test_masked_unrelated_rotation_nan_does_not_break_position_gradient(tmp_path):
    obj, sample = objective(tmp_path, [constraint()])
    pred = actions(4); pred[-1, 2] = 0.2; pred[:, 3:9] = float('nan'); pred.requires_grad_()
    sample['action_mask'][:, 3:9] = False
    result = obj.bind([sample])(pred[None]); result['loss'].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[-1, 2] > 0


def test_weighted_average_and_ddp_denominator(tmp_path, monkeypatch):
    c, other = constraint(), constraint()
    other.update(id='c1', subject='right_eef'); other['penalty']['weight'] = 3
    obj, sample = objective(tmp_path, [c, other])
    pred = actions(4); pred[-1, 2], pred[-1, 12] = 0.24, 0.34; pred.requires_grad_()
    local = obj.bind([sample])(pred[None])
    torch.testing.assert_close(local['loss'], torch.tensor(0.5*(4+3*9)/4))
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda x: x.add_(2.0))  # other rank active weight
    distributed = obj.bind([sample])(pred[None])
    torch.testing.assert_close(distributed['loss'], torch.tensor(0.5*(4+3*9)*2/6))


def test_execution_prefix_and_reference_detached(tmp_path):
    obj, sample = objective(tmp_path, [constraint()], execution_horizon=2)
    sample['action'].requires_grad_(); sample['proprio'].requires_grad_()
    pred = actions(4).requires_grad_()
    result = obj.bind([sample])(pred[None]); result['loss'].backward()
    assert result['metrics']['constraint_count'] == 0
    assert sample['action'].grad is None and sample['proprio'].grad is None


def test_embedded_hdf5_roundtrip_and_resume_freezes_annotations(tmp_path):
    b = bundle([constraint()])
    path = tmp_path/'data.hdf5'
    with h5py.File(path, 'w') as h:
        h.create_dataset('endpose/left_endpose', data=np.zeros((5, 7)))
        h.create_dataset(HDF5_KEY, data=canonical_json(b), dtype=h5py.string_dtype('utf-8'))
    with h5py.File(path) as h:
        assert read_embedded(h) == b
    obj, sample = objective(tmp_path, b['constraints'])
    obj.save_provenance(tmp_path/'run'); validate_constraint_resume(tmp_path/'run', obj)
    sample['trajectory_constraints']['constraints'][0]['penalty']['weight'] = 2
    with pytest.raises(ValueError, match='changed'):
        obj.bind([sample])
    obj.execution_horizon = 2
    with pytest.raises(ValueError, match='changed'):
        validate_constraint_resume(tmp_path/'run', obj)


@pytest.mark.parametrize('mutation', [
    lambda b: b['constraints'][0].update(type='grasp_success'),
    lambda b: b['constraints'][0]['penalty'].update(scale=0),
    lambda b: b['constraints'][0]['parameters'].update(axis=[0, 0, 2]),
    lambda b: b['constraints'][0]['activation'].update(end=50),
    lambda b: b['constraints'][0]['parameters'].update(bounds=[1, -1]),
    lambda b: b['constraints'][0].update(primitive='lift'),
    lambda b: b['provenance'].update(variant='randomized_500'),
    lambda b: b['action_contract'].update(position='step_delta'),
])
def test_invalid_annotations_rejected(mutation):
    b = bundle([constraint()]); mutation(b)
    with pytest.raises(ValueError):
        validate_bundle(b, 10)


def quat_matrix(q):
    from openwam.train.meta_constraints import frame_rotation
    return frame_rotation(q, torch.zeros(()))


def test_native_wxyz_permutation_restores_true_rotation_with_gradients():
    torch.manual_seed(7)
    for q in torch.randn(10, 4):
        q = q/q.norm()
        encoded = quat_matrix(q[[3, 0, 1, 2]])
        six = torch.cat((encoded[:, 0], encoded[:, 1])).requires_grad_()
        matrix = physical_rotation(six, 'robotwin_wxyz_read_as_xyzw')
        torch.testing.assert_close(matrix, quat_matrix(q), atol=1e-6, rtol=1e-6)
        (matrix*torch.arange(9).reshape(3, 3)).sum().backward()
        assert torch.isfinite(six.grad).all() and six.grad.abs().sum() > 0


def test_delta_decoding_accumulation_and_body_composition():
    initial = actions(1)[0]; initial[0] = 1
    delta = actions(3); delta[:, 0] = torch.tensor([0.1, 0.2, 0.3]); delta.requires_grad_()
    step = restore_eef_trajectory(delta, initial, EEFActionSpec(position='step_delta'))
    start = restore_eef_trajectory(delta, initial, EEFActionSpec(position='start_delta'))
    torch.testing.assert_close(step[:, 0], torch.tensor([1.1, 1.3, 1.6]))
    torch.testing.assert_close(start[:, 0], torch.tensor([1.1, 1.2, 1.3]))
    step[-1, 0].backward(); torch.testing.assert_close(delta.grad[:, 0], torch.ones(3))
    initial[3:9] = torch.tensor([0., 1, 0, -1, 0, 0])
    world = restore_eef_trajectory(delta.detach(), initial, EEFActionSpec(position='step_delta'))
    body = restore_eef_trajectory(delta.detach(), initial, EEFActionSpec(position='step_delta', translation_frame='body'))
    assert not torch.allclose(world[:, :3], body[:, :3])


def test_affine_and_unified_action_inverse():
    reader = SimpleNamespace(action_mode='eef', normalize_mode='min-max', _unify_dst_index=np.r_[0:10, 34:44],
                             _normalizer=SimpleNamespace(_scale=np.ones(20)*2, _offset=np.ones(20)*0.25))
    x = torch.zeros(2, 80, requires_grad=True)
    decoded = ActionDecoder.from_reader(reader).decode(x)
    torch.testing.assert_close(decoded, torch.full((2, 20), 0.25))
    decoded[:, 10].sum().backward()
    assert x.grad[:, 34].sum() > 0 and x.grad[:, :34].count_nonzero() == 0


def test_zero_weight_is_noop_and_joints_require_actual_fk_adapter():
    assert build_constraint_objective(SimpleNamespace(training={'constraints': {'weight': 0}}), None) is None
    with pytest.raises(ValueError, match='action_mode=eef'):
        ActionDecoder.from_reader(SimpleNamespace(action_mode='joint'))
