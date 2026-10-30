"""Loss contributions, time context and physical-scale overrides are auditable."""
import copy

import pytest
import torch

from openwam.train.constraint_supervision import reduce_constraint_metrics
from tests.test_constraint_supervision import actions, constraint, objective


def prediction():
    x = actions(4)
    x[-1, 2] = 0.24
    x[-1, 12] = 0.34
    return x[None].requires_grad_()


def test_contributions_sum_to_loss_and_sigma_endpoints_are_counted(tmp_path):
    c = constraint()
    other = constraint('gripper_interval')
    other.update(id='grip', subject='right_eef')
    other['penalty']['weight'] = 3
    obj, sample = objective(tmp_path, [c, other])
    pred = prediction()
    pred = pred.clone()
    pred[0, -1, 19] = 0.4
    callback = obj.bind([sample, sample])
    assert callback.accepts_fm_context
    out = callback(pred.repeat(2, 1, 1), fm_sigma=torch.tensor([0., 1.]),
                   fm_timestep_weight=torch.tensor([0.3, 0.8]))
    reduced = reduce_constraint_metrics(None, dict(constraint=out['loss'], constraint_metrics=out['metrics']))
    metrics = reduced['constraint_metrics']
    parts = [v for k, v in metrics.items() if k.endswith('_loss_contribution')
             and not k.endswith('_raw_loss_contribution') and not k.startswith('sigma_bin_')]
    assert sum(parts) == pytest.approx(float(out['loss']))
    assert sum(metrics[f'sigma_bin_{i}_loss_contribution'] for i in range(4)) == pytest.approx(float(out['loss']))
    assert metrics['sigma_bin_0_count'] == 2
    assert metrics['sigma_bin_3_count'] == 2


def test_override_changes_curvature_without_changing_allowed_set_or_labels(tmp_path):
    cs = [constraint()]
    original = copy.deepcopy(cs)
    baseline, sample = objective(tmp_path, cs)
    scaled, _ = objective(tmp_path, cs, penalty_scale_overrides={'axis_position_interval': 0.2})
    pred = prediction()
    old = baseline.bind([sample])(pred)['loss']
    new = scaled.bind([sample])(pred)['loss']
    assert float(new) == pytest.approx(float(old)/4)
    assert cs == original
    inside = actions(4)[None]
    assert baseline.bind([sample])(inside)['loss'] == 0
    assert scaled.bind([sample])(inside)['loss'] == 0
    assert baseline.provenance() != scaled.provenance()


def test_matched_time_weight_is_explicit_and_requires_context(tmp_path):
    obj, sample = objective(tmp_path, [constraint()], timestep_strategy='fm_matched')
    with pytest.raises(ValueError, match='context'):
        obj.bind([sample])(prediction())
    base, _ = objective(tmp_path, [constraint()])
    pred = prediction()
    plain = base.bind([sample])(pred)['loss']
    matched = obj.bind([sample])(pred, fm_sigma=torch.tensor([0.5]),
                                fm_timestep_weight=torch.tensor([0.2]))['loss']
    assert float(matched) == pytest.approx(float(plain)*0.2/0.5**2)
    matched.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_global_contributions_use_global_denominator_not_rank_means(tmp_path, monkeypatch):
    obj, sample = objective(tmp_path, [constraint()])
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda denominator: denominator.add_(3))
    out = obj.bind([sample])(prediction(), fm_sigma=torch.tensor([0.5]),
                             fm_timestep_weight=torch.tensor([0.2]))
    # Local N=4,D=1; remote N=27,D=3; coefficient=.5. The global
    # loss is .5*(4+27)/4=3.875, not the rank-mean .5*(4+9)/2=3.25.
    assert float(out['loss']) == pytest.approx(1.0)  # world_size * local N / global D

    class Distributed:
        num_processes = 2

        def gather(self, values):
            if values.shape[-1] == 1:
                return torch.cat((values, values.new_tensor([6.75])))
            remote = values.clone()
            names = sorted(out['metrics'])
            for i, name in enumerate(names):
                if name.endswith('_weight_sum') or name.endswith('_count'):
                    remote[:, i] *= 3
                elif name.endswith('_penalty_sum'):
                    remote[:, i] *= 27/4
            return torch.cat((values, remote))

    reduced = reduce_constraint_metrics(Distributed(), dict(constraint=out['loss'], constraint_metrics=out['metrics']))
    assert reduced['loss_constraint'] == pytest.approx(3.875)
    assert reduced['constraint_metrics']['axis_position_interval_loss_contribution'] == pytest.approx(3.875)
    assert reduced['constraint_metrics']['sigma_bin_2_loss_contribution'] == pytest.approx(3.875)


@pytest.mark.parametrize('kwargs', [dict(timestep_strategy='invalid'), dict(sigma_floor=0),
                                  dict(sigma_floor=1.1), dict(penalty_scale_overrides={'wrong': 1}),
                                  dict(penalty_scale_overrides={'axis_position_interval': -1})])
def test_invalid_calibration_rejected_before_training(tmp_path, kwargs):
    with pytest.raises(ValueError):
        objective(tmp_path, [constraint()], **kwargs)
