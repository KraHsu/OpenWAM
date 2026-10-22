import copy
import math

import pytest
import torch

from openwam.train.constraint_schema import validate_bundle, validate_constraint
from openwam.train.meta_constraints import penalty
from tests.test_constraint_supervision import actions, bundle, constraint, evaluate, objective


def cone(end=5):
    c = constraint('motion_direction_cone', start=1, end=end, mode='interval')
    c['parameters'] = {'axis':[1,0,0], 'half_angle_rad':math.radians(60), 'bounds':[0,.0001]}
    c['penalty']['scale'] = .005
    return c


def test_different_paths_speeds_and_translation_inside_cone_have_zero_loss():
    c=cone(); validate_constraint(c)
    for steps in ([[.01,0,0]]*4, [[.03,.02,0],[.01,-.01,0],[.02,.01,.01],[.05,0,-.02]]):
        x=actions(); x[1:,:3]=torch.tensor(steps).cumsum(0)
        for offset in ([0,0,0],[2,-3,7]):
            translated=x.clone();translated[:,:3]+=torch.tensor(offset)
            assert evaluate(translated,c).max()==0
    # Direction alone does not force arrival or speed.
    stationary=actions().requires_grad_(); penalty(evaluate(stationary,c),c['penalty']).backward()
    assert torch.isfinite(stationary.grad).all() and stationary.grad.count_nonzero()==0


def test_backward_and_sideways_steps_receive_correct_gradients_in_short_chunks():
    c=cone(end=200)
    for step, component in [([-0.02,0,0],0),([0,.02,0],1)]:
        x=actions(2);x[1,:3]=torch.tensor(step);x.requires_grad_()
        v=evaluate(x,c,start=100);assert v.item()>0
        penalty(v,c['penalty']).backward()
        assert torch.isfinite(x.grad).all()
        assert x.grad[1,component]*step[component]>0
        torch.testing.assert_close(x.grad[0,:3],-x.grad[1,:3])


def test_invalid_previous_position_is_excluded_not_read():
    c=cone();x=actions();x[2,:3]=float('nan');x.requires_grad_()
    valid=torch.ones_like(x,dtype=torch.bool);valid[2,:3]=False
    result=evaluate(x,c,mask=valid)
    assert result.numel()==2 and result.max()==0
    result.square().sum().backward();assert torch.isfinite(x.grad).all()


def test_full_coverage_gate_rejects_sparse_data_and_cones_are_schema_1_1(tmp_path):
    with pytest.raises(ValueError,match='Incomplete chunk-safe motion coverage'):
        objective(tmp_path,[constraint()],require_full_coverage=True)
    left=cone(end=10);right=copy.deepcopy(left);right.update(id='right',subject='right_eef')
    b=bundle([left,right])
    with pytest.raises(ValueError,match='schema 1.1'):validate_bundle(b,10)
    b['schema_version']='1.1';validate_bundle(b,10)
    for mutation in [lambda c:c['activation'].update(start=0), lambda c:c['parameters'].update(half_angle_rad=math.pi/2)]:
        bad=cone();mutation(bad)
        with pytest.raises(ValueError):validate_constraint(bad)
