"""Joint physical grip masks, amax subgradients, and schema compatibility."""
import math
import pytest
import torch
from openwam.train.constraint_schema import validate_bundle
from openwam.train.meta_constraints import penalty
from tests.test_constraint_supervision import actions, constraint, bundle, evaluate
from tests.test_constraint_window import fixture as window_fixture


def joint():
    c=constraint('gripper_interval',mode='interval')
    c.update(type='gripper_pair_max_interval',subject='both_grippers')
    c['parameters']['bounds']=[.25,1.];c['penalty']['scale']=.02
    return c


@pytest.mark.parametrize('left,right,accepted', [(0,.5,True),(.5,0,True),(0,0,False)])
def test_recombined_two_arm_failure_requires_joint_feature(left,right,accepted):
    x=actions();x[:,9]=left;x[:,19]=right
    r=evaluate(x,joint());assert bool((r==0).all())==accepted


def test_equal_max_ties_backpropagate_to_both_independent_grip_dimensions():
    x=actions().requires_grad_();c=joint()
    penalty(evaluate(x,c),c['penalty']).backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[1:,[9,19]]<0).all()
    torch.testing.assert_close(x.grad[1:,9],x.grad[1:,19])
    assert x.grad[0].count_nonzero()==0
    assert x.grad[:,[i for i in range(20) if i not in (9,19)]].count_nonzero()==0


@pytest.mark.parametrize('dim',[9,19])
def test_either_invalid_grip_removes_whole_joint_feature_not_just_one_hand(dim):
    x=actions();valid=torch.ones_like(x,dtype=torch.bool);valid[1,dim]=False
    x[1,dim]=float('nan')
    assert evaluate(x,joint(),mask=valid).shape==(3,)
    valid[:,dim]=False
    assert evaluate(x,joint(),mask=valid) is None
    valid[2,dim]=True;x[2,dim]=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):evaluate(x,joint(),mask=valid)


def test_schema_13_window_and_fixed_anchor_survive_version_14_upgrade():
    _,w=window_fixture();b=bundle([w]);b['schema_version']='1.3';validate_bundle(b)
    fixed=constraint('axis_displacement_interval');fixed['parameters']['anchor']='reference_frame'
    for version in ['1.2','1.3','1.4']:
        b=bundle([fixed]);b['schema_version']=version;validate_bundle(b)
    b=bundle([joint()]);b['schema_version']='1.4';validate_bundle(b)
    b['schema_version']='1.3'
    with pytest.raises(ValueError,match='1.4'):validate_bundle(b)


def test_positive_lower_angle_has_known_exact_alignment_zero_gradient():
    # Intentionally documents an unresolved annulus-center cusp, not a fix.
    c=constraint('axis_angle_interval',mode='interval')
    c['parameters'].update(local_axis=[0,1,0],reference_axis=[0,1,0],bounds=[math.radians(25),math.radians(65)])
    x=actions().double().requires_grad_();loss=penalty(evaluate(x,c),c['penalty']);loss.backward()
    assert loss>0 and torch.isfinite(x.grad).all() and x.grad.count_nonzero()==0
    for sign in [-1,1]:
        angle=torch.tensor(sign*1e-4,dtype=torch.float64,requires_grad=True)
        pred=actions().double()
        pred[:,6:9]=torch.stack((angle*0,angle.cos(),angle.sin()))
        loss=penalty(evaluate(pred,c),c['penalty']);derivative=torch.autograd.grad(loss,angle)[0]
        assert torch.isfinite(derivative) and derivative*sign<0


def test_joint_objective_counts_once_and_uses_same_weight_scale(tmp_path):
    from types import SimpleNamespace
    from openwam.train.constraint_supervision import RoboTwinConstraintObjective
    b=bundle([joint()]);b['schema_version']='1.4';path=str(tmp_path/'episode.hdf5')
    reader=SimpleNamespace(action_mode='eef',normalize_mode=None,_normalizer=None,_unify_dst_index=None,
        variant='clean_50',split='train',_episode_files=[path],_episode_lengths=[5],_constraint_bundles={path:b})
    objective=RoboTwinConstraintObjective(reader,tmp_path,weight=.5,require_full_coverage=False)
    ref=actions();sample=dict(episode_path=path,start_frame=0,proprio=ref[:1],action=ref[1:],
        proprio_mask=torch.ones_like(ref[:1],dtype=torch.bool),action_mask=torch.ones_like(ref[1:],dtype=torch.bool),trajectory_constraints=b)
    pred=actions(4).requires_grad_();result=objective.bind([sample])(pred[None])
    torch.testing.assert_close(result['loss'],torch.tensor(.5*(.25/.02)**2))
    assert result['metrics']['constraint_count']==1
    result['loss'].backward();assert (pred.grad[:,[9,19]]<0).all()
