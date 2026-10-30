"""Physical local markers share geometry across decoder, reference and loss."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.train.action_trajectory import EEFActionSpec, restore_eef_trajectory
from openwam.train.constraint_schema import SCHEMA_VERSION, validate_bundle, validate_constraint
from openwam.train.constraint_supervision import RoboTwinConstraintObjective
from openwam.train.meta_constraints import penalty, residual, rigid_point_positions
from tests.test_constraint_supervision import actions, bundle, constraint


def local_point(offset=(.2,0.,0.)):
    return dict(frame='subject_eef',offset_m=list(offset),provenance=dict(source='known_geometry',
        reference='synthetic rigid marker',sha256='a'*64,rationale='Analytic marker, no contact claim'))


def marker(kind='axis_position_interval',*,n=3,anchor='predicted_start'):
    c=constraint(kind,end=n,mode='interval');c['parameters']['local_point']=local_point()
    c['parameters']['bounds']=[0.,0.]
    if kind!='point_distance_interval':c['parameters']['axis']=[1,0,0]
    if kind=='axis_displacement_interval':c['parameters']['anchor']=anchor
    c['reference_frame']['origin_m']=[1.,2.,0.]
    c['reference_frame']['quaternion_wxyz']=[math.sqrt(.5),0,0,math.sqrt(.5)]
    return c


def evaluate(x,c,valid=None,start=0,convention='physical',reference=None):
    validate_constraint(c)
    return residual(x,x.detach() if reference is None else reference,
        torch.ones_like(x,dtype=torch.bool) if valid is None else valid,c,
        start_frame=start,rotation_convention=convention)


def rotated(theta,legacy=False):
    a,b=math.cos(theta),math.sin(theta)
    return [1.,0,0,0,-a,b] if legacy else [a,b,0,-b,a,0]


def trajectory():
    x=actions(3).double();x[:,:3]=torch.tensor([1.,2.,0.])
    for i,t in enumerate([0.,math.pi/2,math.pi]):x[i,3:9]=torch.tensor(rotated(t))
    return x


@pytest.mark.parametrize('kind,expected',[
    ('axis_position_interval',[.2,0.]),('radial_distance_interval',[0.,.2]),
    ('point_distance_interval',[.2,.2]),('axis_displacement_interval',[.2,0.])])
def test_analytic_point_rotation_ref_basis_and_numpy_parity(kind,expected):
    x=trajectory();c=marker(kind);v=evaluate(x,c)
    torch.testing.assert_close(v,torch.tensor(expected,dtype=x.dtype),atol=1e-14,rtol=0)
    # Optional sibling integration; core analytic checks always run in this repo.
    sibling=Path(__file__).resolve().parents[2]/'rax'
    if sibling.is_dir():
        import sys
        sys.path.insert(0,str(sibling))
        try:
            from services.robotwin.annotation import reference_values
            h={f'endpose/{a}_endpose':np.array([[1,2,0,1,0,0,0],[1,2,0,math.sqrt(.5),0,0,math.sqrt(.5)],[1,2,0,0,0,0,1]]) for a in ('left','right')}
            h.update({f'endpose/{a}_gripper':np.zeros(3) for a in ('left','right')})
            np.testing.assert_allclose(v.detach().numpy(),np.abs(reference_values(h,c)[1:]),atol=1e-14)
        finally:sys.path.remove(str(sibling))


def test_point_gradient_matches_finite_difference_and_analytic_rotation():
    c=marker(n=2);c['reference_frame']=dict(origin_m=[0.,0.,0.],quaternion_wxyz=[1,0,0,0])
    c['parameters'].update(axis=[0,1,0],local_point=local_point((.2,.03,.05)))
    x=actions(2).double();x[-1,:3]=torch.tensor([.1,.2,.3]);x[-1,3:9]=torch.tensor(rotated(.4),dtype=x.dtype);x.requires_grad_()
    f=lambda t: penalty(evaluate(t,c),c['penalty'])
    assert torch.autograd.gradcheck(f,(x,),eps=1e-6,atol=2e-5,rtol=1e-4)
    f(x).backward();assert x.grad[-1,:3].abs().sum()>0 and x.grad[-1,3:9].abs().sum()>0
    theta=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    rot=torch.stack((theta.cos(),theta.sin(),theta*0,-theta.sin(),theta.cos(),theta*0))
    y=rigid_point_positions(torch.zeros(3,dtype=theta.dtype),rot,c['parameters'],'physical')[1]
    derivative=torch.autograd.grad(y,theta)[0]
    assert derivative.item()==pytest.approx(.2*math.cos(.4)-.03*math.sin(.4))
    assert x.grad[0].count_nonzero()==0  # Measured row 0 is not an active loss sample.


def test_nonzero_point_masks_rotation_and_displacement_anchor_not_identity_fallback():
    c=marker('axis_displacement_interval');x=trajectory();valid=torch.ones_like(x,dtype=torch.bool)
    valid[1,8]=False;x[1,3:9]=float('nan')
    assert evaluate(x,c,valid).shape==(1,)
    valid[0,3]=False
    assert evaluate(x,c,valid) is None  # Anchor orientation is required too.
    assert evaluate(trajectory(),c,start=1) is None  # Original anchor outside chunk.
    fixed=marker('axis_displacement_interval',anchor='reference_frame');fixed['activation']['end']=4
    assert evaluate(trajectory(),fixed,start=1).shape==(2,)
    valid=torch.ones_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='orientation'):evaluate(x,c,valid)
    x[1,3:9]=0
    with pytest.raises(ValueError,match='degenerate'):evaluate(x,c,valid)


def test_explicit_zero_offset_keeps_old_position_only_masks_and_numbers():
    x=trajectory();valid=torch.ones_like(x,dtype=torch.bool);valid[:,3:9]=False;x[:,3:9]=float('nan')
    c=marker('axis_displacement_interval');c['parameters']['local_point']=local_point((0,0,0))
    old=copy.deepcopy(c);del old['parameters']['local_point']
    torch.testing.assert_close(evaluate(x,c,valid),evaluate(x,old,valid),rtol=0,atol=0)


def test_reference_event_uses_frozen_reference_gripper_with_local_marker():
    c=marker(n=3);c['activation']=dict(mode='reference_event',start=0,end=2,event=dict(frame=1,direction='decreasing',threshold=.5))
    x=trajectory();x[:,9]=1  # Model cannot evade fixed activation by refusing to close.
    ref=x.detach().clone();ref[:,9]=torch.tensor([1,0,0])
    assert evaluate(x,c,reference=ref).item()==pytest.approx(.2)
    valid=torch.ones_like(x,dtype=torch.bool);valid[1,3]=False
    assert evaluate(x,c,reference=ref,valid=valid) is None


@pytest.mark.parametrize('arm',['left','right'])
def test_normalized_unified_legacy_decoder_to_loss_detaches_measured_anchor(tmp_path,arm):
    c=marker('axis_displacement_interval',n=2);c['subject']=arm+'_eef'
    c['reference_frame']=dict(origin_m=[0.,0.,0.],quaternion_wxyz=[1,0,0,0]);c['parameters']['axis']=[0,1,0]
    b=bundle([c]);b['schema_version']=SCHEMA_VERSION;b['action_contract']['quaternion_order']='wxyz'
    path=str(tmp_path/'episode0.hdf5');indices=np.r_[0:10,34:44]
    reader=SimpleNamespace(action_mode='eef',normalize_mode='min-max',_unify_dst_index=indices,
        _normalizer=SimpleNamespace(_scale=np.full(20,2.),_offset=np.full(20,.25)),
        variant='clean_50',split='train',_episode_files=[path],_episode_lengths=[2],_constraint_bundles={path:b})
    obj=RoboTwinConstraintObjective(reader,tmp_path,weight=1,require_full_coverage=False)
    physical=actions(2)
    for offset in (0,10):physical[:,offset+3:offset+9]=torch.tensor(rotated(0,legacy=True))
    offset=0 if arm=='left' else 10
    physical[-1,offset+3:offset+9]=torch.tensor(rotated(math.pi/3,legacy=True))
    encoded=torch.zeros(2,80);encoded[:,indices]=(physical-.25)*2
    measured=encoded[:1].clone().requires_grad_();pred=encoded[1:].clone().unsqueeze(0).requires_grad_()
    mask=torch.zeros_like(encoded,dtype=torch.bool);mask[:,indices]=True
    sample=dict(episode_path=path,start_frame=0,proprio=measured,action=encoded[1:],proprio_mask=mask[:1],action_mask=mask[1:],trajectory_constraints=b)
    result=obj.bind([sample])(pred);assert result['loss'].item()==pytest.approx(3.,abs=1e-5)
    assert result['metrics']['axis_displacement_interval_count'].item()==1
    result['loss'].backward();assert measured.grad is None
    assert torch.isfinite(pred.grad).all() and pred.grad[0,0,indices[offset+3:offset+9]].abs().sum()>0
    unused=np.setdiff1d(np.arange(80),indices);assert pred.grad[:,:,unused].count_nonzero()==0
    assert pred.grad[0,0,indices[10-offset:20-offset]].count_nonzero()==0


def test_delta_restoration_precedes_rigid_marker_and_preserves_gradient():
    start=actions(1).double()[0];delta=actions(2).double()
    delta[:,3:9]=torch.tensor(rotated(math.pi/2),dtype=delta.dtype);delta.requires_grad_()
    restored=restore_eef_trajectory(delta,start,EEFActionSpec(position='step_delta',rotation='step_delta'))
    c=marker('axis_displacement_interval');c['reference_frame']=dict(origin_m=[0.,0.,0.],quaternion_wxyz=[1,0,0,0])
    c['parameters']['axis']=[1,0,0]
    v=evaluate(torch.cat((start[None],restored)),c)
    torch.testing.assert_close(v,torch.tensor([.2,.4],dtype=v.dtype))
    penalty(v,c['penalty']).backward();assert torch.isfinite(delta.grad).all() and delta.grad[:,3:9].abs().sum()>0


def test_schema_1_5_and_shared_contract_roundtrip():
    c=marker();b=bundle([c])
    with pytest.raises(ValueError,match='1.5'):validate_bundle(b)
    b['schema_version']=SCHEMA_VERSION;assert validate_bundle(json.loads(json.dumps(b)))==b
    for bad in ['world','reference_frame','raw_checkpoint']:
        trial=copy.deepcopy(c);trial['parameters']['local_point']['frame']=bad
        with pytest.raises(ValueError):validate_constraint(trial)
    wrong=constraint('axis_angle_interval');wrong['parameters']['local_point']=local_point()
    with pytest.raises(ValueError):validate_constraint(wrong)
    sibling=Path(__file__).resolve().parents[2]/'rax/services/robotwin/constraint_schema.py'
    if sibling.exists():assert sibling.read_bytes()==(Path(__file__).resolve().parents[1]/'openwam/train/constraint_schema.py').read_bytes()
