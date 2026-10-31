"""First-order generalized selection at an exact angular cusp, CPU only."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.train.action_trajectory import physical_rotation
from openwam.train.constraint_schema import SCHEMA_VERSION, make_center_tangent_hint, validate_constraint
from openwam.train.constraint_supervision import (RoboTwinConstraintObjective, build_constraint_objective, reduce_constraint_metrics,
                                                 validate_constraint_resume)
from openwam.train.meta_constraints import frame_rotation, interval_axis_angle, penalty, residual
from tests.test_constraint_supervision import actions, bundle, constraint


def band(n=100,lo=.5,hi=1.):
    c=constraint('axis_angle_interval',end=n,mode='interval')
    c['parameters'].update(local_axis=[0.,1.,0.],reference_axis=[0.,1.,0.],bounds=[lo,hi])
    q=[math.cos(math.pi/8),-math.sin(math.pi/8),0.,0.]
    c['parameters']['center_tangent_hint']=make_center_tangent_hint(c,q,'test',0)
    validate_constraint(c)
    return c


def axis(theta):
    return torch.stack((theta*0,theta.cos(),theta.sin()))


def theta_loss(theta,c,policy):
    a=axis(theta); target=a.new_tensor([0.,1.,0.])
    phi=interval_axis_angle(a,target,torch.eye(3,dtype=a.dtype),c,policy)
    lo,hi=c['parameters']['bounds']
    return penalty(torch.relu(lo-phi)+torch.relu(phi-hi),c['penalty'])


def evaluate(x,c,policy,*,valid=None,start=0,convention='physical',reference=None):
    return residual(x,x.detach() if reference is None else reference,
        torch.ones_like(x,dtype=torch.bool) if valid is None else valid,c,start_frame=start,
        rotation_convention=convention,angle_center_backward=policy)


def test_exact_center_forward_identical_nonzero_gradient_one_sided_descent():
    c=band();x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    old=theta_loss(x,c,'legacy');new=theta_loss(x,c,'fixed_annotation_v1')
    go=torch.autograd.grad(old,x)[0];gn=torch.autograd.grad(new,x)[0]
    assert torch.equal(old,new) and go==0 and gn>0
    assert gn==pytest.approx(2*.5/.1**2)
    assert theta_loss(x.detach()-.001*gn,c,'legacy')<old
    for eps in (1e-3,1e-5,1e-7):
        plus=theta_loss(x.detach()+eps,c,'legacy');minus=theta_loss(x.detach()-eps,c,'legacy')
        assert (plus-old)/eps<0 and (minus-old)/eps<0
        assert abs(float((plus-minus)/(2*eps)))<1e-12


@pytest.mark.parametrize('theta',[-2.,-.6,-.1,-1e-12,1e-12,.1,.6,2.])
def test_every_nonzero_point_keeps_forward_and_autograd_bitwise(theta):
    c=band();x=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    a=theta_loss(x,c,'legacy');b=theta_loss(x,c,'fixed_annotation_v1')
    ga=torch.autograd.grad(a,x)[0];gb=torch.autograd.grad(b,x)[0]
    assert torch.equal(a,b) and torch.equal(ga,gb)
    if abs(theta)>.01:
        assert torch.autograd.gradcheck(lambda t:theta_loss(t,c,'fixed_annotation_v1'),(x,))


@pytest.mark.parametrize('convention', ['physical','robotwin_wxyz_read_as_xyzw'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_physical_and_legacy_6d_row0_and_other_arm_are_unchanged(convention,dtype):
    x=actions(4).to(dtype)
    if convention!='physical':x[:,[7,17]]=-1
    x.requires_grad_();c=band(4)
    old=penalty(evaluate(x,c,'legacy',convention=convention),c['penalty'])
    new=penalty(evaluate(x,c,'fixed_annotation_v1',convention=convention),c['penalty'])
    assert torch.equal(old,new)
    grad=torch.autograd.grad(new,x)[0]
    assert grad.isfinite().all() and grad[1:,:10].norm()>0
    assert grad[0].count_nonzero()==0 and grad[:,10:].count_nonzero()==0
    nxt=x.detach()-.001*grad
    assert penalty(evaluate(nxt,c,'legacy',convention=convention),c['penalty'])<old
    # Canonical reference-directed escape is a rotation about X for this test.
    r=physical_rotation(nxt[1:,3:9],convention)
    torch.testing.assert_close(r[:,:,0],r.new_tensor([1.,0.,0.]).expand(3,3),rtol=0,atol=1e-7)


def test_masks_long_chunks_and_fixed_hint_do_not_require_observed_source_frame():
    c=band(150);x=actions(33).requires_grad_();valid=torch.ones_like(x,dtype=torch.bool)
    # Source frame0 is outside chunk64..96; the dataset already validated it.
    v=evaluate(x,c,'fixed_annotation_v1',valid=valid,start=64)
    assert len(v)==32
    valid[4,3]=False;bad=x.detach().clone();bad[4,3:9]=float('nan');bad.requires_grad_()
    v=evaluate(bad,c,'fixed_annotation_v1',valid=valid,start=64)
    penalty(v,c['penalty']).backward()
    assert len(v)==31 and bad.grad.isfinite().all() and bad.grad[4].count_nonzero()==0
    valid[:,3:9]=False
    assert evaluate(bad,c,'fixed_annotation_v1',valid=valid,start=64) is None


def test_antiparallel_and_zero_lower_keep_legacy_behavior():
    for lo,a in [(0.,[0.,1.,0.]),(.5,[0.,-1.,0.])]:
        c=band(lo=lo);x=torch.tensor(a,dtype=torch.float64,requires_grad=True);target=x.new_tensor([0.,1.,0.])
        old=interval_axis_angle(x,target,torch.eye(3,dtype=x.dtype),c,'legacy')
        new=interval_axis_angle(x,target,torch.eye(3,dtype=x.dtype),c,'fixed_annotation_v1')
        assert torch.equal(old,new)
        assert torch.equal(torch.autograd.grad(old,x)[0],torch.autograd.grad(new,x)[0])


def test_nonzero_cross_underflow_is_not_misclassified_as_exact_center():
    c=band();x=torch.tensor([0.,1.,1e-30],requires_grad=True);target=x.new_tensor([0.,1.,0.])
    metrics={k:x.new_zeros(()) for k in ('angle_center_hit_count','angle_center_hint_frame_count')}
    old=interval_axis_angle(x,target,torch.eye(3),c,'legacy')
    new=interval_axis_angle(x,target,torch.eye(3),c,'fixed_annotation_v1',metrics)
    assert metrics['angle_center_hit_count']==0 and torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,x)[0],torch.autograd.grad(new,x)[0])


def test_cpu_autocast_is_finite_and_reference_event_mask_is_data_only():
    c=band(3);x=actions(3).requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        old=penalty(evaluate(x,c,'legacy'),c['penalty'])
        new=penalty(evaluate(x,c,'fixed_annotation_v1'),c['penalty'])
    assert torch.equal(old,new)
    grad=torch.autograd.grad(new,x)[0];assert grad.isfinite().all() and grad.norm()>0
    c['activation']=dict(mode='reference_event',start=0,end=2,event=dict(frame=1,direction='decreasing',threshold=.5))
    c['parameters']['center_tangent_hint']=make_center_tangent_hint(c,[math.cos(math.pi/8),-math.sin(math.pi/8),0,0],'test',1)
    ref=x.detach().clone();ref[:,9]=torch.tensor([1.,0.,0.]);valid=torch.ones_like(x,dtype=torch.bool)
    x.data[:,9]=1. # Prediction cannot evade the reference closure.
    assert len(evaluate(x,c,'fixed_annotation_v1',reference=ref,valid=valid))==1
    valid[0,9]=False
    assert evaluate(x,c,'fixed_annotation_v1',reference=ref,valid=valid) is None


def make_objective(tmp_path,policy='fixed_annotation_v1',hint=True,**settings):
    c=band(100)
    if not hint:c['parameters'].pop('center_tangent_hint')
    b=bundle([c]);b['schema_version']=SCHEMA_VERSION
    path=str(tmp_path/'episode0.hdf5')
    reader=SimpleNamespace(action_mode='eef',normalize_mode=None,_normalizer=None,_unify_dst_index=None,
        variant='clean_50',split='train',_episode_files=[path],_episode_lengths=[100],_constraint_bundles={path:b})
    obj=RoboTwinConstraintObjective(reader,tmp_path,weight=.5,require_full_coverage=False,
        angle_center_backward=policy,**settings)
    ref=actions(33);sample=dict(episode_path=path,start_frame=40,proprio=ref[:1],action=ref[1:],
        proprio_mask=torch.ones_like(ref[:1],dtype=torch.bool),action_mask=torch.ones_like(ref[1:],dtype=torch.bool),trajectory_constraints=b)
    return obj,sample


def test_default_unchanged_optin_provenance_resume_and_fixed_id_metrics(tmp_path):
    old,sample=make_objective(tmp_path,'legacy');new,_=make_objective(tmp_path)
    assert old.provenance()['objective_version']=='meta_intervals_v5_rigid_local_point'
    assert 'angle_center_backward' not in old.provenance()
    new.save_provenance(tmp_path/'run');validate_constraint_resume(tmp_path/'run',new)
    with pytest.raises(ValueError,match='changed'):validate_constraint_resume(tmp_path/'run',old)
    x=actions(32)[None].requires_grad_();a=old.bind([sample])(x);b=new.bind([sample])(x)
    assert torch.equal(a['loss'],b['loss'])
    assert b['metrics']['angle_center_hit_count']==32 and b['metrics']['angle_center_invalid_hint_count']==0
    report=reduce_constraint_metrics(None,dict(constraint=b['loss'],constraint_metrics=b['metrics']))['constraint_metrics']
    keys=[k for k in report if k.startswith('angle_center_c_')]
    assert len(keys)==1 and report[keys[0]]==pytest.approx(float(b['loss'].detach()))
    b['loss'].backward();assert x.grad.isfinite().all() and x.grad.norm()>0
    assert len(new.provenance()['angle_center_backward']['contribution_metric_catalog'])==1
    sample['trajectory_constraints']=copy.deepcopy(sample['trajectory_constraints'])
    sample['trajectory_constraints']['constraints'][0]['parameters']['center_tangent_hint']['source']['frame']=1
    with pytest.raises(ValueError):new.bind([sample])


def test_positive_lower_missing_hint_rejected_at_build_not_guessed_from_batch(tmp_path):
    make_objective(tmp_path,'legacy',hint=False)
    with pytest.raises(ValueError,match='lacks its fixed'):make_objective(tmp_path,hint=False)
    with pytest.raises(ValueError,match='policy'):make_objective(tmp_path,policy='nearest_world_axis')


def test_numpy_forward_feature_parity_unchanged_with_and_without_hint():
    c=band();q=np.array([[math.cos(t/2),math.sin(t/2),0.,0.] for t in [-1.2,-.4,0,.2,.8,1.4]])
    expected=np.abs(2*np.arctan2(q[:,1],q[:,0]))
    theta=torch.tensor([-1.2,-.4,0,.2,.8,1.4],dtype=torch.float64)
    a=torch.stack((theta*0,theta.cos(),theta.sin()),-1);target=a.new_tensor([0,1,0]).expand_as(a)
    before=interval_axis_angle(a,target,torch.eye(3,dtype=a.dtype),c,'legacy')
    after=interval_axis_angle(a,target,torch.eye(3,dtype=a.dtype),c,'fixed_annotation_v1')
    assert torch.equal(before,after);np.testing.assert_allclose(after.numpy(),expected,atol=1e-15)


def test_common_physical_frame_rotation_preserves_selected_gradient_covariance():
    c=band();target=torch.tensor([0.,1.,0.],dtype=torch.float64)
    point=target.clone().requires_grad_()
    phi=interval_axis_angle(point,target,torch.eye(3,dtype=point.dtype),c,'fixed_annotation_v1')
    g=torch.autograd.grad((.5-phi)**2/.1**2,point)[0]
    q=[.5,.5,.5,.5];Q=frame_rotation(q,point)
    rotated=copy.deepcopy(c);rotated['reference_frame']['quaternion_wxyz']=q
    a,b,_,_=c['parameters']['center_tangent_hint']['source']['quaternion_wxyz']
    source=[.5*a-.5*b,.5*b+.5*a,.5*a+.5*b,.5*a-.5*b]
    rotated['parameters']['center_tangent_hint']=make_center_tangent_hint(rotated,source,'test',0)
    validate_constraint(rotated)
    point2=(Q@target).requires_grad_();target2=Q@target
    phi2=interval_axis_angle(point2,target2,Q,rotated,'fixed_annotation_v1')
    g2=torch.autograd.grad((.5-phi2)**2/.1**2,point2)[0]
    torch.testing.assert_close(g2,Q@g,rtol=0,atol=1e-13)


def test_fixed_id_contributions_are_aligned_across_different_ddp_samples(tmp_path):
    c=band(100);b=bundle([c]);b['schema_version']=SCHEMA_VERSION
    paths=[str(tmp_path/f'episode{i}.hdf5') for i in range(2)]
    reader=SimpleNamespace(action_mode='eef',normalize_mode=None,_normalizer=None,_unify_dst_index=None,
        variant='clean_50',split='train',_episode_files=paths,_episode_lengths=[100,100],
        _constraint_bundles={p:copy.deepcopy(b) for p in paths})
    obj=RoboTwinConstraintObjective(reader,tmp_path,weight=.5,require_full_coverage=False,
        angle_center_backward='fixed_annotation_v1')
    ref=actions(3)
    outputs=[]
    for path in paths:
        sample=dict(episode_path=path,start_frame=20,proprio=ref[:1],action=ref[1:],
            proprio_mask=torch.ones_like(ref[:1],dtype=torch.bool),action_mask=torch.ones_like(ref[1:],dtype=torch.bool),
            trajectory_constraints=b)
        outputs.append(obj.bind([sample])(ref[None,1:]))
    names=sorted(outputs[0]['metrics']);assert names==sorted(outputs[1]['metrics'])
    class Gather:
        num_processes=2
        def gather(self,values):
            other=(outputs[1]['loss'].reshape(1) if values.shape[-1]==1 else
                   torch.stack([outputs[1]['metrics'][n] for n in names]).reshape(1,-1))
            return torch.cat((values,other))
    report=reduce_constraint_metrics(Gather(),dict(constraint=outputs[0]['loss'],constraint_metrics=outputs[0]['metrics']))
    selected=[v for k,v in report['constraint_metrics'].items() if k.startswith('angle_center_c_')]
    assert len(selected)==2 and selected[0]==pytest.approx(selected[1])
    assert sum(selected)==pytest.approx(report['loss_constraint'])
    assert report['constraint_metrics']['angle_center_hit_count']==4


def test_config_builder_and_diagnostic_toggle_are_resume_bound(tmp_path):
    class Settings(dict):
        __getattr__=dict.__getitem__
    c=band(10);b=bundle([c]);b['schema_version']=SCHEMA_VERSION;path=str(tmp_path/'episode.hdf5')
    reader=SimpleNamespace(action_mode='eef',normalize_mode=None,_normalizer=None,_unify_dst_index=None,
        variant='clean_50',split='train',_episode_files=[path],_episode_lengths=[10],_constraint_bundles={path:b})
    settings=dict(weight=1.,require_full_coverage=False,angle_center_backward='fixed_annotation_v1',
                  angle_center_log_per_constraint=True)
    cfg=SimpleNamespace(training=Settings(constraints=settings,lambda_action=1.),
        dataloader=SimpleNamespace(type='robotwin',variant='clean_50',dataset_dir=tmp_path),
        model=SimpleNamespace(architecture=SimpleNamespace(framework='dual_system',variant='joint_self_attn')))
    obj=build_constraint_objective(cfg,reader);assert obj.angle_center_backward=='fixed_annotation_v1'
    obj.save_provenance(tmp_path/'resume')
    settings['angle_center_log_per_constraint']=False
    other=build_constraint_objective(cfg,reader);assert not other.angle_center_metric_catalog
    with pytest.raises(ValueError,match='changed'):validate_constraint_resume(tmp_path/'resume',other)
    settings['angle_center_log_per_constraint']='false'
    with pytest.raises(ValueError,match='boolean'):build_constraint_objective(cfg,reader)
