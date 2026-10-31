"""Real HDF5 recorder -> real reader -> 32-action physical loss; CPU only."""
import copy
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from openwam.train.constraint_schema import HDF5_KEY, canonical_json, make_center_tangent_hint
from openwam.train.constraint_supervision import ActionDecoder, RoboTwinConstraintObjective
from tests.test_constraint_angle_center import band


@pytest.mark.parametrize('unified,normalized',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('quaternion_order',['wxyz','xyzw'])
def test_real_hdf5_fixed_hint_long_chunk_6d_affine_mask_and_checkpoint(tmp_path,monkeypatch,unified,normalized,quaternion_order):
    rax=Path(os.environ.get('RAX_CONSTRAINT_TEST_ROOT',str(Path(__file__).resolve().parents[2]/'rax')))
    if not rax.exists():pytest.skip('Set RAX_CONSTRAINT_TEST_ROOT to the paired RAX checkout')
    monkeypatch.syspath_prepend(str(rax))
    from services.robotwin.recording import EpisodeRecorder
    from openwam.dataloader.robotwin import RoboTwinDataset
    root=tmp_path/'task/aloha-agilex_clean_50/data';path=root/'episode0.hdf5'
    q=[math.cos(math.pi/8),-math.sin(math.pi/8),0.,0.]
    c=band(100);recorder=EpisodeRecorder(path,recording_id='test',instruction='Synthetic angle center')
    recorder.upsert(c,0);pixels=np.full((16,16,3),100,dtype=np.uint8)
    for _ in range(100):
        recorder.append(dict(endpose=dict(left_endpose=[0,0,0,*q],right_endpose=[0,0,0,1,0,0,0],
            left_gripper=0.,right_gripper=1.),joint_action=dict(vector=np.zeros(14)),
            images={name:pixels for name in ('head_camera','left_camera','right_camera')}))
    recorder.finish(success=True)
    if quaternion_order=='xyzw':
        # A second explicit physical source format, created only inside this
        # temporary test dataset. Native RAX recording remains WXYZ.
        import json
        with h5py.File(path,'r+') as h:
            b=json.loads(h[HDF5_KEY][()].decode());b['action_contract']['quaternion_order']='xyzw'
            for arm in ('left','right'):
                key=f'endpose/{arm}_endpose';pose=h[key][:];pose[:,3:]=pose[:,[4,5,6,3]];h[key][:]=pose
            h[HDF5_KEY][()]=canonical_json(b)
    stats=tmp_path/'stats.npy';np.save(stats,{'eef':{'mean':np.zeros(20),'min':np.full(20,-2.),'max':np.full(20,3.)}})
    indices=np.r_[0:10,34:44] if unified else np.arange(20)
    kwargs=dict(data_root=str(root),num_frames=33,height=32,width=32,action_mode='eef',video_stride=4,
        normalize_mode='min-max' if normalized else None,normalization_stats_path=str(stats) if normalized else None,
        variant='clean_50',unify_action=unified,unify_action_map=indices.tolist() if unified else None)
    dataset=RoboTwinDataset(**kwargs);sample=dataset[dataset._window_index.index((0,64))]
    assert sample['start_frame']==64 and len(sample['action'])==32
    received=next(c for c in sample['trajectory_constraints']['constraints'] if c['id']=='c0')
    assert received==c and received['parameters']['center_tangent_hint']['source']['frame']==0
    decoder=ActionDecoder.from_reader(dataset);pred=sample['action'].clone()
    # Exact raw legacy-6D identity after WXYZ-as-XYZW reader decoding.
    raw=torch.tensor([1.,0.,0.,0.,-1. if quaternion_order=='wxyz' else 1.,0.])
    encoded=(raw-torch.tensor(decoder.offset[3:9]))*torch.tensor(decoder.divisor[3:9])
    pred[:,indices[3:9]]=encoded;pred.requires_grad_();sample['proprio'].requires_grad_()
    objective=RoboTwinConstraintObjective(dataset,tmp_path,weight=1.,angle_center_backward='fixed_annotation_v1')
    legacy=RoboTwinConstraintObjective(dataset,tmp_path,weight=1.)
    callback=objective.bind([sample]);out=callback(pred[None]);old=legacy.bind([sample])(pred[None])
    assert torch.equal(out['loss'],old['loss']) and out['metrics']['angle_center_hit_count']==32
    out['loss'].backward();assert pred.grad.isfinite().all() and pred.grad.norm()>0
    assert sample['proprio'].grad is None # measured initial state is detached
    assert pred.grad[:,indices[10:]].count_nonzero()==0
    if unified:assert pred.grad[:,np.setdiff1d(np.arange(80),indices)].count_nonzero()==0
    updated=pred.detach()-1e-4*pred.grad
    assert legacy.bind([sample])(updated[None])['loss']<old['loss']
    objective.save_provenance(tmp_path/'checkpoint')
    assert (tmp_path/'checkpoint/constraint_provenance.json').is_file()
    masked=copy.deepcopy(sample);masked['action_mask'][5,indices[3]]=False
    p2=pred.detach().clone();p2[5,indices[3:9]]=float('nan');p2.requires_grad_()
    out2=objective.bind([masked])(p2[None]);out2['loss'].backward()
    assert out2['metrics']['angle_center_hit_count']==31 and p2.grad.isfinite().all()
    # A self-consistent forged hint still fails full-episode HDF reference validation.
    with h5py.File(path,'r+') as h:
        import json
        b=json.loads(h[HDF5_KEY][()].decode());ca=next(c for c in b['constraints'] if c['id']=='c0')
        ca['parameters']['center_tangent_hint']=make_center_tangent_hint(ca,[q[0],-q[1],0,0],'test',0)
        h[HDF5_KEY][()]=canonical_json(b)
    with pytest.raises(ValueError,match='HDF5 reference'):RoboTwinDataset(**kwargs)
