"""Actual recorder/reader pipeline; requires ordinary RoboTwin CPU reader dependencies."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.train.constraint_supervision import ActionDecoder, RoboTwinConstraintObjective
from tests.test_constraint_local_point import local_point, rotated
from tests.test_constraint_supervision import constraint


@pytest.mark.parametrize('unified,normalized',[(False,False),(True,False),(True,True)])
def test_local_point_hdf5_real_reader_decode_loss(tmp_path,monkeypatch,unified,normalized):
    rax=Path(__file__).resolve().parents[2]/'rax'
    if not rax.exists():pytest.skip('Put the RAX fork beside OpenWAM for recorder integration')
    monkeypatch.syspath_prepend(str(rax))
    from services.robotwin.recording import EpisodeRecorder
    from openwam.dataloader.robotwin import RoboTwinDataset
    root=tmp_path/'task/aloha-agilex_clean_50/data';path=root/'episode0.hdf5'
    c=constraint(end=5);c['parameters'].update(axis=[0,1,0],bounds=[0.,0.],local_point=local_point())
    recorder=EpisodeRecorder(path,recording_id='marker-integration',instruction='Synthetic marker test')
    recorder.upsert(c,0);pixels=np.full((16,16,3),100,dtype=np.uint8)
    for _ in range(9):
        recorder.append(dict(endpose=dict(left_endpose=[0,0,0,1,0,0,0],right_endpose=[0,0,0,1,0,0,0],left_gripper=0.,right_gripper=1.),
            joint_action=dict(vector=np.zeros(14)),images={name:pixels for name in ('head_camera','left_camera','right_camera')}))
    recorder.finish(success=True)
    stats=tmp_path/'stats.npy'
    np.save(stats,{'eef':{'mean':np.zeros(20),'min':np.full(20,-2.),'max':np.full(20,3.)}})
    indices=np.r_[0:10,34:44] if unified else np.arange(20)
    dataset=RoboTwinDataset(data_root=str(root),num_frames=5,height=32,width=32,action_mode='eef',video_stride=1,
        normalize_mode='min-max' if normalized else None,normalization_stats_path=str(stats) if normalized else None,
        variant='clean_50',unify_action=unified,unify_action_map=indices.tolist() if unified else None)
    sample=dataset[0];assert sample['start_frame']==0
    assert sample['trajectory_constraints']['constraints'][0]==c
    decoder=ActionDecoder.from_reader(dataset)
    physical=decoder.decode(sample['action']);torch.testing.assert_close(physical[-1,3:9],torch.tensor(rotated(0,True)))
    pred=sample['action'].clone()
    target=torch.tensor(rotated(math.pi/3,True))
    encoded=(target-torch.tensor(decoder.offset[3:9]))*torch.tensor(decoder.divisor[3:9])
    pred[-1,indices[3:9]]=encoded.to(pred.dtype);pred.requires_grad_()
    obj=RoboTwinConstraintObjective(dataset,tmp_path,weight=1)
    result=obj.bind([sample])(pred[None])
    assert result['loss']>0 and result['metrics']['axis_position_interval_residual_max'].item()==pytest.approx(.2*math.sin(math.pi/3),abs=1e-6)
    result['loss'].backward()
    assert pred.grad[-1,indices[3:9]].abs().sum()>0 and torch.isfinite(pred.grad).all()
    assert pred.grad[:-1].count_nonzero()==0
    if unified:
        assert pred.grad[:,np.setdiff1d(np.arange(80),indices)].count_nonzero()==0
