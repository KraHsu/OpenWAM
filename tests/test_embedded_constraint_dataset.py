"""RAX recording -> actual RoboTwin reader -> differentiable geometric loss."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch


def test_rax_recording_reader_loss_roundtrip(tmp_path, monkeypatch):
    rax = Path(__file__).resolve().parents[2]/'rax'
    if not rax.exists():
        pytest.skip('Cross-repository integration: put the RAX fork beside OpenWAM')
    monkeypatch.syspath_prepend(str(rax))
    from services.robotwin.recording import EpisodeRecorder
    from openwam.dataloader.robotwin import RoboTwinDataset
    from openwam.train.constraint_supervision import RoboTwinConstraintObjective
    from tests.test_constraint_supervision import constraint
    root = tmp_path/'task/aloha-agilex_clean_50/data'
    path = root/'episode0.hdf5'
    recorder = EpisodeRecorder(path, recording_id='integration-test', instruction='Synthetic integration instruction')
    c = constraint(end=5); c['parameters']['bounds'] = [0, 0.1]
    recorder.upsert(c, 0)
    pixels = np.full((16, 16, 3), (200, 80, 20), dtype=np.uint8)
    for index in range(9):
        recorder.append({'endpose': {'left_endpose': [0, 0, 0.01*index, 1, 0, 0, 0],
                                     'right_endpose': [0, 0, 0, 1, 0, 0, 0], 'left_gripper': 0., 'right_gripper': 1.},
                         'joint_action': {'vector': np.zeros(14)},
                         'images': {name: pixels for name in ('head_camera', 'left_camera', 'right_camera')}})
    recorder.finish(success=True)
    dataset = RoboTwinDataset(data_root=str(root), num_frames=5, height=32, width=32,
                              action_mode='eef', video_stride=1, normalize_mode=None, variant='clean_50')
    sample = dataset[0]
    assert sample['trajectory_constraints']['constraints'][0] == c
    assert 'Synthetic integration instruction' in sample['prompt']
    np.testing.assert_allclose(np.asarray(sample['video'][0]).mean((0, 1)), [200, 80, 20], atol=3)
    obj = RoboTwinConstraintObjective(dataset, tmp_path, weight=0.1)
    pred = sample['action'].clone()
    pred[-1, 2] = 0.4
    pred.requires_grad_()
    result = obj.bind([sample])(pred[None])
    assert result['loss'] > 0 and result['metrics']['constraint_count'] >= 1
    result['loss'].backward()
    assert pred.grad[-1, 2] > 0 and pred.grad[:-1].count_nonzero() == 0
    assert torch.isfinite(pred.grad).all()
