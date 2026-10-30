import json
import os
from types import SimpleNamespace

import pytest

from scripts.robotwin_training_gate import (
    check_smoke, copy_snapshot, expected_training_steps, publication_status,
    selected_checkpoint, training_command, validate_plan,
)
from scripts import robotwin_training_gate as gate


def test_batch_is_global_and_accumulation_is_exact():
    plan = {'num_epochs': 5, 'global_batch': 256, 'gpus': list(range(8)), 'micro_batch': 2}
    assert validate_plan(plan) == 16
    assert validate_plan(dict(plan, micro_batch=8)) == 4
    assert validate_plan(dict(plan, gpus=list(range(4)), micro_batch=16)) == 4
    with pytest.raises(ValueError):
        validate_plan(dict(plan, micro_batch=3))
    with pytest.raises(ValueError):
        validate_plan(dict(plan, global_batch=128))


@pytest.mark.parametrize('gpus', [[], [0, 0, 1, 2], [-1, 0, 1, 2], [False, 1, 2, 3]])
def test_invalid_gpu_selections_are_rejected(gpus):
    with pytest.raises(ValueError):
        validate_plan({'num_epochs': 5, 'global_batch': 256, 'gpus': gpus, 'micro_batch': 16})


def test_four_gpu_launch_and_five_epoch_completion_use_actual_world_size():
    plan = {'num_epochs': 5, 'global_batch': 256, 'gpus': [0, 1, 2, 3],
            'micro_batch': 16, 'python': '/env/bin/python'}
    command = training_command(plan, '/run')
    assert '--nproc-per-node=4' in command
    assert '--nproc-per-node=8' not in command
    assert expected_training_steps(549998, plan) == 42970
    assert expected_training_steps(549998, dict(plan, gpus=list(range(8)))) == 21485


def test_gpu_wait_only_considers_reserved_devices_and_rejects_missing_device(monkeypatch):
    def query(command, **kwargs):
        if '--query-gpu=index,uuid' in command:
            return SimpleNamespace(stdout='0, GPU-A\n1, GPU-B\n4, GPU-E\n')
        return SimpleNamespace(stdout='111, GPU-A, our-device-busy\n222, GPU-E, unrelated-job\n')
    monkeypatch.setattr(gate.subprocess, 'run', query)
    assert gate.gpu_processes([0, 1]) == ['111, GPU-A, our-device-busy']
    assert gate.gpu_processes([1]) == []
    with pytest.raises(ValueError, match='unavailable'):
        gate.gpu_processes([2])


def test_child_import_path_binds_frozen_code_before_inherited_environment(tmp_path, monkeypatch):
    plan = dict(openwam_root=str(tmp_path/'new/OpenWAM'), rax_root=str(tmp_path/'new/rax'))
    monkeypatch.setenv('PYTHONPATH', '/old/editable:/other/library')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5,6,7')
    env = gate.execution_environment(plan, cuda='0,1,2,3')
    assert env['PYTHONPATH'].split(os.pathsep) == [plan['openwam_root'], plan['rax_root'],
                                                   '/old/editable', '/other/library']
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert gate.execution_environment(plan)['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'


def test_publication_requires_each_task_and_completed_annotation(tmp_path):
    tasks = [f'task_{i}' for i in range(50)]
    manifest = {'tasks': tasks, 'per_task': 50, 'collection': 'clean', 'running': {}, 'failed_tasks': {}, 'pending_tasks': []}
    (tmp_path/'batch_manifest.json').write_text(json.dumps(manifest))
    for task in tasks:
        root = tmp_path/task/'aloha-agilex_clean_50'
        (root/'data').mkdir(parents=True)
        (root/'annotation_queue').mkdir()
        for i in range(50):
            (root/'data'/f'episode{i}.hdf5').touch()
            (root/'annotation_queue'/f'episode{i}.json').write_text('{"annotation_status":"done"}')
    assert publication_status(tmp_path)['ready']
    receipt = root/'annotation_queue/episode49.json'
    receipt.write_text('{"annotation_status":"processing"}')
    assert not publication_status(tmp_path)['ready']
    receipt.write_text('{"annotation_status":"done"}')
    (root/'data/episode49.hdf5').unlink()
    assert not publication_status(tmp_path)['ready']
    (tmp_path/tasks[0]/'aloha-agilex_clean_50/data/episode50.hdf5').touch()
    with pytest.raises(ValueError, match='quota'):
        publication_status(tmp_path)


def test_checkpoint_is_explicit_and_architecture_mismatch_is_rejected(tmp_path):
    assert selected_checkpoint({'initial_checkpoint': None}) is None
    (tmp_path/'config.yaml').write_text('model:\n  architecture:\n    framework: la_tri_system\n    variant: idm\n    action_dim: 20\n')
    (tmp_path/'checkpoint_step_1.safetensors').touch()
    with pytest.raises(ValueError, match='adaptation'):
        selected_checkpoint({'initial_checkpoint': str(tmp_path)})


def test_snapshot_is_independent_of_later_annotation_revisions(tmp_path):
    source, dest = tmp_path/'source', tmp_path/'snapshot'
    episode = source/'task/aloha-agilex_clean_50/data/episode0.hdf5'
    episode.parent.mkdir(parents=True)
    episode.write_bytes(b'old-label-and-trajectory')
    (source/'batch_manifest.json').write_text('{}')
    copy_snapshot(source, dest)
    episode.write_bytes(b'new-label-and-trajectory')
    frozen = dest/episode.relative_to(source)
    assert frozen.read_bytes() == b'old-label-and-trajectory'
    assert frozen.stat().st_ino != episode.stat().st_ino
    copy_snapshot(source, dest)
    assert frozen.read_bytes() == b'old-label-and-trajectory'


def test_resume_smoke_requires_actual_next_update_and_finite_gradient(tmp_path):
    directory = tmp_path/'accel_state_step_32'
    directory.mkdir()
    (directory/'trainer_state.json').write_text('{"global_step":32,"opt_step":2}')
    row = {'step': 16, 'loss_total': 1., 'loss_action': .5, 'loss_constraint': .1,
           'grad_norm': 2., 'constraint_metrics': {'constraint_count': 8}}
    journal = tmp_path/'metrics.jsonl'
    journal.write_text(json.dumps(row)+'\n'+json.dumps(dict(row, step=32))+'\n')
    assert len(check_smoke(tmp_path, 16, resumed=True)) == 2
    journal.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='duplicated or skipped'):
        check_smoke(tmp_path, 16, resumed=True)
