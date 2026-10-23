import json
from pathlib import Path

import pytest

from scripts.robotwin_training_gate import (
    check_smoke, copy_snapshot, publication_status, selected_checkpoint, validate_plan,
)


def test_batch_is_global_and_accumulation_is_exact():
    plan = {'num_epochs': 5, 'global_batch': 256, 'gpus': list(range(8)), 'micro_batch': 2}
    assert validate_plan(plan) == 16
    assert validate_plan(dict(plan, micro_batch=8)) == 4
    with pytest.raises(ValueError):
        validate_plan(dict(plan, micro_batch=3))
    with pytest.raises(ValueError):
        validate_plan(dict(plan, global_batch=128))


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
