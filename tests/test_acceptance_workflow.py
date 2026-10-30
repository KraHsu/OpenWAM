"""A new run cannot overtake the benchmark or bypass failed acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import robotwin_acceptance_workflow as workflow
from scripts.robotwin_training_gate import validate_plan as validate_training_plan


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def plan(tmp_path):
    root = tmp_path/'new'
    return dict(version=1, run_root=str(root), benchmark_root=str(tmp_path/'benchmark'),
                source_dataset_root=str(tmp_path/'original'), tasks=[f'task_{i}' for i in range(50)],
                gpus=[0, 1, 2, 3], openwam_root=str(root/'code/OpenWAM'), rax_root=str(root/'code/rax'),
                python='/env/bin/python',
                pilot_tasks=['task_0'], poll_seconds=.01,
                training=dict(run_root=str(root/'training_run'), dataset_root=str(root/'dataset'),
                              openwam_root=str(root/'code/OpenWAM'), rax_root=str(root/'code/rax'),
                              python='/env/bin/python',
                              gpus=[0, 1, 2, 3], global_batch=256, micro_batch=16, num_epochs=5,
                              require_acceptance_v2=True, published_dataset_is_snapshot=True))


def completed(root, tasks):
    write(root/'state.json', dict(phase='complete'))
    write(root/'summary.json', dict(complete=True, completed_episodes=500, completed_tasks=50,
                                   tasks=[dict(task=t, completed_episodes=10, complete=True) for t in tasks]))


def test_only_complete_50_by_10_can_release_next_run(tmp_path):
    p = plan(tmp_path)
    root = Path(p['benchmark_root'])
    assert not workflow.benchmark_complete(root, p['tasks'])
    write(root/'state.json', dict(phase='pipelined_evaluation'))
    assert not workflow.benchmark_complete(root, p['tasks'])
    completed(root, p['tasks'])
    assert workflow.benchmark_complete(root, p['tasks'])
    summary = json.loads((root/'summary.json').read_text())
    summary['tasks'][0]['completed_episodes'] = 9
    write(root/'summary.json', summary)
    with pytest.raises(ValueError, match='50 × 10'):
        workflow.benchmark_complete(root, p['tasks'])
    write(root/'state.json', dict(phase='failed'))
    with pytest.raises(RuntimeError, match='failed'):
        workflow.benchmark_complete(root, p['tasks'])


def test_new_paths_and_acceptance_are_mandatory(tmp_path):
    p = plan(tmp_path)
    workflow.validate_plan(p)
    with pytest.raises(ValueError, match='separate'):
        workflow.validate_plan(dict(p, source_dataset_root=p['run_root']))
    with pytest.raises(ValueError, match='snapshots'):
        workflow.validate_plan(dict(p, rax_root=str(tmp_path/'old_rax')))
    p['training']['require_acceptance_v2'] = False
    with pytest.raises(ValueError):
        workflow.validate_plan(p)
    with pytest.raises(ValueError, match='acceptance-v2'):
        validate_training_plan(p['training'])


def test_plan_changes_cannot_silently_resume_existing_workflow(tmp_path):
    p = tmp_path/'resolved.json'
    workflow.bind(p, dict(weight=.1))
    workflow.bind(p, dict(weight=.1))
    with pytest.raises(ValueError, match='changed'):
        workflow.bind(p, dict(weight=1))


@pytest.mark.parametrize('key,value', [('dataset_root', '/previous/dataset'),
                                     ('openwam_root', '/previous/OpenWAM'),
                                     ('rax_root', '/previous/rax'),
                                     ('python', '/different/python'),
                                     ('published_dataset_is_snapshot', False)])
def test_training_cannot_escape_frozen_workflow_paths(tmp_path, key, value):
    p = plan(tmp_path)
    p['training'][key] = value
    with pytest.raises(ValueError):
        workflow.validate_plan(p)


def test_failed_pilot_never_calls_publication_or_training(tmp_path, monkeypatch):
    p = plan(tmp_path)
    run = workflow.Workflow(p)
    completed(Path(p['benchmark_root']), p['tasks'])
    monkeypatch.setattr(run, 'wait_idle', lambda: None)
    monkeypatch.setattr(run, 'prepare_templates', lambda: [dict(task='task_0')])
    calls = []

    def fail(jobs):
        raise RuntimeError('failed_variations_pass_constraints')

    monkeypatch.setattr(run, 'validate_jobs', fail)
    monkeypatch.setattr(run, 'execute', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match='failed_variations'):
        run.run()
    assert calls == []
    assert json.loads((Path(p['run_root'])/'state.json').read_text())['phase'] == 'blocked'


def test_gpu_filter_does_not_claim_or_kill_unrelated_jobs(monkeypatch):
    answers = iter(['0, GPU-A\n1, GPU-B\n2, GPU-C\n',
                    '111, GPU-A, python\n222, GPU-C, unrelated\n'])
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(answers)))
    assert workflow.selected_gpu_jobs([0, 1]) == ['111, GPU-A, python']


@pytest.mark.parametrize('previous_phase', ['complete', 'training_gate'])
def test_completed_training_is_terminal_even_if_workflow_died_before_final_state(tmp_path, monkeypatch, previous_phase):
    p = plan(tmp_path)
    run = workflow.Workflow(p)
    write(run.root/'training_plan.json', p['training'])
    write(run.root/'state.json', dict(phase=previous_phase))
    trained = dict(phase='complete', run=str(run.root/'training_run/training/run1'))
    write(Path(p['training']['run_root'])/'state.json', trained)
    def forbidden(*args, **kwargs):
        pytest.fail('Completed runs must not wait, publish, or launch a child again')
    monkeypatch.setattr(workflow, 'benchmark_complete', forbidden)
    monkeypatch.setattr(run, 'execute', forbidden)
    run.run()
    assert workflow.read(run.root/'state.json')['phase'] == 'complete'
    assert workflow.read(run.root/'state.json')['training'] == trained


def test_terminal_resume_rejects_mismatched_training_identity(tmp_path):
    p = plan(tmp_path)
    run = workflow.Workflow(p)
    write(Path(p['training']['run_root'])/'state.json', dict(phase='complete'))
    write(run.root/'training_plan.json', dict(p['training'], micro_batch=8))
    with pytest.raises(ValueError, match='matching frozen training plan'):
        run.run()
