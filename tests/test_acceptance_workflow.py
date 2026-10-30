"""A new run cannot overtake the benchmark or bypass failed acceptance."""
import json
import hashlib
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


def stopped_receipt(p):
    benchmark = Path(p['benchmark_root'])
    write(benchmark/'state.json', dict(phase='stopped_by_user'))
    write(benchmark/'summary.json', dict(complete=False, completed_episodes=31))
    receipt = dict(version=1, kind='user_requested_benchmark_stop', benchmark_root=p['benchmark_root'],
                   run_root=p['run_root'], stopped_unix=1800000000,
                   summary_sha256=hashlib.sha256((benchmark/'summary.json').read_bytes()).hexdigest())
    write(benchmark/'user_stop.json', dict(receipt, owned_processes_stopped=True))
    receipt['benchmark_stop_receipt_sha256'] = hashlib.sha256((benchmark/'user_stop.json').read_bytes()).hexdigest()
    path = Path(p['run_root'])/'benchmark_stop_receipt.json'
    write(path, receipt)
    return path


def test_explicit_user_stop_releases_dependency_without_claiming_completion(tmp_path, monkeypatch):
    p = plan(tmp_path)
    receipt_path = stopped_receipt(p)
    assert not workflow.benchmark_complete(p['benchmark_root'], p['tasks'])
    assert workflow.benchmark_stop_release(p) is None  # Presence alone is not authorization.
    release = workflow.benchmark_stop_release(p, receipt_path)
    assert release['receipt_sha256'] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert not (Path(p['run_root'])/'benchmark_release.json').exists()
    run = workflow.Workflow(p, benchmark_stop_receipt=receipt_path)
    entered = []
    monkeypatch.setattr(run, 'wait_idle', lambda: entered.append('wait_idle'))
    def reach_acceptance():
        entered.append('acceptance')
        raise RuntimeError('test reached acceptance without waiting for 500')
    monkeypatch.setattr(run, 'prepare_templates', reach_acceptance)
    with pytest.raises(RuntimeError, match='reached acceptance'):
        run.run()
    assert entered == ['wait_idle', 'acceptance']
    assert workflow.read(Path(p['run_root'])/'benchmark_release.json') == release
    assert workflow.benchmark_stop_release(p) == release  # Explicit authorization survives restart.
    assert workflow.read(Path(p['benchmark_root'])/'state.json')['phase'] == 'stopped_by_user'
    assert workflow.read(Path(p['benchmark_root'])/'summary.json')['complete'] is False


@pytest.mark.parametrize('field,value', [('run_root', '/different/run'),
                                      ('benchmark_root', '/different/benchmark'),
                                      ('summary_sha256', '0'*64),
                                      ('benchmark_stop_receipt_sha256', '0'*64)])
def test_stop_receipt_rejects_wrong_roots_and_hashes(tmp_path, field, value):
    p = plan(tmp_path)
    path = stopped_receipt(p)
    receipt = workflow.read(path)
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError):
        workflow.benchmark_stop_release(p, path)


def test_stop_receipt_requires_actual_stop_state_and_matching_evidence(tmp_path):
    p = plan(tmp_path)
    path = stopped_receipt(p)
    benchmark = Path(p['benchmark_root'])
    write(benchmark/'state.json', dict(phase='failed'))
    with pytest.raises(ValueError, match='stopped_by_user'):
        workflow.benchmark_stop_release(p, path)
    with pytest.raises(RuntimeError, match='failed'):
        workflow.benchmark_complete(benchmark, p['tasks'])
    write(benchmark/'state.json', dict(phase='stopped_by_user'))
    raw = workflow.read(benchmark/'user_stop.json')
    raw['stopped_unix'] += 1
    write(benchmark/'user_stop.json', raw)
    receipt = workflow.read(path)
    receipt['benchmark_stop_receipt_sha256'] = hashlib.sha256((benchmark/'user_stop.json').read_bytes()).hexdigest()
    write(path, receipt)
    with pytest.raises(ValueError, match='differs in stopped_unix'):
        workflow.benchmark_stop_release(p, path)


def test_frozen_release_rejects_changed_receipt_even_when_reformatted(tmp_path):
    p = plan(tmp_path)
    path = stopped_receipt(p)
    release = workflow.benchmark_stop_release(p, path)
    workflow.bind(Path(p['run_root'])/'benchmark_release.json', release)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='Frozen benchmark release changed'):
        workflow.benchmark_stop_release(p)


def test_check_only_validates_stop_receipt_without_writing_release(tmp_path, monkeypatch, capsys):
    p = plan(tmp_path)
    receipt = stopped_receipt(p)
    path = tmp_path/'plan.json'
    write(path, p)
    monkeypatch.setattr(workflow.sys, 'argv', ['workflow', '--plan', str(path), '--check-only',
                                            '--benchmark-stop-receipt', str(receipt)])
    workflow.main()
    assert json.loads(capsys.readouterr().out) == dict(valid=True, benchmark_complete=False,
                                                     benchmark_released_by_user=True)
    assert not (Path(p['run_root'])/'benchmark_release.json').exists()
