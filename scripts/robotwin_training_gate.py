#!/usr/bin/env python3
"""Persistent, fail-closed clean-data -> audit -> real smoke/resume -> training gate.

Only launches this experiment's children. Never kills collectors or other GPU
jobs. A missing checkpoint waits for the plan to be filled; a failed validation
stops with a durable error, rather than silently starting an unchecked run.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_robotwin_constraints import atomic_json


def publication_status(root):
    root = Path(root)
    m = json.loads((root/'batch_manifest.json').read_text())
    if len(set(m['tasks'])) != 50 or m['per_task'] != 50 or m['collection'] != 'clean':
        raise ValueError('Expected 50 clean tasks × 50 distinct episodes')
    counts = {task: len(list((root/task/'aloha-agilex_clean_50/data').glob('episode*.hdf5'))) for task in m['tasks']}
    if any(n > 50 for n in counts.values()):
        raise ValueError('Per-task quota exceeded')
    queues = list(root.glob('*/aloha-agilex_clean_50/annotation_queue/episode*.json'))
    done = sum(json.loads(p.read_text()).get('annotation_status') == 'done' for p in queues)
    return {'ready': (all(n == 50 for n in counts.values()) and len(queues) == done == 2500
                      and not m.get('running') and not m.get('pending_tasks') and not m.get('failed_tasks')),
            'published': sum(counts.values()), 'per_task': counts, 'queue_done': done,
            'failed_tasks': m.get('failed_tasks', {})}


def validate_plan(plan):
    if plan['num_epochs'] != 5 or plan['global_batch'] != 256:
        raise ValueError('This experiment was requested as 5 epochs / global batch 256')
    gpus = plan['gpus']
    world = len(gpus)
    micro = plan['micro_batch']
    if (not world or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != world or type(micro) is not int
            or micro < 1 or 256 % (world*micro)):
        raise ValueError('Require distinct GPU indices and world size × microbatch dividing global batch 256')
    return 256 // (world*micro)


def expected_training_steps(windows, plan):
    """Count microsteps after Accelerate shards/pads the per-rank dataloaders."""
    validate_plan(plan)
    return plan['num_epochs']*math.ceil(windows/(len(plan['gpus'])*plan['micro_batch']))


def training_command(plan, root):
    validate_plan(plan)
    return [plan['python'], '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            f"--nproc-per-node={len(plan['gpus'])}", 'scripts/train.py', '--config-path', str(root)]


def selected_checkpoint(plan):
    """Require an explicit, compatible initialization; never auto-pick a Robotwin run."""
    if not plan.get('initial_checkpoint'):
        return None
    from omegaconf import OmegaConf
    from openwam.train.utils.checkpointing import find_latest_weights
    p = Path(plan['initial_checkpoint']).resolve()
    directory = p if p.is_dir() else p.parent
    cfg = OmegaConf.load(directory/'config.yaml')
    a = cfg.model.architecture
    if (a.framework, a.variant, int(a.action_dim)) != ('dual_system', 'joint_self_attn', 80):
        raise ValueError('Selected checkpoint is not dual_system/joint_self_attn/80D; explicit architecture adaptation is required')
    weights = Path(find_latest_weights(str(directory))) if p.is_dir() else p
    if not weights.is_file() or weights.suffix != '.safetensors':
        raise ValueError('Selected checkpoint has no safetensors weights')
    return str(weights)


def make_config(plan, snapshot, checkpoint):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    accum = validate_plan(plan)
    with initialize_config_dir(config_dir=str(Path(plan['openwam_root'])/'configs'), version_base=None):
        cfg = compose(config_name='train')
    # Use the selected model's exact architecture/component settings. Dataset,
    # schedule and objective remain those of this clean-only experiment.
    ckpt_cfg = OmegaConf.load(Path(checkpoint).parent/'config.yaml')
    cfg.model = ckpt_cfg.model
    cfg.dataloader.dataset_dir = str(snapshot)
    cfg.dataloader.variant = 'clean_50'
    cfg.dataloader.action_mode = 'eef'
    cfg.dataloader.unify_action = True
    cfg.dataloader.normalize_mode = 'min-max'
    cfg.dataloader.num_frames = 33
    cfg.dataloader.window_stride = 1
    cfg.dataloader.video_stride = 4
    cfg.training.num_epochs = 5
    cfg.training.batch_size = plan['micro_batch']
    cfg.training.gradient_accumulation_steps = accum
    cfg.training.learning_rate = plan.get('learning_rate', 1e-4)
    cfg.training.constraints.weight = plan.get('constraint_weight', .1)
    cfg.training.constraints.execution_horizon = None
    cfg.training.constraints.require_full_coverage = True
    cfg.training.finetune_ckpt_path = checkpoint
    cfg.training.output_path = str(Path(plan['run_root'])/'training')
    cfg.training.save_steps = plan.get('save_steps', 2000)
    cfg.training.save_full_states_for_resume = True
    cfg.training.keep_last_k_ckpts = 2
    cfg.training.metrics_jsonl = True
    cfg.project.wandb = None
    return cfg


def prepare_data(config_path, output):
    """Actual reader + fresh shared stats + objective on normalized/unified samples."""
    import h5py
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.registry import build_dataset
    from openwam.train.constraint_supervision import ActionDecoder, build_constraint_objective
    torch.set_num_threads(2)
    cfg = OmegaConf.load(config_path)
    ds = build_dataset(cfg.dataloader, split='train')
    obj = build_constraint_objective(cfg, ds)
    samples = 0
    max_reference_loss = 0.
    gradients = []
    for reader in ds._sub_datasets:
        decoder = ActionDecoder.from_reader(reader)
        for ep in sorted({0, len(reader._episode_files)//2, len(reader._episode_files)-1}):
            length = reader._episode_lengths[ep]
            bundle = reader._constraint_bundles[str(Path(reader._episode_files[ep]).resolve())]
            events = [c['activation']['event']['frame']-1 for c in bundle['constraints']
                      if c['activation']['mode'] == 'reference_event']
            for start in sorted({0, length-2, *events}):
                sample = reader._build_sample(ep, start)
                with h5py.File(reader._episode_files[ep]) as f:
                    raw = reader._read_eef_actions(f, start, min(length, start+33))
                decoded = decoder.decode(torch.cat((sample['proprio'], sample['action']))).numpy()[:len(raw)]
                np.testing.assert_allclose(decoded, raw, atol=2e-6, rtol=2e-6)
                prediction = sample['action'].clone().requires_grad_()
                value = obj.bind([sample])(prediction[None])
                loss = value['loss']
                if not torch.isfinite(loss) or float(loss.detach()) > 1e-7 or value['metrics']['constraint_count'] <= 0:
                    raise ValueError('Reference has nonzero loss or no active constraints')
                loss.backward()
                if not torch.isfinite(prediction.grad).all():
                    raise ValueError('Non-finite reference gradient')
                max_reference_loss = max(max_reference_loss, float(loss.detach()))
                samples += 1
                if start == 0 and ep == 0:
                    prediction = sample['action'].clone()
                    # Perturb one EEF coordinate by a physical 10 cm, mapped
                    # through the exact affine scale and 80D scatter.
                    prediction[0, decoder.indices[0]] += .1*float(decoder.divisor[0])
                    prediction.requires_grad_()
                    bad = obj.bind([sample])(prediction[None])['loss']
                    bad.backward()
                    if not torch.isfinite(bad) or not torch.isfinite(prediction.grad).all():
                        raise ValueError('Non-finite violation gradient')
                    gradients.append(float(prediction.grad.abs().sum()))
    if not any(g > 0 for g in gradients):
        raise ValueError('Perturbations produced no constraint gradient')
    report = {'passed': True, 'tasks': len(ds._sub_datasets), 'windows_per_epoch': len(ds),
              'samples_checked': samples, 'max_reference_loss': max_reference_loss,
              'nonzero_perturbation_gradients': sum(g > 0 for g in gradients),
              'global_batch': 256, 'epochs': 5, 'objective': obj.provenance()}
    atomic_json(output, report)


def copy_snapshot(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if not (target/'snapshot_complete.json').exists():
            raise ValueError('Incomplete snapshot exists; inspect before retrying')
        return
    target.mkdir(parents=True)
    inventory = []
    for path in sorted(source.glob('*/aloha-agilex_clean_50/data/episode*.hdf5')):
        before = path.stat()
        relative = path.relative_to(source)
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)  # Separate inode; subsequent label revisions cannot change the run.
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Source changed while freezing dataset')
        h = hashlib.sha256()
        with dest.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        inventory.append({'path': str(relative), 'sha256': h.hexdigest(), 'bytes': dest.stat().st_size})
    shutil.copy2(source/'batch_manifest.json', target/'batch_manifest.json')
    atomic_json(target/'snapshot_complete.json', {'source': str(source), 'files': inventory})


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=30)
    return [line for line in result.stdout.splitlines() if line.strip()]


def check_smoke(run, accum, *, resumed=False):
    expected_step = accum*(2 if resumed else 1)
    meta = json.loads((run/f'accel_state_step_{expected_step}/trainer_state.json').read_text())
    if meta['global_step'] != expected_step or meta['opt_step'] != (2 if resumed else 1):
        raise ValueError('Incorrect optimizer/accumulation progress in real smoke run')
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != ([accum, accum*2] if resumed else [accum]):
        raise ValueError('Resume duplicated or skipped optimizer updates')
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('loss_total', 'loss_action', 'loss_constraint', 'grad_norm')):
            raise ValueError('Non-finite real smoke metrics')
        if row['grad_norm'] <= 0 or row['constraint_metrics']['constraint_count'] <= 0:
            raise ValueError('No gradients or constraints in real smoke run')
    return rows


def weights_changed(before, after):
    from safetensors import safe_open
    import torch
    with safe_open(before, framework='pt', device='cpu') as a, safe_open(after, framework='pt', device='cpu') as b:
        names = [n for n in a.keys() if 'action_backbone' in n and n in b.keys()]
        for name in names:
            av, bv = a.get_slice(name), b.get_slice(name)
            shape = av.get_shape()
            if shape != bv.get_shape():
                raise ValueError('Checkpoint tensor shape changed unexpectedly')
            slices = tuple(slice(0, min(4, n)) for n in shape)
            if shape and not torch.equal(av[slices], bv[slices]):
                return True
    return False


def run_gate(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan['run_root']); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'gate.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root/'state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('phase') in (
            'failed', 'training', 'complete', 'smoke', 'resume_smoke'):
        raise ValueError('Previous run requires inspection; refusing duplicate launch')

    def state(phase, **extra):
        atomic_json(state_path, dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **extra))

    def execute(args, log, *, cuda=''):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   WANDB_MODE='disabled', HYDRA_FULL_ERROR='1', NCCL_DEBUG='WARN', PYTHONUNBUFFERED='1',
                   USE_TF='0', USE_FLAX='0')
        with (root/log).open('a') as stream:
            subprocess.run(args, cwd=plan['openwam_root'], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        while True:
            plan = json.loads(plan_path.read_text())
            accum = validate_plan(plan)
            checkpoint = selected_checkpoint(plan)
            published = publication_status(plan['dataset_root'])
            state('waiting', checkpoint=checkpoint, checkpoint_required=checkpoint is None, publication=published)
            if checkpoint and published['ready']:
                break
            time.sleep(30)
        atomic_json(root/'resolved_plan.json', dict(plan, initial_checkpoint=checkpoint, gradient_accumulation_steps=accum))
        python = plan['python']
        state('freeze')
        snapshot = root/'dataset'
        copy_snapshot(Path(plan['dataset_root']), snapshot)
        state('audit')
        execute([python, 'scripts/audit_robotwin_constraints.py', '--root', str(snapshot), '--rax-root', plan['rax_root'],
                 '--output', str(root/'final_audit.json'), '--workers', '4', '--all-images', '--require-complete'], 'audit.log')
        from omegaconf import OmegaConf
        cfg = make_config(plan, snapshot, checkpoint)
        OmegaConf.save(cfg, root/'train.yaml')
        execute([python, 'scripts/robotwin_training_gate.py', '--prepare-data', str(root/'train.yaml'),
                 '--report', str(root/'reader_audit.json')], 'reader_audit.log')
        while True:
            busy = gpu_processes()
            if not busy:
                break
            state('waiting_for_gpus', processes=busy)
            time.sleep(30)
        cuda = ','.join(map(str, plan['gpus']))
        launch = training_command(plan, root)
        smoke = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        smoke.training.output_path = str(root/'smoke')
        smoke.training.max_steps = accum
        smoke.training.save_steps = accum
        smoke.training.keep_resume_state_on_finish = True
        smoke.training.lr_scheduler = None
        OmegaConf.save(smoke, root/'smoke.yaml')
        state('smoke')
        execute(launch+['--config-name=smoke'], 'smoke.log', cuda=cuda)
        runs = [p.parent for p in (root/'smoke').glob('*/config.yaml')]
        if len(runs) != 1:
            raise ValueError('Expected exactly one smoke run')
        run = runs[0]
        check_smoke(run, accum)
        if not weights_changed(checkpoint, str(run/f'checkpoint_step_{accum}.safetensors')):
            raise ValueError('Actual action model weights did not update')
        smoke.training.finetune_ckpt_path = None
        smoke.training.resume_ckpt_path = str(run)
        smoke.training.max_steps = accum*2
        OmegaConf.save(smoke, root/'resume_smoke.yaml')
        state('resume_smoke')
        execute(launch+['--config-name=resume_smoke'], 'resume_smoke.log', cuda=cuda)
        metrics = check_smoke(run, accum, resumed=True)
        if not weights_changed(str(run/f'checkpoint_step_{accum}.safetensors'), str(run/f'checkpoint_step_{accum*2}.safetensors')):
            raise ValueError('Resumed action model weights did not update')
        atomic_json(root/'smoke_passed.json', {'passed': True, 'metrics': metrics, 'world_size': len(plan['gpus']),
                                             'global_batch': 256, 'resume_tested': True})
        state('training', config=str(root/'train.yaml'), epochs=5, global_batch=256,
              world_size=len(plan['gpus']), micro_batch=plan['micro_batch'], gradient_accumulation_steps=accum)
        execute(launch+['--config-name=train'], 'training.log', cuda=cuda)
        runs = [p.parent for p in (root/'training').glob('*/config.yaml')]
        if len(runs) != 1 or not list(runs[0].glob('checkpoint_step_*.safetensors')):
            raise ValueError('Training returned without its checkpoint')
        rows = [json.loads(line) for line in (runs[0]/'metrics.jsonl').read_text().splitlines()]
        windows = json.loads((root/'reader_audit.json').read_text())['windows_per_epoch']
        expected_steps = expected_training_steps(windows, plan)
        if not rows or rows[-1]['step'] != expected_steps or rows[-1]['epoch'] != 4:
            raise ValueError('Training did not complete the requested five full epochs')
        state('complete', run=str(runs[0]))
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan')
    p.add_argument('--prepare-data')
    p.add_argument('--report')
    args = p.parse_args()
    if args.prepare_data:
        prepare_data(args.prepare_data, args.report)
    elif args.plan:
        run_gate(args.plan)
    else:
        p.error('Provide --plan or --prepare-data with --report')
