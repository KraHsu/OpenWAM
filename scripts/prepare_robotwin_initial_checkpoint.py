#!/usr/bin/env python3
"""Fetch the current official foundation release on the training host and arm its plan."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_robotwin_constraints import atomic_json
from scripts.robotwin_training_gate import selected_checkpoint

REPO = 'OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model'


def prepare(plan_path, model_root):
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    run = Path(plan['run_root'])
    lock = (run/'checkpoint_download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run/'checkpoint_download.json'
    target = Path(model_root).resolve()/REPO.split('/')[-1]
    state = dict(repo_id=REPO, requested_revision='main', path=str(target))

    def status(phase, **extra):
        atomic_json(status_path, dict(state, status=phase, updated_unix=time.time(), **extra))

    try:
        status('metadata')
        metadata = HfApi().model_info(REPO, revision='main', files_metadata=True, token=False)
        # Resolve main once so all files belong to one coherent release.
        state.update(revision=metadata.sha, expected_bytes=sum(s.size or 0 for s in metadata.siblings))
        status('downloading')
        snapshot_download(REPO, revision=metadata.sha, local_dir=str(target), token=False, max_workers=4)
        status('verifying')
        verified = []
        for entry in metadata.siblings:
            path = target/entry.rfilename
            if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
                raise ValueError(f'Download size mismatch: {entry.rfilename}')
            lfs = entry.lfs
            expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
            if expected:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f'Download SHA256 mismatch: {entry.rfilename}')
                verified.append({'file': entry.rfilename, 'sha256': expected})
        weights = selected_checkpoint(dict(plan, initial_checkpoint=str(target)))
        with safe_open(weights, framework='pt', device='cpu') as checkpoint:
            tensors = len(checkpoint.keys())
            if not any('action_backbone' in key for key in checkpoint.keys()):
                raise ValueError('Missing action model weights')
        # Never expose a partial checkpoint to the training gate.
        current = json.loads(plan_path.read_text())
        if current.get('initial_checkpoint') not in (None, weights, str(target)):
            raise ValueError('Plan initialization changed during download; refusing to overwrite')
        current.update(initial_checkpoint=weights, initial_checkpoint_hf=dict(repo_id=REPO, revision=metadata.sha))
        atomic_json(plan_path, current)
        status('ready', weights=weights, tensors=tensors, verified=verified)
        print(json.dumps(dict(status='ready', weights=weights, revision=metadata.sha, tensors=tensors)), flush=True)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--model-root', required=True)
    args = parser.parse_args()
    prepare(args.plan, args.model_root)
