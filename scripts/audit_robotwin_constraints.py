#!/usr/bin/env python3
"""Read-only audit of native clean recordings, using both annotation and training math.

Run on the data host. This establishes label consistency, NOT physical feasibility
of the entire allowed region. --require-complete is the training publication gate.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def audit_episode(job):
    filename, rax_root, all_images = job
    sys.path.insert(0, rax_root)
    import cv2
    import h5py
    import numpy as np
    import torch
    from services.robotwin.annotation import rotation, validate_reference_bounds
    from services.robotwin.collection import scene_fingerprint
    from openwam.dataloader.robotwin import RoboTwinDataset
    from openwam.train.action_trajectory import physical_rotation
    from openwam.train.constraint_schema import read_embedded, canonical_json
    from openwam.train.constraint_coverage import coverage_report, chunk_candidate
    from openwam.train.meta_constraints import residual
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    path = Path(filename)
    before = path.stat()
    with h5py.File(path, 'r') as f:
        bundle = read_embedded(f)
        if bundle is None or bundle['provenance'].get('native_success') is not True:
            raise ValueError('Missing successful clean provenance')
        meta = json.loads(f.attrs['robotwin_expert_provenance'])
        if (meta['task'] != path.parents[2].name or meta['native_success'] is not True
                or meta['collection'] != 0 or meta['task_config'] != 'demo_clean'):
            raise ValueError('Native success/task/clean metadata mismatch')
        if not str(f.attrs.get('instruction', '')).strip():
            raise ValueError('Missing instruction')
        scene = f['initial_scene/json'][()]
        scene_bytes = scene if isinstance(scene, bytes) else scene.encode()
        # Collection fingerprints use canonical JSON, independent of whitespace.
        scene_obj = json.loads(scene_bytes)
        if scene_fingerprint(scene_obj) != meta['initial_scene_sha256']:
            raise ValueError('Initial-scene fingerprint does not match recording')
        scene_digest = hashlib.sha256(json.dumps(scene_obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        length = len(f['endpose/left_endpose'])
        if length < 2:
            raise ValueError('Episode has no action')
        raw = RoboTwinDataset._read_eef_actions(None, f, 0, length)
        if raw.shape != (length, 20) or not np.isfinite(raw).all():
            raise ValueError('Invalid EEF actions')
        joints = np.asarray(f['joint_action/vector'])
        if joints.shape != (length, 14) or not np.isfinite(joints).all():
            raise ValueError('Invalid joint recording')
        convention = ('robotwin_wxyz_read_as_xyzw' if bundle['action_contract']['quaternion_order'] == 'wxyz'
                      else 'physical')
        if convention == 'physical':
            raise ValueError('Native expert audit requires WXYZ recording')
        pred = torch.from_numpy(raw)
        valid = torch.ones_like(pred, dtype=torch.bool)
        for arm, offset in [('left', 0), ('right', 10)]:
            pose = np.asarray(f[f'endpose/{arm}_endpose'])
            grip = np.asarray(f[f'endpose/{arm}_gripper']).reshape(-1)
            if pose.shape != (length, 7) or grip.shape != (length,) or np.any((grip < 0) | (grip > 1)):
                raise ValueError('Invalid pose/gripper shape or range')
            np.testing.assert_allclose(physical_rotation(pred[:, offset+3:offset+9], convention).numpy(),
                                       rotation(pose[:, 3:]), atol=2e-6)
        accepted = [c for c in bundle['constraints'] if c['evidence']['status'] == 'accepted']
        coverage = coverage_report(accepted, length)
        if not coverage['complete']:
            raise ValueError('Incomplete motion coverage')
        validate_reference_bounds(bundle, f)  # Independent NumPy, native WXYZ math.
        max_residual = 0.
        masked_displacements = 0
        for c in accepted:
            v = residual(pred, pred, valid, c, start_frame=0, rotation_convention=convention)
            if v is not None:
                max_residual = max(max_residual, float(v.max()))
                if not torch.isfinite(v).all() or float(v.max()) > 2e-6:
                    raise ValueError(f'Training decoder excludes reference: {c["id"]}')
            a = c['activation']
            target = a.get('event', {}).get('frame', a['end']-1)
            if ('displacement' in c['type'] and target-a['start'] > 32
                    and c['parameters'].get('anchor', 'predicted_start') == 'predicted_start'):
                masked_displacements += 1
                if a['mode'] != 'interval':
                    raise ValueError(f'Constraint can never activate in a 32-action chunk: {c["id"]}')
            # Last-frame and event windows: exact training indexing, including
            # the action whose preceding observation is the chunk initial state.
            if a['mode'] in ('endpoint', 'reference_event') and target > 0:
                start = max(0, target-32)
                end = min(length, start+33)
                if chunk_candidate(c, start, end-start-1):
                    v = residual(pred[start:end], pred[start:end], valid[start:end], c,
                                 start_frame=start, rotation_convention=convention)
                    if v is None or not torch.isfinite(v).all() or float(v.max()) > 2e-6:
                        raise ValueError(f'Endpoint/event alignment failure: {c["id"]}')
        images_checked = 0
        for camera in ('head_camera', 'left_camera', 'right_camera'):
            images = f[f'observation/{camera}/rgb']
            if len(images) != length:
                raise ValueError('Image/action length mismatch')
            indices = range(length) if all_images else sorted({0, length//2, length-1})
            for index in indices:
                data = np.frombuffer(bytes(images[index]), np.uint8)
                decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if decoded is None or decoded.ndim != 3 or decoded.shape[-1] != 3:
                    raise ValueError(f'Invalid JPEG: {camera}/{index}')
                images_checked += 1
        result = {'path': str(path), 'task': meta['task'], 'seed': meta['seed'],
                  'scene_sha256': meta['initial_scene_sha256'], 'canonical_scene_sha256': scene_digest,
                  'recording_id': bundle['provenance']['recording_id'], 'frames': length,
                  'action_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                  'bundle_sha256': hashlib.sha256(canonical_json(bundle).encode()).hexdigest(),
                  'constraints': dict(Counter(c['type'] for c in accepted)),
                  'long_displacements_masked_in_32_action_chunk': masked_displacements,
                  'reference_max_residual': max_residual, 'images_checked': images_checked,
                  'size': before.st_size, 'mtime_ns': before.st_mtime_ns}
    after = path.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError('File changed during audit; rerun on stable publication')
    return result


def checked_episode(job):
    try:
        return {'ok': True, **audit_episode(job)}
    except Exception as exc:
        return {'ok': False, 'path': job[0], 'error': f'{type(exc).__name__}: {exc}'}


def audit(root, rax_root, *, workers=4, all_images=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'batch_manifest.json').read_text())
    tasks, quota = manifest['tasks'], manifest['per_task']
    if manifest['collection'] != 'clean' or len(set(tasks)) != 50 or quota != 50:
        raise ValueError('Expected 50 clean tasks, 50 episodes each')
    paths = sorted(root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5'))
    jobs = [(str(p), str(Path(rax_root).resolve()), all_images) for p in paths]
    entries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(checked_episode, jobs), 1):
            entries.append(result)
            if index % 100 == 0:
                print(json.dumps({'audited': index, 'total': len(paths), 'errors': sum(not e['ok'] for e in entries)}), flush=True)
    errors = [e for e in entries if not e['ok']]
    counts = {}
    for task in tasks:
        rows = [e for e in entries if e['ok'] and e['task'] == task]
        counts[task] = len(rows)
        if len(rows) > quota:
            errors.append({'task': task, 'error': 'Exceeds per-task quota'})
        for key in ('seed', 'scene_sha256', 'canonical_scene_sha256', 'recording_id'):
            if len({r[key] for r in rows}) != len(rows):
                errors.append({'task': task, 'error': 'Duplicate '+key})
    if any(e['ok'] and e['task'] not in tasks for e in entries):
        errors.append({'error': 'Unexpected task directory'})
    complete = not errors and len(paths) == 2500 and all(n == quota for n in counts.values())
    return {'version': 1, 'created_unix': time.time(), 'root': str(root),
            'passed': not errors, 'complete': complete, 'episodes': len(paths), 'per_task': counts,
            'frames': sum(e.get('frames', 0) for e in entries), 'errors': errors,
            'image_validation': 'every frame' if all_images else 'first/middle/last per camera per episode',
            'meaning': 'Reference consistency and decoding/coverage audit; allowed-region physical feasibility is not established.',
            'entries': entries}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--rax-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--all-images', action='store_true')
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    report = audit(args.root, args.rax_root, workers=args.workers, all_images=args.all_images)
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('entries', 'per_task', 'errors')}), flush=True)
    raise SystemExit(0 if report['passed'] and (report['complete'] or not args.require_complete) else 1)


if __name__ == '__main__':
    main()
