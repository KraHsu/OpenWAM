#!/usr/bin/env python3
"""Capture a genuine closed-loop policy trace for fixed-reference red-team search.

The reference supplies only the frozen scene, horizon and constraint labels.
Each step submits a fresh observation; the server may consume its cached action
chunk. No trajectory resampling, expert suffix, or claim of full native step-limit
evaluation is made.
Run in the native environment against a separately owned OpenWAM policy server.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def native_action(raw):
    from benchmarks.utils.action_conversion import eef20d_to_ee16d
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (20,) or not np.isfinite(raw).all():
        raise ValueError('Policy must return finite physical EEF20 actions')
    # This checkpoint uses the official legacy reader/client rot6d encoding.
    # Its inverse returns the original native WXYZ component ordering, despite
    # the shared helper's XYZW name. Never add a second quaternion permutation.
    row = eef20d_to_ee16d(raw).astype(float)
    for offset in (3, 11):
        if not np.isclose(np.linalg.norm(row[offset:offset+4]), 1., atol=1e-4):
            raise ValueError('Degenerate model rotation; do not invent a valid pose')
    unclipped = row.copy()
    row[[7, 15]] = np.clip(row[[7, 15]], 0., 1.)
    return row, bool(np.any(row != unclipped))


def rows_to_trajectory(rows):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 16:
        raise ValueError('Expected native XYZ+WXYZ+gripper for both arms')
    return {arm: {'pose': rows[:, offset:offset+7].tolist(),
                  'gripper': rows[:, offset+7].tolist()}
            for arm, offset in (('left', 0), ('right', 8))}


def ground_instruction_info(task, native, *, readonly_reader, expert, fingerprint, expected_scene):
    """Prefer audited initial-scene fields; isolate legacy expert fallback."""
    before = fingerprint(native)
    if before != expected_scene:
        raise ValueError('Instruction scene differs from frozen reference')
    metadata = readonly_reader(task, native)
    if metadata is not None:
        if fingerprint(native) != before:
            raise ValueError('Read-only instruction metadata changed the scene')
        return metadata['info'], dict(metadata['provenance'], scene_before=before, scene_after=before)
    info = expert(native)
    if not native.plan_success or not native.check_success():
        raise ValueError('Frozen reference expert cannot ground scene-specific instruction')
    return info['info'], {'adapter': 'isolated_expert_then_scene_reset', 'expert_actions_for_instruction': True,
                          'scene_before': before}


def validate_reference_initialization(task, reference, source):
    """Reject references without fresh-state proof before starting native work."""
    from services.robotwin.success_metadata import (
        metadata_policy, validate_metadata_policy, validate_metadata_receipt,
    )
    policy = source.get('success_metadata_policy')
    validate_metadata_policy(task, policy)
    validate_metadata_receipt(task, reference.get('success_metadata'),
                              reference['provenance']['scene_sha256'], policy=policy)
    if metadata_policy(task) != policy:
        raise ValueError('Native metadata runtime differs from frozen source context')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--rax-root', type=Path, required=True)
    p.add_argument('--checkpoint-receipt', type=Path, required=True)
    p.add_argument('--server-config', type=Path, required=True)
    p.add_argument('--server-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18881)
    args = p.parse_args()
    sys.path.insert(0, str(args.rax_root.resolve()))
    from services.robotwin.acceptance import derived_values_equal, trajectory_violations
    from services.robotwin.acceptance_replay import (
        ALIGNMENT, arrays, command_rows, install_base_frame_adapter, sample, samples_to_trajectory,
    )
    from services.robotwin.collection import initial_scene, install_planner_cache, scene_fingerprint
    from services.robotwin.runtime import initialize_runtime, native_config, source_commit
    from services.robotwin.instruction_metadata import readonly_instruction_metadata
    from services.robotwin.success_metadata import MARKER, initialize_success_metadata, metadata_policy
    from scripts.collect_robotwin_clean50 import play_expert
    from benchmarks.robotwin.openwam2robotwin_interface import ModelClient, _extract_eef_proprio

    ref = json.loads(args.reference.read_text())
    source = json.loads(args.source_manifest.read_text())
    checkpoint = json.loads(args.checkpoint_receipt.read_text())
    server = json.loads(args.server_receipt.read_text())
    if not isinstance(checkpoint.get('sha256'), str) or len(checkpoint['sha256']) != 64:
        raise ValueError('Explicit verified checkpoint digest required')
    if (server.get('checkpoint_sha256') != checkpoint['sha256']
            or server.get('config_sha256') != sha(args.server_config)
            or server.get('port') != args.port):
        raise ValueError('Owned policy server receipt differs from requested model/config')
    task, seed = ref['provenance']['task'], ref['provenance']['seed']
    if task != source['task'] or source_commit() != source['native_commit']:
        raise ValueError('Reference and installed native version mismatch')
    if sha(args.reference) not in {v['sha256'] for v in source['reference_artifacts']}:
        raise ValueError('Reference is outside the frozen source manifest')
    validate_reference_initialization(task, ref, source)
    n = len(arrays(ref['trajectory'])['left']['pose'])
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, created_unix=time.time(), task=task, scene_seed=seed,
                    scene_sha256=ref['provenance']['scene_sha256'], reference_sha256=sha(args.reference),
                    source_manifest_sha256=sha(args.source_manifest), native_commit=source['native_commit'],
                    checkpoint_sha256=checkpoint['sha256'], checkpoint_receipt_sha256=sha(args.checkpoint_receipt),
                    server_config_sha256=sha(args.server_config), collector_sha256=sha(__file__),
                    server_receipt_sha256=sha(args.server_receipt),
                    horizon=n-1, alignment=ALIGNMENT, source_kind='model_closed_loop',
                    action_contract=ref['bundle']['action_contract'],
                    termination='fixed_reference_horizon_native_ever_success',
                    gripper_boundary='clip_to_native_0_1_with_raw_outputs_retained',
                    rotation_encoding='checkpoint_legacy_reader_client_roundtrip_native_wxyz')
    if metadata_policy(task) is not None:
        manifest['success_metadata_policy'] = metadata_policy(task)
    put(output/'configuration.json', manifest)
    put(output/'state.json', dict(phase='instruction_grounding', pid=os.getpid(), updated_unix=time.time()))
    initialize_runtime(); install_planner_cache()
    from generate_episode_instructions import generate_episode_descriptions
    cls = getattr(importlib.import_module('envs.'+task), task)
    kwargs = native_config(task, 0)
    kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                  eval_video_save_dir=None, save_path=str(output/'native_scratch'))
    def factory():
        random.seed(seed)
        native = cls()
        try:
            native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
            if scene_fingerprint(initial_scene(native)) != ref['provenance']['scene_sha256']:
                raise ValueError('Policy scene differs from frozen reference')
            initialize_success_metadata(task, native, expected_scene_sha256=ref['provenance']['scene_sha256'])
            return native
        except BaseException:
            native.close_env(); raise
    native = factory()
    try:
        info, metadata_provenance = ground_instruction_info(
            task, native, readonly_reader=readonly_instruction_metadata, expert=play_expert,
            fingerprint=lambda env: scene_fingerprint(initial_scene(env)), expected_scene=ref['provenance']['scene_sha256'])
        random.seed(seed)
        texts = generate_episode_descriptions(task, [info], 1)[0]['unseen']
        instruction = random.Random(seed).choice(texts)
        put(output/'instruction.json', dict(instruction=instruction, kind='unseen', scene_info=info,
                                           metadata_provenance=metadata_provenance))
    finally:
        native.close_env()
    native = factory()
    policy = None
    if hasattr(native, MARKER):
        manifest['success_metadata'] = getattr(native, MARKER)
    observations, raw_actions, plans, successes, actual = [], [], [], [], []
    rows = command_rows(ref['trajectory'])[:1].tolist()
    error = None
    try:
        adapter = install_base_frame_adapter(native)
        if not derived_values_equal(adapter, ref['replay_adapter']):
            raise ValueError('Policy replay adapter differs from source acceptance')
        native.set_instruction(instruction)
        native.step_lim, native.take_action_cnt, native.eval_success = n-1, 0, False
        actual.append(sample(native))
        for arm in ('left', 'right'):
            measured = actual[0][arm]
            expected = ref['trajectory'][arm]
            if (not np.allclose(measured['pose'][:3], expected['pose'][0][:3], atol=1e-5, rtol=0)
                    or abs(np.dot(measured['pose'][3:], expected['pose'][0][3:])) < 1-1e-6
                    or abs(measured['gripper']-expected['gripper'][0]) > 1e-5):
                raise ValueError('Initial policy state differs from reference')
        original_success = native.check_success
        def predicate():
            successes.append(bool(original_success())); return False
        native.check_success = predicate
        for arm in ('left', 'right'):
            original = getattr(native.robot, arm+'_plan_path')
            def observed(*a, _arm=arm, _original=original, **kw):
                result = _original(*a, **kw)
                plans.append(dict(frame=len(rows), arm=_arm, status=str(result.get('status'))))
                return result
            setattr(native.robot, arm+'_plan_path', observed)
        policy = ModelClient(host=args.host, port=args.port, state_dim=20, action_type='ee', request_timeout=300)
        if policy._client.ping().get('readiness_token') != server['readiness_token']:
            raise ValueError('Connected server is not the owned model process')
        policy._client.predict = policy._client.predict_once
        policy.reset(instruction)
        manifest.update(instruction_sha256=sha(output/'instruction.json'), replay_adapter=adapter)
        put(output/'source_run.json', manifest)
        for frame in range(1, n):
            observation = native.get_obs(); cameras = observation['observation']
            cams = {'head': cameras['head_camera']['rgb'],
                    'left': cameras.get('left_camera', {}).get('rgb'),
                    'right': cameras.get('right_camera', {}).get('rgb')}
            state = _extract_eef_proprio(observation)
            prediction = policy.step(dict(cams=cams, lang=instruction, state=state))
            row, clipped = native_action(prediction)
            observations.append(dict(frame=frame, state=state.tolist(),
                                     camera_sha256={k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                                                    for k, v in cams.items() if v is not None}))
            raw_actions.append(dict(frame=frame, eef20=prediction.tolist(), gripper_clipped=clipped))
            rows.append(row.tolist()); native.take_action(row, action_type='ee')
            if native.take_action_cnt != frame:
                raise ValueError('Policy controller failed to advance one command')
            actual.append(sample(native))
            put(output/'state.json', dict(phase='model_closed_loop', frame=frame, horizon=n-1,
                                          native_ever_success=any(successes), pid=os.getpid(), updated_unix=time.time()))
        final = bool(original_success()); successes.append(final)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if policy is not None: policy._client.close()
        native.close_env()
    complete = len(rows) == len(actual) == n and error is None
    put(output/'raw_policy_actions.json', raw_actions)
    put(output/'observation_hashes.json', observations)
    put(output/'actual.json', samples_to_trajectory(actual))
    result = dict(complete=complete, error=error, native_success=any(successes),
                  planner_success=len(plans)==2*(n-1) and all(v['status']=='Success' for v in plans),
                  planner_results=plans, horizon=n-1)
    if complete:
        trajectory = rows_to_trajectory(rows)
        result['commanded_constraints'] = trajectory_violations(ref['bundle'], ref['trajectory'], trajectory)
        result['actual_constraints'] = trajectory_violations(ref['bundle'], ref['trajectory'], samples_to_trajectory(actual))
        provenance = dict(checkpoint_sha256=checkpoint['sha256'], source_run_sha256=sha(output/'source_run.json'),
                          source_kind='model_closed_loop', full_policy_rollout=True,
                          native_step_limit_evaluation=False, horizon_source='frozen_reference_length',
                          raw_policy_output_in_native_domain=not any(r['gripper_clipped'] for r in raw_actions),
                          applied_action_transform={'gripper_clip_0_1_frames': sum(r['gripper_clipped'] for r in raw_actions),
                                                    'time_resampling': False, 'expert_action_blend': False},
                          raw_policy_actions_sha256=sha(output/'raw_policy_actions.json'),
                          observation_hashes_sha256=sha(output/'observation_hashes.json'))
        candidates = dict(version=1, task=task, scene_seed=seed, scene_sha256=manifest['scene_sha256'],
                          alignment=ALIGNMENT, action_contract=ref['bundle']['action_contract'],
                          candidates=[dict(candidate_id=f'{task}_seed{seed}_model_closed_loop', trajectory=trajectory,
                                           source='model', family='model_closed_loop', model_provenance=provenance)])
        put(output/'model_candidates.json', candidates)
    put(output/'result.json', result)
    put(output/'state.json', dict(phase='complete' if complete else 'failed', error=error, updated_unix=time.time()))
    if not complete: raise RuntimeError(error or 'Incomplete model rollout')
    print(json.dumps({k:v for k,v in result.items() if k not in ('planner_results','commanded_constraints','actual_constraints')}))


if __name__ == '__main__':
    main()
