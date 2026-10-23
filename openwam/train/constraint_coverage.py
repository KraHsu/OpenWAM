"""Temporal coverage and CPU chunk selection; shared verbatim with OpenWAM.

Full coverage means every predicted frame of each arm has a position/motion
constraint usable in any chunk containing that frame. It does not mean every
degree of freedom is fixed. Rotation and gripper coverage are reported separately.
"""
import math

POSITION = {'axis_position_interval', 'radial_distance_interval', 'point_distance_interval', 'motion_direction_cone'}


def active_range(c):
    a = c['activation']
    if a['mode'] == 'endpoint': return a['end']-1, a['end']
    if a['mode'] == 'reference_event': return a['event']['frame'], a['event']['frame']+1
    return a['start'], a['end']


def chunk_candidate(c, start, horizon):
    """Fast reference-only prefilter; residual() still checks dimension masks."""
    lo, hi = active_range(c)
    if hi <= start+1 or lo > start+horizon: return False
    if (c['type'] in ('axis_displacement_interval', 'angular_displacement_interval')
            and c['parameters'].get('anchor', 'predicted_start') == 'predicted_start'):
        if not start <= c['activation']['start'] <= start+horizon: return False
    return True


def coverage_masks(constraints, length):
    masks = {arm: {kind: [False]*length for kind in ('position_or_motion', 'orientation', 'gripper')}
             for arm in ('left', 'right')}
    for c in constraints:
        if c['evidence']['status'] != 'accepted': continue
        kind = c['type']; bounds = c['parameters']['bounds']
        # Predicted-start displacement origins may be outside a short chunk.
        fixed = c['parameters'].get('anchor') == 'reference_frame'
        if kind in POSITION or (kind == 'axis_displacement_interval' and fixed): group = 'position_or_motion'
        elif (kind == 'axis_angle_interval' or (kind == 'angular_displacement_interval' and fixed)) and bounds != [0, math.pi]: group = 'orientation'
        elif kind == 'gripper_interval' and bounds != [0, 1]: group = 'gripper'
        else: continue
        lo, hi = active_range(c)
        arm = c['subject'].split('_')[0]
        masks[arm][group][max(0, lo):min(length, hi)] = [True]*max(0, min(length, hi)-max(0, lo))
    return masks


def runs(mask, *, value=False, start=1):
    begin = None
    for i in range(start, len(mask)+1):
        if i < len(mask) and mask[i] == value:
            if begin is None: begin = i
        elif begin is not None:
            yield begin, i
            begin = None


def coverage_report(constraints, length):
    if length < 2: raise ValueError('Coverage requires an initial state and at least one action')
    masks = coverage_masks(constraints, length); per_arm = {}
    for arm, groups in masks.items():
        per_arm[arm] = {kind: {'covered': sum(mask[1:]), 'total': length-1,
                              'ratio': sum(mask[1:])/(length-1), 'gaps': list(runs(mask))}
                        for kind, mask in groups.items()}
    covered = sum(all(masks[a]['position_or_motion'][i] for a in masks) for i in range(1, length))
    return {'version': 1, 'scope': 'position_or_motion_each_arm_each_action_frame',
            'initial_state_frame': 0, 'total': length-1, 'covered': covered, 'ratio': covered/(length-1),
            'complete': covered == length-1, 'chunk_safe': True, 'per_arm': per_arm}
