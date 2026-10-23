"""Portable embedded-data contract, shared verbatim with RAX. Standard library only."""
import json
import math

SCHEMA_VERSION = '1.2'
HDF5_KEY = 'trajectory_constraints/json'
KINDS = ('axis_position_interval', 'radial_distance_interval', 'point_distance_interval',
         'axis_angle_interval', 'axis_displacement_interval', 'angular_displacement_interval', 'gripper_interval',
         'motion_direction_cone')


def keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value)-set(required)-set(optional) or set(required)-set(value):
        raise ValueError(f'Expected fields {sorted(required)}, optional {sorted(optional)}')


def number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError('Expected a finite number' + (' > 0' if positive else ''))
    return value


def vector(value, size, *, unit=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'Expected length-{size} vector')
    for x in value:
        number(x)
    if unit and not math.isclose(sum(x*x for x in value), 1, abs_tol=1e-5):
        raise ValueError('Axis/quaternion must have unit norm')


def validate_constraint(c, length=None):
    keys(c, ('id', 'type', 'subject', 'reference_frame', 'parameters', 'activation', 'penalty', 'evidence'))
    if not isinstance(c['id'], str) or not c['id'].strip():
        raise ValueError('Constraint needs a nonempty id')
    kind = c['type']
    if kind not in KINDS or c['subject'] not in ('left_eef', 'right_eef'):
        raise ValueError('Unsupported base constraint or subject; no object/contact inference')
    frame = c['reference_frame']
    keys(frame, ('origin_m', 'quaternion_wxyz'))
    vector(frame['origin_m'], 3)
    vector(frame['quaternion_wxyz'], 4, unit=True)
    p = c['parameters']
    required = {'bounds'}
    if kind in ('axis_position_interval', 'radial_distance_interval', 'axis_displacement_interval'):
        required.add('axis')
    if kind == 'axis_angle_interval':
        required.update(('local_axis', 'reference_axis'))
    if kind == 'motion_direction_cone':
        required.update(('axis', 'half_angle_rad'))
    displacement = kind in ('axis_displacement_interval', 'angular_displacement_interval')
    keys(p, required, ('anchor',) if displacement else ())
    if displacement and p.get('anchor', 'predicted_start') not in ('predicted_start', 'reference_frame'):
        raise ValueError('Displacement anchor must be predicted_start or reference_frame')
    for key in ('axis', 'local_axis', 'reference_axis'):
        if key in p:
            vector(p[key], 3, unit=True)
    vector(p['bounds'], 2)
    lo, hi = p['bounds']
    if lo > hi:
        raise ValueError('Bounds must be ordered')
    if kind in ('radial_distance_interval', 'point_distance_interval') and lo < 0:
        raise ValueError('Distance bounds must be nonnegative')
    if kind in ('axis_angle_interval', 'angular_displacement_interval') and not 0 <= lo <= hi <= math.pi:
        raise ValueError('Angular bounds must lie in [0,pi] radians')
    if kind == 'gripper_interval' and not 0 <= lo <= hi <= 1:
        raise ValueError('Opening bounds must lie in [0,1], where 1=open')
    if kind == 'motion_direction_cone':
        if lo != 0 or hi < 0 or not 0 < number(p['half_angle_rad']) < math.pi/2:
            raise ValueError('Direction cone needs [0, slack_m] and half angle in (0,pi/2)')
    activation = c['activation']
    keys(activation, ('mode', 'start', 'end'), ('event',))
    start, end = activation['start'], activation['end']
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('Activation uses integer observation indices [start,end)')
    if length is not None and end > length:
        raise ValueError('Activation extends beyond the recorded episode')
    if activation['mode'] not in ('interval', 'endpoint', 'reference_event'):
        raise ValueError('Unknown activation mode')
    if kind == 'motion_direction_cone' and (start < 1 or activation['mode'] != 'interval'):
        raise ValueError('Direction cones use adjacent positions at interval frames >= 1')
    if (activation['mode'] == 'reference_event') != ('event' in activation):
        raise ValueError('Only reference_event activation requires event metadata')
    if 'event' in activation:
        e = activation['event']
        keys(e, ('frame', 'direction', 'threshold'))
        if type(e['frame']) is not int or not max(1, start) <= e['frame'] < end:
            raise ValueError('Event frame must be inside activation and have a preceding frame')
        if e['direction'] not in ('increasing', 'decreasing') or not 0 < number(e['threshold']) < 1:
            raise ValueError('Reference gripper crossing requires direction and threshold in (0,1)')
    penalty = c['penalty']
    keys(penalty, ('kind', 'scale', 'weight'))
    if penalty['kind'] not in ('squared_hinge', 'quartic_hinge'):
        raise ValueError('Unsupported differentiable penalty')
    number(penalty['scale'], positive=True); number(penalty['weight'], positive=True)
    evidence = c['evidence']
    keys(evidence, ('source', 'status', 'report_id', 'rationale'))
    if evidence['source'] not in ('manual', 'known_geometry', 'demonstration_analysis', 'agent_annotation'):
        raise ValueError('Unknown geometry/annotation source')
    if evidence['status'] not in ('proposed', 'accepted', 'rejected', 'unknown'):
        raise ValueError('Unknown annotation status')
    for key in ('report_id', 'rationale'):
        if not isinstance(evidence[key], str) or not evidence[key].strip():
            raise ValueError('Evidence report and rationale required')
    return c


def validate_bundle(bundle, length=None):
    keys(bundle, ('schema_version', 'action_contract', 'provenance', 'constraints'))
    if bundle['schema_version'] not in ('1.0', '1.1', SCHEMA_VERSION):
        raise ValueError('Unsupported embedded constraint schema')
    contract = bundle['action_contract']
    keys(contract, ('position', 'rotation', 'quaternion_order', 'gripper', 'frame'))
    if (contract['position'], contract['rotation'], contract['gripper'], contract['frame']) != (
            'absolute', 'absolute', 'absolute_open_fraction', 'world'):
        raise ValueError('This RoboTwin reader supports absolute world EEF data only; use a matching adapter for deltas/joints')
    if contract['quaternion_order'] not in ('wxyz', 'xyzw'):
        raise ValueError('Source quaternion order must be explicit')
    provenance = bundle['provenance']
    keys(provenance, ('split', 'variant', 'recording_id', 'source'), ('native_success', 'sampling'))
    if provenance['split'] != 'train' or provenance['variant'] != 'clean_50':
        raise ValueError('C2R constraint data must be train / clean_50')
    for key in ('recording_id', 'source'):
        if not isinstance(provenance[key], str) or not provenance[key].strip():
            raise ValueError('Recording provenance required')
    if 'native_success' in provenance and type(provenance['native_success']) is not bool:
        raise ValueError('native_success must be a boolean')
    if not isinstance(bundle['constraints'], list):
        raise ValueError('constraints must be a list of conjunctive base constraints')
    ids = set()
    for c in bundle['constraints']:
        if 'anchor' in c['parameters'] and bundle['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Explicit displacement anchors require schema 1.2')
        if c['type'] == 'motion_direction_cone' and bundle['schema_version'] == '1.0':
            raise ValueError('Direction cones require schema 1.1')
        validate_constraint(c, length)
        if c['id'] in ids:
            raise ValueError('Duplicate constraint id')
        ids.add(c['id'])
    return bundle


def canonical_json(bundle):
    validate_bundle(bundle)
    return json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_embedded(handle):
    if HDF5_KEY not in handle:
        return None
    value = handle[HDF5_KEY][()]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    bundle = validate_bundle(json.loads(value), len(handle['endpose/left_endpose']))
    validate_reference_events(bundle, handle)
    return bundle


def validate_reference_events(bundle, handle):
    for c in bundle['constraints']:
        a = c['activation']
        if c['evidence']['status'] != 'accepted' or a['mode'] != 'reference_event':
            continue
        e = a['event']; frame = e['frame']
        arm = 'left' if c['subject'] == 'left_eef' else 'right'
        gripper = handle[f'endpose/{arm}_gripper']
        before, after = float(gripper[frame-1]), float(gripper[frame])
        direction = 1 if e['direction'] == 'increasing' else -1
        if not (math.isfinite(before) and math.isfinite(after)
                and direction*(before-e['threshold']) < 0 <= direction*(after-e['threshold'])):
            raise ValueError(f"Accepted constraint {c['id']} has no annotated reference threshold crossing")
