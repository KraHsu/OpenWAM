"""Portable embedded-data contract, shared verbatim with RAX. Standard library only."""
import hashlib
import json
import math

SCHEMA_VERSION = '1.6'
HDF5_KEY = 'trajectory_constraints/json'
LOCAL_POINT_KINDS = ('axis_position_interval', 'radial_distance_interval',
                     'point_distance_interval', 'axis_displacement_interval')
KINDS = ('axis_position_interval', 'radial_distance_interval', 'point_distance_interval',
         'axis_angle_interval', 'axis_displacement_interval', 'angular_displacement_interval', 'gripper_interval',
         'motion_direction_cone', 'window_direction_cone', 'gripper_pair_max_interval')


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


def validate_local_point(point, kind):
    """A rigid marker in the decoded physical subject EEF frame, in metres."""
    if kind not in LOCAL_POINT_KINDS:
        raise ValueError('local_point is only supported by position/distance primitives')
    keys(point, ('frame', 'offset_m', 'provenance'))
    if point['frame'] != 'subject_eef':
        raise ValueError('local_point must use the decoded physical subject_eef frame')
    vector(point['offset_m'], 3)
    provenance = point['provenance']
    keys(provenance, ('source', 'reference', 'sha256', 'rationale'))
    if provenance['source'] not in ('known_geometry', 'manual'):
        raise ValueError('A fixed local point requires known geometry or explicit manual provenance')
    digest = provenance['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or set(digest)-set('0123456789abcdef'):
        raise ValueError('Local point provenance needs a lowercase SHA256 source identity')
    for key in ('reference', 'rationale'):
        if not isinstance(provenance[key], str) or not provenance[key].strip():
            raise ValueError('Local point source reference and geometric rationale required')
    return point


def _normalized_quaternion(value):
    """Physical WXYZ, matching reader normalization, with a canonical sign."""
    vector(value, 4)
    q = [float(x) for x in value]
    norm = math.sqrt(sum(x*x for x in q))
    if not math.isclose(norm, 1., abs_tol=1e-4, rel_tol=1e-5):
        raise ValueError('Angular-center reference quaternion is not measured unit WXYZ')
    q = [x/norm for x in q]
    sign = next((1 if x > 0 else -1 for x in q if x != 0), 1)
    return [x*sign for x in q]


def _rotate_vector(q, v, transpose=False):
    w,x,y,z = _normalized_quaternion(q)
    matrix = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    if transpose:
        matrix = list(zip(*matrix))
    return [sum(a*b for a,b in zip(row,v)) for row in matrix]


def _hint_payload(c, source):
    return dict(subject=c['subject'], reference_frame=c['reference_frame'],
                local_axis=c['parameters']['local_axis'], reference_axis=c['parameters']['reference_axis'],
                source={k:v for k,v in source.items() if k != 'sha256'})


def _project_reference_axis(c, quaternion):
    p = c['parameters']
    axis = _rotate_vector(quaternion, p['local_axis'])
    axis = _rotate_vector(c['reference_frame']['quaternion_wxyz'], axis, transpose=True)
    target = p['reference_axis']; dot = sum(a*b for a,b in zip(axis,target))/sum(v*v for v in target)
    tangent = [a-dot*b for a,b in zip(axis,target)]
    norm = math.sqrt(sum(x*x for x in tangent))
    # This is an annotation validity threshold, NOT a predicted near-center
    # region with a substituted derivative. Never guess a world direction.
    if norm <= 1e-8:
        raise ValueError('Reference axis cannot determine a nondegenerate angular-center tangent')
    return [x/norm for x in tangent]


def make_center_tangent_hint(c, quaternion_wxyz, recording_id, frame):
    """Fixed annotation only. Does not inspect a predicted chunk or alter bounds."""
    q = [float(x) for x in quaternion_wxyz]
    _normalized_quaternion(q)  # Validate; retain raw measured values in source identity.
    sign = next((1 if x > 0 else -1 for x in q if x != 0), 1)
    source = dict(kind='reference_axis_projection', recording_id=recording_id, frame=frame,
                  quaternion_wxyz=[x*sign for x in q])
    source['sha256'] = hashlib.sha256(json.dumps(_hint_payload(c,source),sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    hint = dict(version=1, frame='reference', direction=_project_reference_axis(c,source['quaternion_wxyz']),source=source)
    return hint


def validate_center_tangent_hint(c, *, length=None):
    if c['type'] != 'axis_angle_interval':
        raise ValueError('Angular-center tangent is only defined for axis_angle_interval')
    hint = c['parameters']['center_tangent_hint']
    keys(hint, ('version','frame','direction','source'))
    if type(hint['version']) is not int or hint['version'] != 1 or hint['frame'] != 'reference':
        raise ValueError('Unsupported fixed angular-center hint contract')
    vector(hint['direction'],3,unit=True)
    source = hint['source']; keys(source, ('kind','recording_id','frame','quaternion_wxyz','sha256'))
    selected = c['activation'].get('event',{}).get('frame',c['activation']['start'])
    if (source['kind'] != 'reference_axis_projection' or type(source['frame']) is not int
            or source['frame'] != selected or source['frame'] < 0
            or (length is not None and source['frame'] >= length)
            or not isinstance(source['recording_id'],str) or not source['recording_id'].strip()):
        raise ValueError('Angular-center hint needs the fixed reference selection-start frame')
    _normalized_quaternion(source['quaternion_wxyz'])
    expected = make_center_tangent_hint(c,source['quaternion_wxyz'],source['recording_id'],source['frame'])
    # Quantized directions must not silently select an unrelated exit branch.
    if (not isinstance(source['sha256'],str) or source['sha256'] != expected['source']['sha256']
            or max(abs(a-b) for a,b in zip(hint['direction'],expected['direction'])) > 1e-8):
        raise ValueError('Angular-center tangent/source hash differs from the fixed reference projection')
    return hint


def validate_reference_center_hints(bundle, handle):
    """Bind a hint to full-episode HDF5 before chunks discard its source frame."""
    for c in bundle['constraints']:
        hint = c['parameters'].get('center_tangent_hint')
        if hint is None:
            continue
        arm = {'left_eef':'left','right_eef':'right'}[c['subject']]
        q = [float(v) for v in handle[f'endpose/{arm}_endpose'][hint['source']['frame']][3:]]
        if bundle['action_contract']['quaternion_order'] == 'xyzw':
            q = [q[3],*q[:3]]
        expected = make_center_tangent_hint(c,q,bundle['provenance']['recording_id'],hint['source']['frame'])
        if hint != expected:
            raise ValueError(f'Angular-center hint {c["id"]} does not match its HDF5 reference orientation')


def validate_constraint(c, length=None):
    keys(c, ('id', 'type', 'subject', 'reference_frame', 'parameters', 'activation', 'penalty', 'evidence'))
    if not isinstance(c['id'], str) or not c['id'].strip():
        raise ValueError('Constraint needs a nonempty id')
    kind = c['type']
    subjects = ('both_grippers',) if kind == 'gripper_pair_max_interval' else ('left_eef', 'right_eef')
    if kind not in KINDS or c['subject'] not in subjects:
        raise ValueError('Unsupported base constraint or subject; no object/contact inference')
    frame = c['reference_frame']
    keys(frame, ('origin_m', 'quaternion_wxyz'))
    vector(frame['origin_m'], 3)
    vector(frame['quaternion_wxyz'], 4, unit=True)
    if kind == 'gripper_pair_max_interval' and frame != {'origin_m': [0, 0, 0], 'quaternion_wxyz': [1, 0, 0, 0]}:
        raise ValueError('Joint gripper fractions use the canonical identity frame')
    p = c['parameters']
    required = {'bounds'}
    if kind in ('axis_position_interval', 'radial_distance_interval', 'axis_displacement_interval'):
        required.add('axis')
    if kind == 'axis_angle_interval':
        required.update(('local_axis', 'reference_axis'))
    if kind in ('motion_direction_cone', 'window_direction_cone'):
        required.update(('axis', 'half_angle_rad'))
    if kind == 'window_direction_cone':
        required.update(('window_steps', 'minimum_window_steps', 'minimum_reference_progress_m', 'fallback_radius_m'))
    displacement = kind in ('axis_displacement_interval', 'angular_displacement_interval')
    optional = ((('anchor',) if displacement else ()) + (('local_point',) if kind in LOCAL_POINT_KINDS else ())
                + (('center_tangent_hint',) if kind == 'axis_angle_interval' else ()))
    keys(p, required, optional)
    if 'local_point' in p:
        validate_local_point(p['local_point'], kind)
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
    if kind in ('gripper_interval', 'gripper_pair_max_interval') and not 0 <= lo <= hi <= 1:
        raise ValueError('Opening bounds must lie in [0,1], where 1=open')
    if kind in ('motion_direction_cone', 'window_direction_cone'):
        if lo != 0 or hi < 0 or not 0 < number(p['half_angle_rad']) < math.pi/2:
            raise ValueError('Direction cone needs [0, slack_m] and half angle in (0,pi/2)')
    if kind == 'window_direction_cone':
        if (type(p['window_steps']) is not int or type(p['minimum_window_steps']) is not int
                or not 2 <= p['minimum_window_steps'] <= p['window_steps'] <= 128):
            raise ValueError('Window direction requires 2 <= minimum_window_steps <= window_steps <= 128')
        number(p['minimum_reference_progress_m'], positive=True)
        number(p['fallback_radius_m'], positive=True)
    activation = c['activation']
    keys(activation, ('mode', 'start', 'end'), ('event',))
    start, end = activation['start'], activation['end']
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('Activation uses integer observation indices [start,end)')
    if length is not None and end > length:
        raise ValueError('Activation extends beyond the recorded episode')
    if activation['mode'] not in ('interval', 'endpoint', 'reference_event'):
        raise ValueError('Unknown activation mode')
    if kind == 'gripper_pair_max_interval' and activation['mode'] != 'interval':
        raise ValueError('Joint gripper constraints use fixed reference intervals')
    if kind in ('motion_direction_cone', 'window_direction_cone') and (start < 1 or activation['mode'] != 'interval'):
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
    if 'center_tangent_hint' in p:
        validate_center_tangent_hint(c, length=length)
    return c


def validate_bundle(bundle, length=None):
    keys(bundle, ('schema_version', 'action_contract', 'provenance', 'constraints'))
    if bundle['schema_version'] not in ('1.0', '1.1', '1.2', '1.3', '1.4', '1.5', SCHEMA_VERSION):
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
        if 'center_tangent_hint' in c['parameters']:
            if bundle['schema_version'] != SCHEMA_VERSION:
                raise ValueError('Fixed angular-center hints require schema 1.6')
        if 'local_point' in c['parameters'] and bundle['schema_version'] not in ('1.5', SCHEMA_VERSION):
            raise ValueError('Rigid local point constraints require schema 1.5')
        if 'anchor' in c['parameters'] and bundle['schema_version'] not in ('1.2', '1.3', '1.4', '1.5', SCHEMA_VERSION):
            raise ValueError('Explicit displacement anchors require schema 1.2')
        if c['type'] == 'motion_direction_cone' and bundle['schema_version'] == '1.0':
            raise ValueError('Direction cones require schema 1.1')
        if c['type'] == 'window_direction_cone' and bundle['schema_version'] not in ('1.3', '1.4', '1.5', SCHEMA_VERSION):
            raise ValueError('Window direction cones require schema 1.3')
        if c['type'] == 'gripper_pair_max_interval' and bundle['schema_version'] not in ('1.4', '1.5', SCHEMA_VERSION):
            raise ValueError('Joint gripper max constraints require schema 1.4')
        validate_constraint(c, length)
        if ('center_tangent_hint' in c['parameters'] and
                c['parameters']['center_tangent_hint']['source']['recording_id'] != provenance['recording_id']):
            raise ValueError('Angular-center hint belongs to a different reference recording')
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
    validate_reference_center_hints(bundle, handle)
    return bundle


def validate_reference_events(bundle, handle):
    for c in bundle['constraints']:
        a = c['activation']
        if c['evidence']['status'] != 'accepted' or a['mode'] != 'reference_event':
            continue
        e = a['event']; frame = e['frame']
        if c['subject'] not in ('left_eef', 'right_eef'):
            raise ValueError('Reference gripper events require one explicit arm')
        arm = {'left_eef': 'left', 'right_eef': 'right'}[c['subject']]
        gripper = handle[f'endpose/{arm}_gripper']
        before, after = float(gripper[frame-1]), float(gripper[frame])
        direction = 1 if e['direction'] == 'increasing' else -1
        if not (math.isfinite(before) and math.isfinite(after)
                and direction*(before-e['threshold']) < 0 <= direction*(after-e['threshold'])):
            raise ValueError(f"Accepted constraint {c['id']} has no annotated reference threshold crossing")
