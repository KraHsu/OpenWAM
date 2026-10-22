"""Composable physical constraints; independent of operation names and data layout."""
import torch

from openwam.train.action_trajectory import physical_rotation


def frame_rotation(q, like):
    w, x, y, z = q
    return like.new_tensor([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_angle(r):
    skew = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1)
    return torch.atan2(torch.linalg.vector_norm(skew, dim=-1)*0.5,
                       (r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*0.5)


def residual(pred, reference, valid, constraint, *, start_frame, rotation_convention):
    """Return physical interval violations (m/rad/open fraction), or None.

    The tensors include the measured initial state at row 0. Activation and
    dimensional validity depend solely on data, never on generated actions.
    No multiplication of a valid loss by a zero/NaN padding mask is used.
    """
    c, kind = constraint, constraint['type']
    a, p = c['activation'], c['parameters']
    offset = 0 if c['subject'] == 'left_eef' else 10
    frames = torch.arange(len(pred), device=pred.device) + start_frame
    active = (frames >= a['start']) & (frames < a['end']) & (frames > start_frame)
    if a['mode'] == 'endpoint':
        active &= frames == a['end']-1
    elif a['mode'] == 'reference_event':
        event = a['event']; i = event['frame']-start_frame
        if i < 1 or i >= len(pred) or not valid[i-1:i+1, offset+9].all():
            return None
        g = reference[i-1:i+1, offset+9].detach()
        if not torch.isfinite(g).all():
            raise ValueError('Non-finite reference event')
        sign = 1 if event['direction'] == 'increasing' else -1
        signed = sign*(g-event['threshold'])
        if not bool((signed[0] < 0) & (signed[1] >= 0)):
            return None
        active &= frames == event['frame']
    if kind in ('axis_angle_interval', 'angular_displacement_interval'):
        dims = slice(offset+3, offset+9)
    elif kind == 'gripper_interval':
        dims = slice(offset+9, offset+10)
    else:
        dims = slice(offset, offset+3)
    active &= valid[:, dims].all(-1)
    if not active.any():
        return None
    initial = None
    if kind in ('axis_displacement_interval', 'angular_displacement_interval'):
        i = a['start']-start_frame
        if i < 0 or i >= len(pred) or not valid[i, dims].all():
            return None
        initial = pred[i, dims]
        if not torch.isfinite(initial).all():
            raise ValueError('Non-finite displacement origin')
    values = pred[active, dims]
    if not torch.isfinite(values).all():
        raise ValueError('Non-finite predicted value in an active constraint')
    frame = c['reference_frame']
    basis = frame_rotation(frame['quaternion_wxyz'], values)
    if kind == 'gripper_interval':
        scalar = values[:, 0]
    elif kind == 'axis_angle_interval':
        rotation = physical_rotation(values, rotation_convention)
        axis = (rotation @ values.new_tensor(p['local_axis'])[..., None]).squeeze(-1)
        target = (basis @ values.new_tensor(p['reference_axis'])).expand_as(axis)
        scalar = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(axis, target), dim=-1), (axis*target).sum(-1))
    elif kind == 'angular_displacement_interval':
        rotation = physical_rotation(values, rotation_convention)
        origin = physical_rotation(initial, rotation_convention)
        scalar = rotation_angle(origin.transpose(-1, -2) @ rotation)
    else:
        delta = values - (initial if initial is not None else values.new_tensor(frame['origin_m']))
        if kind == 'point_distance_interval':
            scalar = torch.linalg.vector_norm(delta, dim=-1)
        else:
            axis = basis @ values.new_tensor(p['axis'])
            axial = (delta*axis).sum(-1)
            scalar = (torch.linalg.vector_norm(delta-axial[:, None]*axis, dim=-1)
                      if kind == 'radial_distance_interval' else axial)
    low, high = p['bounds']
    return torch.relu(low-scalar) + torch.relu(scalar-high)


def penalty(violation, spec):
    power = 2 if spec['kind'] == 'squared_hinge' else 4
    return (violation/spec['scale']).pow(power).mean()
