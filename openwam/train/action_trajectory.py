"""Differentiable EEF action decoding. No simulator or dynamics dependency.

Inputs here are *physical*, inverse-normalized 20D EEF values. The two arms
each use xyz, two rotation-matrix columns, and continuous gripper opening.
The initial state is absolute even when actions are increments.
"""
from dataclasses import dataclass

import torch


def rotation_6d_to_matrix(value):
    a, b = value[..., :3], value[..., 3:]
    u = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    orthogonal = b - (u * b).sum(-1, keepdim=True) * u
    if torch.any(torch.linalg.vector_norm(a, dim=-1) < 1e-8) or torch.any(
        torch.linalg.vector_norm(orthogonal, dim=-1) < 1e-8
    ):
        raise ValueError("degenerate rotation: cannot evaluate physical orientation")
    v = torch.nn.functional.normalize(orthogonal, dim=-1, eps=1e-8)
    return torch.stack((u, v, torch.linalg.cross(u, v)), dim=-1)


def physical_rotation(value, convention):
    matrix = rotation_6d_to_matrix(value)
    if convention == "physical":
        return matrix
    if convention != "robotwin_wxyz_read_as_xyzw":
        raise ValueError(f"unknown rotation convention: {convention}")
    # Native RoboTwin stores WXYZ. The current OpenWAM reader feeds those
    # four numbers to an XYZW converter. Invert that permutation on SO(3),
    # without changing the checkpoint representation or choosing a quat sign.
    r = matrix
    return torch.stack((
        torch.stack((-r[..., 2, 2], r[..., 1, 2], r[..., 0, 2]), -1),
        torch.stack((r[..., 2, 1], -r[..., 1, 1], -r[..., 0, 1]), -1),
        torch.stack((-r[..., 2, 0], r[..., 1, 0], r[..., 0, 0]), -1),
    ), -2)


@dataclass(frozen=True)
class EEFActionSpec:
    position: str = "absolute"
    rotation: str = "absolute"
    translation_frame: str = "world"
    rotation_frame: str = "world"
    gripper: str = "absolute"
    rotation_convention: str = "physical"

    def __post_init__(self):
        for mode in (self.position, self.rotation, self.gripper):
            if mode not in {"absolute", "step_delta", "start_delta"}:
                raise ValueError("action modes must be absolute, step_delta or start_delta")
        if self.translation_frame not in {"world", "body"} or self.rotation_frame not in {"world", "body"}:
            raise ValueError("delta frames must be world or body")
        if self.rotation_convention not in {"physical", "robotwin_wxyz_read_as_xyzw"}:
            raise ValueError("explicit supported rotation convention required")


def restore_eef_trajectory(actions, initial, spec):
    """Decode one valid, unpadded chunk to absolute EEF targets [T,20].

    Step/body translation uses the previous target orientation. Start/body
    translation uses the chunk's initial orientation. World rotations compose
    on the left; body rotations on the right. This is a command trajectory,
    not a prediction of achieved robot/object motion. Callers must exclude
    any delta-dependent suffix after a missing increment.
    """
    if actions.ndim != 2 or actions.shape[-1] != 20 or initial.shape != (20,):
        raise ValueError("expected actions [T,20] and absolute initial state [20]")
    if not torch.isfinite(actions).all() or not torch.isfinite(initial).all():
        raise ValueError("decode only valid, finite trajectory prefixes")
    if not len(actions):
        return actions.clone()
    arms = []
    for offset in (0, 10):
        start_p, start_g = initial[offset:offset+3], initial[offset+9:offset+10]
        start_r = physical_rotation(initial[offset+3:offset+9], spec.rotation_convention)
        action_r = physical_rotation(actions[:, offset+3:offset+9], spec.rotation_convention)
        previous_p, previous_r, previous_g = start_p, start_r, start_g
        rows = []
        for index, row in enumerate(actions):
            p, r, g = row[offset:offset+3], action_r[index], row[offset+9:offset+10]
            if spec.position != "absolute":
                origin_p, origin_r = ((previous_p, previous_r) if spec.position == "step_delta"
                                      else (start_p, start_r))
                p = origin_p + (origin_r @ p if spec.translation_frame == "body" else p)
            if spec.rotation != "absolute":
                origin_r = previous_r if spec.rotation == "step_delta" else start_r
                r = r @ origin_r if spec.rotation_frame == "world" else origin_r @ r
            if spec.gripper != "absolute":
                g = (previous_g if spec.gripper == "step_delta" else start_g) + g
            rows.append(torch.cat((p, r[:, 0], r[:, 1], g)))
            previous_p, previous_r, previous_g = p, r, g
        arms.append(torch.stack(rows))
    return torch.cat(arms, -1)
