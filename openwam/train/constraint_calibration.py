"""Explicit, offline calibration helpers for constraint-loss experiments.

These helpers do not choose constraint tolerances or start a training run.
Gradient measurements cover exactly the caller's parameter sequence. They are
not DDP-reduced, and are not full-model gradients for sharded/ZeRO parameters.
Use the same complete, unsharded parameter sequence for meaningful comparisons,
or name the subset explicitly when reporting a subset diagnostic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations

import torch
from torch import Tensor


def timestep_factors(
    sigma: Tensor,
    fm_weight: Tensor,
    *,
    strategy: str = "clean",
    sigma_floor: float = 0.05,
) -> Tensor:
    """Return detached FP32 per-sample factors for clean-action geometry loss.

    ``clean`` leaves the geometric objective unchanged. ``fm_matched`` uses
    ``w / max(sigma, sigma_floor)**2``. Since clean = noisy - sigma * velocity,
    the latter cancels the sigma-squared curvature factor of a quadratic clean
    loss when sigma >= floor, matching the FM *time weighting*. It does not
    match physical-unit scales, feature Jacobians, reductions, or gradient norms.
    At sigma=0 the factor is finite and the clean-to-velocity derivative is zero.
    The floor is an explicit experiment parameter, not a tuned recommendation.
    """
    if strategy not in {"clean", "fm_matched"}:
        raise ValueError("strategy must be 'clean' or 'fm_matched'")
    if isinstance(sigma_floor, bool) or not math.isfinite(sigma_floor) or not 0 < sigma_floor <= 1:
        raise ValueError("sigma_floor must be finite and in (0, 1]")
    if not isinstance(sigma, Tensor) or not isinstance(fm_weight, Tensor):
        raise TypeError("sigma and fm_weight must be tensors")
    if sigma.ndim != 1 or fm_weight.shape != sigma.shape:
        raise ValueError("sigma and fm_weight must have matching per-sample [B] shapes")
    if not sigma.is_floating_point() or not fm_weight.is_floating_point():
        raise TypeError("sigma and fm_weight must be real floating-point tensors")
    if sigma.device != fm_weight.device:
        raise ValueError("sigma and fm_weight must be on the same device")
    sigma, fm_weight = sigma.detach().float(), fm_weight.detach().float()
    if not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
        raise ValueError("sigma must be finite and in [0, 1]")
    if not torch.isfinite(fm_weight).all() or (fm_weight < 0).any():
        raise ValueError("fm_weight must be finite and nonnegative")
    if strategy == "clean":
        return torch.ones_like(sigma)
    factors = fm_weight / sigma.clamp_min(sigma_floor).square()
    if not torch.isfinite(factors).all():
        raise ValueError("time weighting overflowed FP32; increase sigma_floor or reduce fm_weight")
    return factors


def measure_loss_gradients(terms: Mapping[str, Tensor], params: Sequence[Tensor]) -> dict:
    """Measure weighted-term gradients without changing any parameter's ``.grad``.

    Pass terms *after* their training coefficients. Each scalar must belong to
    the same retained computation graph and refer to the same fixed parameters.
    ``autograd.grad(..., retain_graph=True, allow_unused=True)`` leaves ordinary
    ``.grad`` buffers untouched and permits a subsequent training backward.
    Caller-installed backward hooks can still have their own side effects; this
    helper does not suppress hooks or invoke distributed collectives.

    Norms aggregate all entries of the provided sequence, including zero for
    frozen/unused parameters. For each unordered pair, ``norm_ratio`` means
    norm(left)/norm(right). A zero denominator produces None plus a validity
    flag; a cosine involving any zero gradient likewise produces None.
    No epsilon is used to disguise an undefined diagnostic.
    """
    parameters = tuple(params)
    if any(not isinstance(parameter, Tensor) for parameter in parameters):
        raise TypeError("params must be a fixed sequence of tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("params must not contain duplicate tensors")
    if any(not parameter.is_floating_point() for parameter in parameters):
        raise TypeError("params must contain real floating-point tensors")
    active = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients, norms = {}, {}
    for name, term in terms.items():
        if not isinstance(name, str) or not name:
            raise ValueError("loss term names must be nonempty strings")
        if not isinstance(term, Tensor) or term.ndim != 0 or not term.is_floating_point():
            raise TypeError("every weighted loss term must be a real scalar tensor")
        if not bool(torch.isfinite(term)):
            raise ValueError(f"non-finite loss term: {name}")
        values = (
            torch.autograd.grad(term, active, retain_graph=True, allow_unused=True)
            if active and term.requires_grad
            else (None,) * len(active)
        )
        detached = []
        for value in values:
            if value is not None:
                value = value.detach()
                if value.is_sparse:
                    value = value.to_dense()
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"non-finite gradient for loss term: {name}")
            detached.append(value)
        gradients[name] = detached
        norms[name] = math.sqrt(sum(float(g.double().square().sum()) for g in detached if g is not None))
    pairs = []
    for left, right in combinations(terms, 2):
        left_norm, right_norm = norms[left], norms[right]
        cosine_valid = left_norm > 0 and right_norm > 0
        dot = sum(
            float(a.double().mul(b.double()).sum())
            for a, b in zip(gradients[left], gradients[right], strict=True)
            if a is not None and b is not None
        )
        pairs.append(
            dict(
                left=left,
                right=right,
                cosine=max(-1.0, min(1.0, dot / (left_norm * right_norm))) if cosine_valid else None,
                cosine_valid=cosine_valid,
                norm_ratio=left_norm / right_norm if right_norm > 0 else None,
                ratio_valid=right_norm > 0,
            )
        )
    return dict(
        scope="provided_parameters_only",
        parameter_count=len(parameters),
        trainable_parameter_count=len(active),
        norms=norms,
        pairs=pairs,
    )
