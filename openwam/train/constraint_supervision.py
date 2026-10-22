"""Verified geometric supervision for RoboTwin EEF action windows.

Templates are training targets, never policy inputs. Indices refer to raw HDF5
observation frames: prediction k in a window starting at s targets s + 1 + k.
No simulator, agent, or RAX runtime is imported by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _vector(value, n, name, *, unit=False):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{name} must be a length-{n} list")
    out = tuple(_number(v, name) for v in value)
    if unit and not math.isclose(sum(v * v for v in out), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} must have unit norm")
    return out


def _keys(value, required, optional=(), *, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{name}: missing={sorted(missing)}, unknown={sorted(extra)}")


@dataclass(frozen=True)
class Constraint:
    episode_id: str
    segment_id: str
    start: int
    end: int
    kind: str
    subject: str
    scope: str
    parameters: dict
    evidence: dict
    attachment: dict | None

    @classmethod
    def parse(cls, row):
        _keys(
            row,
            ("schema_version", "episode_id", "segment_id", "interval", "constraint", "evidence", "provenance"),
            ("primitive",),
            name="template",
        )
        if row["schema_version"] != "0.1":
            raise ValueError("unsupported constraint schema_version")
        episode_id = row["episode_id"]
        if not isinstance(episode_id, str) or not episode_id or "\\" in episode_id:
            raise ValueError("episode_id must be a dataset-relative POSIX path")
        path = PurePosixPath(episode_id)
        if path.is_absolute() or ".." in path.parts or str(path) != episode_id or path.suffix != ".hdf5":
            raise ValueError("episode_id must be a canonical dataset-relative .hdf5 path")
        if not isinstance(row["segment_id"], str) or not row["segment_id"]:
            raise ValueError("segment_id must be a nonempty string")
        interval = row["interval"]
        _keys(interval, ("start", "end"), name="interval")
        start, end = interval["start"], interval["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("interval must use integer half-open observation indices [start, end)")
        provenance = row["provenance"]
        if (
            not isinstance(provenance, dict)
            or provenance.get("split") != "train"
            or provenance.get("variant") != "clean_50"
        ):
            raise ValueError("C2R supervision must come from train / clean_50, never evaluation episodes")
        evidence = row["evidence"]
        _keys(evidence, ("source", "verification_status"), ("report_id", "initial_state_id"), name="evidence")
        if evidence["source"] not in {
            "sim_gt",
            "sensor",
            "vision",
            "manual",
            "agent_hypothesis",
            "demonstration_proxy",
        }:
            raise ValueError("unknown evidence source")
        if evidence["verification_status"] not in {"proposed", "accepted", "rejected", "unknown"}:
            raise ValueError("unknown verification_status")
        if evidence["verification_status"] == "accepted" and (
            not isinstance(evidence.get("report_id"), str) or not evidence["report_id"].strip()
        ):
            raise ValueError("accepted constraints require an evidence report_id")
        c = row["constraint"]
        _keys(
            c,
            ("type", "version", "subject", "frame", "parameters", "scope"),
            ("reference", "attachment"),
            name="constraint",
        )
        if c["version"] != "1" or c["frame"] != "world" or c["scope"] not in {"endpoint", "interval"}:
            raise ValueError("v1 supports world-frame endpoint / interval constraints only")
        kind, p = c["type"], c["parameters"]
        if kind == "REACH":
            _keys(p, ("center_m", "radius_m", "scale_m"), name="REACH parameters")
            _vector(p["center_m"], 3, "center_m")
            _number(p["radius_m"], "radius_m", minimum=0)
            _number(p["scale_m"], "scale_m", positive=True)
        elif kind == "CLEARANCE":
            _keys(p, ("safe_height_m", "scale_m"), name="CLEARANCE parameters")
            _number(p["safe_height_m"], "safe_height_m")
            _number(p["scale_m"], "scale_m", positive=True)
        elif kind == "ORIENTATION":
            _keys(p, ("local_axis", "target_axis", "tolerance_rad", "scale_rad"), name="ORIENTATION parameters")
            _vector(p["local_axis"], 3, "local_axis", unit=True)
            _vector(p["target_axis"], 3, "target_axis", unit=True)
            if not 0 <= _number(p["tolerance_rad"], "tolerance_rad") < math.pi:
                raise ValueError("tolerance_rad must be in [0, pi)")
            _number(p["scale_rad"], "scale_rad", positive=True)
        else:
            raise ValueError(f"unsupported constraint type: {kind}")
        subject, attachment = c["subject"], c.get("attachment")
        if subject in {"robot_eef:left", "robot_eef:right"}:
            if attachment is not None or kind == "CLEARANCE":
                raise ValueError("CLEARANCE requires object geometry; an EEF point is not an object envelope")
        elif isinstance(subject, str) and subject.startswith("object:") and len(subject) > 7:
            _keys(
                attachment,
                ("arm", "position_m", "quaternion_wxyz", "verification_status", "report_id", "vertices_m"),
                name="rigid attachment",
            )
            if (
                attachment["arm"] not in {"left", "right"}
                or attachment["verification_status"] != "accepted"
                or not isinstance(attachment["report_id"], str)
                or not attachment["report_id"].strip()
            ):
                raise ValueError("object constraints require a verified rigid attachment for this interval")
            _vector(attachment["position_m"], 3, "attachment.position_m")
            _vector(attachment["quaternion_wxyz"], 4, "attachment.quaternion_wxyz", unit=True)
            vertices = attachment["vertices_m"]
            if not isinstance(vertices, list) or not vertices:
                raise ValueError("object geometry requires nonempty vertices_m")
            for vertex in vertices:
                _vector(vertex, 3, "vertex")
        else:
            raise ValueError("subject must identify robot_eef:left/right or object:<id>")
        return cls(episode_id, row["segment_id"], start, end, kind, subject, c["scope"], p, evidence, attachment)


class ConstraintStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        payload = self.path.read_bytes()
        self.payload = payload
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.by_episode = defaultdict(list)
        self.status_counts = Counter()
        self.source_counts = Counter()
        seen = set()
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                c = Constraint.parse(row)
                key = (c.episode_id, c.segment_id, c.kind, c.subject, c.start, c.end, c.scope)
                if key in seen:
                    raise ValueError("duplicate constraint identity")
                seen.add(key)
                self.status_counts[c.evidence["verification_status"]] += 1
                self.source_counts[c.evidence["source"]] += 1
                if c.evidence["verification_status"] == "accepted":
                    self.by_episode[c.episode_id].append(c)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
        if not self.by_episode:
            raise ValueError("manifest has no accepted constraints")


def rotation_6d_to_matrix(value):
    """OpenWAM convention: two concatenated COLUMNS, not rows."""
    a, b = value[..., :3], value[..., 3:]
    u = F.normalize(a, dim=-1, eps=1e-8)
    orthogonal = b - (u * b).sum(-1, keepdim=True) * u
    if torch.any(torch.linalg.vector_norm(a, dim=-1) < 1e-8) or torch.any(
        torch.linalg.vector_norm(orthogonal, dim=-1) < 1e-8
    ):
        raise ValueError("degenerate predicted rotation: cannot evaluate physical orientation")
    v = F.normalize(orthogonal, dim=-1, eps=1e-8)
    return torch.stack((u, v, torch.linalg.cross(u, v)), dim=-1)


def _quaternion_matrix(q, like):
    w, x, y, z = q
    return like.new_tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def constraint_residual(actions_eef20, constraint):
    """Return nonnegative residuals in meters or radians, with action gradients."""
    c = constraint
    arm = c.attachment["arm"] if c.attachment else c.subject.split(":")[1]
    offset = 0 if arm == "left" else 10
    position = actions_eef20[..., offset : offset + 3]
    rotation = None
    if c.attachment or c.kind == "ORIENTATION":
        rotation = rotation_6d_to_matrix(actions_eef20[..., offset + 3 : offset + 9])
    if c.attachment:
        position = position + (rotation @ position.new_tensor(c.attachment["position_m"])[..., None]).squeeze(-1)
        rotation = rotation @ _quaternion_matrix(c.attachment["quaternion_wxyz"], position)
    p = c.parameters
    if c.kind == "REACH":
        return torch.relu(
            torch.linalg.vector_norm(position - position.new_tensor(p["center_m"]), dim=-1) - p["radius_m"]
        )
    if c.kind == "CLEARANCE":
        vertices = position.new_tensor(c.attachment["vertices_m"])
        world_vertices = position[..., None, :] + torch.einsum("...ij,vj->...vi", rotation, vertices)
        return torch.relu(p["safe_height_m"] - world_vertices[..., 2].amin(dim=-1))
    axis = (rotation @ position.new_tensor(p["local_axis"])[..., None]).squeeze(-1)
    target = position.new_tensor(p["target_axis"]).expand_as(axis)
    sine = torch.linalg.vector_norm(torch.linalg.cross(axis, target), dim=-1)
    cosine = (axis * target).sum(-1)
    angle = torch.atan2(sine, cosine)
    return torch.relu(angle - p["tolerance_rad"])


@dataclass
class ActionDecoder:
    """Inverse of the exact reader affine map; no numpy conversion of predictions."""

    indices: list[int]
    divisor: np.ndarray
    offset: np.ndarray

    @classmethod
    def from_reader(cls, reader):
        if reader.action_mode != "eef":
            raise ValueError("constraint supervision currently requires RoboTwin action_mode=eef")
        if reader.normalize_mode is not None and reader._normalizer is None:
            raise ValueError("normalization requested but no active normalizer; refusing ambiguous physical units")
        indices = list(range(20)) if reader._unify_dst_index is None else reader._unify_dst_index.tolist()
        normalizer = reader._normalizer
        divisor, offset = np.ones(20, dtype=np.float32), np.zeros(20, dtype=np.float32)
        if normalizer is not None:
            if normalizer._scale is None or normalizer._offset is None:
                raise ValueError("constraint supervision requires an invertible affine action normalizer")
            divisor, offset = np.asarray(normalizer._scale), np.asarray(normalizer._offset)
        if len(indices) != 20 or len(set(indices)) != 20 or min(indices) < 0:
            raise ValueError("expected a one-to-one 20D EEF action mapping")
        if (
            divisor.shape != (20,)
            or offset.shape != (20,)
            or not np.isfinite(divisor).all()
            or not np.isfinite(offset).all()
            or np.any(divisor <= 0)
        ):
            raise ValueError("invalid EEF affine normalization parameters")
        return cls(indices, divisor, offset)

    def decode(self, prediction):
        raw = prediction[..., self.indices].float()
        return raw / raw.new_tensor(self.divisor) + raw.new_tensor(self.offset)


class RoboTwinConstraintObjective:
    def __init__(self, dataset, manifest, dataset_root, *, weight, allow_demonstration_proxy=False):
        self.weight = _number(weight, "constraint weight", positive=True)
        self.store = ConstraintStore(manifest)
        self.root = Path(dataset_root).resolve()
        self.episodes = {}
        readers = getattr(dataset, "_sub_datasets", [dataset])
        for reader in readers:
            if reader.variant != "clean_50" or reader.split != "train":
                raise ValueError("C2R constraint training requires clean_50 train readers")
            decoder = ActionDecoder.from_reader(reader)
            for path, length in zip(reader._episode_files, reader._episode_lengths, strict=True):
                path = Path(path).resolve()
                episode_id = path.relative_to(self.root).as_posix()
                constraints = self.store.by_episode.get(episode_id, [])
                for c in constraints:
                    if c.end <= 1:
                        raise ValueError(f"constraint on {episode_id} has no future action frame (frame 0 is proprio)")
                    if c.end > length:
                        raise ValueError(f"constraint interval exceeds {episode_id} length {length}")
                    if c.evidence["source"] == "demonstration_proxy" and not allow_demonstration_proxy:
                        raise ValueError("demonstration proxies require explicit allow_demonstration_proxy=true")
                self.episodes[str(path)] = (decoder, constraints)
        matched = sum(len(c) for _, c in self.episodes.values())
        if matched == 0:
            raise ValueError("no accepted constraints match selected training episodes")
        self.matched_constraints = matched

    def provenance(self):
        return {
            "manifest_sha256": self.store.sha256,
            "matched_constraints": self.matched_constraints,
            "status_counts": dict(self.store.status_counts),
            "source_counts": dict(self.store.source_counts),
            "weight": self.weight,
            "gradient_entry": "single_step_clean_action_estimate",
            "index_convention": "raw_observation_frame_start_plus_1",
        }

    def save_provenance(self, output_path):
        output = Path(output_path)
        target = output / "constraints.jsonl"
        metadata = output / "constraint_provenance.json"
        if target.exists() or metadata.exists():
            validate_constraint_resume(output, self)
            return
        target.write_bytes(self.store.payload)
        metadata.write_text(json.dumps(self.provenance(), indent=2) + "\n", encoding="utf-8")

    def bind(self, batch):
        """Bind training-only labels; the callback runs AFTER the model forward."""
        entries = []
        for sample in batch:
            path = str(Path(sample["episode_path"]).resolve())
            if path not in self.episodes:
                raise ValueError(f"sample is outside the constraint training dataset: {path}")
            decoder, constraints = self.episodes[path]
            entries.append((decoder, constraints, int(sample["start_frame"]) + 1, sample["action_mask"]))

        def evaluate(clean_actions):
            if clean_actions.ndim != 3 or clean_actions.shape[0] != len(entries):
                raise ValueError("expected clean actions [batch, horizon, action_dim]")
            losses, residuals = [], defaultdict(list)
            for prediction, (decoder, constraints, first_frame, mask) in zip(clean_actions, entries, strict=True):
                physical = decoder.decode(prediction)
                mask = torch.as_tensor(mask, device=prediction.device, dtype=torch.bool)
                if mask.shape != prediction.shape:
                    raise ValueError("constraint training requires a full [horizon, action_dim] valid mask")
                raw_mask = mask[:, decoder.indices]
                frames = torch.arange(prediction.shape[0], device=prediction.device) + first_frame
                for c in constraints:
                    active = (frames >= c.start) & (frames < c.end)
                    if c.scope == "endpoint":
                        active &= frames == c.end - 1
                    arm = c.attachment["arm"] if c.attachment else c.subject.split(":")[1]
                    offset = 0 if arm == "left" else 10
                    width = 9 if c.attachment or c.kind == "ORIENTATION" else 3
                    if c.kind == "ORIENTATION" and c.attachment is None:
                        offset, width = offset + 3, 6
                    active &= raw_mask[:, offset : offset + width].all(dim=-1)
                    if not active.any():
                        continue
                    values = physical[active]
                    if not torch.isfinite(values[:, offset : offset + width]).all():
                        raise ValueError("non-finite predicted geometry in a valid constraint interval")
                    violation = constraint_residual(values, c)
                    scale = c.parameters["scale_rad" if c.kind == "ORIENTATION" else "scale_m"]
                    losses.append((violation / scale).square().mean())
                    residuals[c.kind].append(violation.detach())
            # Empty windows stay differentiable without touching padded NaNs.
            loss_sum = torch.stack(losses).sum() if losses else clean_actions.reshape(-1)[:0].sum()
            count = clean_actions.new_tensor(float(len(losses)), dtype=torch.float32)
            denominator = count.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # DDP/ZeRO average gradients across ranks. Compensate so ranks
                # with no annotated windows do not dilute the global mean.
                torch.distributed.all_reduce(denominator)
                world_size = torch.distributed.get_world_size()
            loss = loss_sum * world_size / denominator.clamp_min(1)
            metrics = {"constraint_count": clean_actions.new_tensor(float(len(losses)))}
            for kind in ("REACH", "CLEARANCE", "ORIENTATION"):
                values = torch.cat(residuals[kind]) if residuals[kind] else clean_actions.new_empty(0)
                metrics[f"{kind.lower()}_count"] = values.new_tensor(float(values.numel()))
                metrics[f"{kind.lower()}_residual_sum"] = values.sum()
                metrics[f"{kind.lower()}_residual_max"] = values.max() if values.numel() else values.new_zeros(())
                metrics[f"{kind.lower()}_violation_count"] = (values > 0).float().sum()
            return {"loss": self.weight * loss, "metrics": metrics}

        return evaluate


def validate_constraint_resume(output_path, objective):
    """A different objective starts a new fine-tune run, never silently resumes."""
    output = Path(output_path)
    manifest = output / "constraints.jsonl"
    metadata = output / "constraint_provenance.json"
    if objective is None:
        if manifest.exists() or metadata.exists():
            raise ValueError("cannot disable constraints when resuming a constraint run; use finetune_ckpt_path")
        return
    if not manifest.exists() or not metadata.exists():
        raise ValueError("constraint resume requires archived manifest/provenance; use finetune_ckpt_path")
    if manifest.read_bytes() != objective.store.payload or json.loads(metadata.read_text()) != objective.provenance():
        raise ValueError("constraint objective changed within an existing training run; use finetune_ckpt_path")


def build_constraint_objective(cfg, dataset):
    settings = cfg.training.get("constraints", {})
    weight = _number(settings.get("weight", 0.0), "constraint weight", minimum=0)
    if weight == 0:
        return None
    if cfg.dataloader.type != "robotwin" or cfg.dataloader.variant != "clean_50":
        raise ValueError("constraint supervision requires dataloader=robotwin and variant=clean_50")
    if cfg.model.architecture.framework != "dual_system" or cfg.model.architecture.variant != "joint_self_attn":
        raise ValueError("initial constraint integration supports OpenWAM joint_self_attn only")
    if cfg.training.lambda_action <= 0:
        raise ValueError("retain a positive action FM weight during the initial constraint experiment")
    manifest = settings.get("manifest")
    if not manifest:
        raise ValueError("training.constraints.manifest is required when weight > 0")
    return RoboTwinConstraintObjective(
        dataset,
        manifest,
        cfg.dataloader.dataset_dir,
        weight=weight,
        allow_demonstration_proxy=settings.get("allow_demonstration_proxy", False),
    )


def reduce_constraint_metrics(accelerator, losses):
    """Global sums/maxima; omit means for unobserved types instead of logging zero."""
    raw = losses["constraint_metrics"]
    names = sorted(raw)
    values = torch.stack([raw[name].detach().float() for name in names]).reshape(1, -1)
    if accelerator is not None and accelerator.num_processes > 1:
        values = accelerator.gather(values)
    totals = {
        name: (values[:, i].max() if name.endswith("_max") else values[:, i].sum()).item()
        for i, name in enumerate(names)
    }
    report = {"constraint_count": totals["constraint_count"]}
    for kind in ("reach", "clearance", "orientation"):
        count = totals[f"{kind}_count"]
        report[f"{kind}_count"] = count
        if count:
            report[f"{kind}_residual_mean"] = totals[f"{kind}_residual_sum"] / count
            report[f"{kind}_residual_max"] = totals[f"{kind}_residual_max"]
            report[f"{kind}_violation_rate"] = totals[f"{kind}_violation_count"] / count
    value = losses["constraint"].detach().float().reshape(1)
    if accelerator is not None and accelerator.num_processes > 1:
        value = accelerator.gather(value)
    return {"loss_constraint": value.mean().item(), "constraint_metrics": report}
