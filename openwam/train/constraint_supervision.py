"""Offline geometric constraints embedded in RoboTwin episodes.

No operation classes, external label manifest, simulator or dynamics model.
The callback receives the differentiable clean estimate, not a velocity field.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from openwam.train.constraint_schema import canonical_json, number, validate_bundle
from openwam.train.meta_constraints import residual, penalty

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
    def __init__(self, dataset, dataset_root, *, weight, execution_horizon=None):
        self.weight = number(weight, positive=True)
        if execution_horizon is not None and (type(execution_horizon) is not int or execution_horizon < 1):
            raise ValueError('execution_horizon must be null or a positive integer')
        self.execution_horizon = execution_horizon
        self.episodes, archived = {}, []
        status_counts, source_counts = Counter(), Counter()
        root = Path(dataset_root).resolve()
        for reader in getattr(dataset, '_sub_datasets', [dataset]):
            if reader.variant != 'clean_50' or reader.split != 'train':
                raise ValueError('C2R constraints require clean_50 train readers')
            decoder = ActionDecoder.from_reader(reader)
            bundles = getattr(reader, '_constraint_bundles', {})
            for path, length in zip(reader._episode_files, reader._episode_lengths, strict=True):
                path = Path(path).resolve()
                bundle = bundles.get(str(path))
                constraints, convention, digest = [], 'physical', None
                if bundle is not None:
                    validate_bundle(bundle, length)
                    digest = hashlib.sha256(canonical_json(bundle).encode()).hexdigest()
                    convention = ('robotwin_wxyz_read_as_xyzw' if bundle['action_contract']['quaternion_order'] == 'wxyz'
                                  else 'physical')
                    for c in bundle['constraints']:
                        status_counts[c['evidence']['status']] += 1
                        source_counts[c['evidence']['source']] += 1
                        if c['evidence']['status'] == 'accepted':
                            constraints.append(c)
                    archived.append({'episode': path.relative_to(root).as_posix(), 'bundle': bundle})
                self.episodes[str(path)] = (decoder, constraints, convention, digest)
        self.matched_constraints = sum(len(entry[1]) for entry in self.episodes.values())
        if not self.matched_constraints:
            raise ValueError('No accepted embedded constraints in the selected training data')
        payload = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in sorted(archived, key=lambda r: r['episode'])).encode()
        self.store = SimpleNamespace(path='embedded:HDF5/trajectory_constraints/json', payload=payload,
                                     sha256=hashlib.sha256(payload).hexdigest(), status_counts=status_counts,
                                     source_counts=source_counts)

    def provenance(self):
        decoders = sorted({json.dumps({'indices': d.indices, 'scale': d.divisor.tolist(), 'offset': d.offset.tolist()},
                                       sort_keys=True) for d, _, _, _ in self.episodes.values()})
        return {'embedded_constraints_sha256': self.store.sha256, 'weight': self.weight,
                'matched_constraints': self.matched_constraints, 'status_counts': dict(self.store.status_counts),
                'source_counts': dict(self.store.source_counts), 'objective_version': 'meta_intervals_v1',
                'execution_horizon': self.execution_horizon, 'gradient_entry': 'single_step_clean_action_estimate',
                'decoder_sha256': hashlib.sha256(json.dumps(decoders).encode()).hexdigest()}

    def save_provenance(self, output_path):
        output = Path(output_path)
        if (output/'constraints.jsonl').exists() or (output/'constraint_provenance.json').exists():
            validate_constraint_resume(output, self)
            return
        output.mkdir(parents=True, exist_ok=True)
        (output/'constraints.jsonl').write_bytes(self.store.payload)
        (output/'constraint_provenance.json').write_text(json.dumps(self.provenance(), indent=2)+'\n')

    def bind(self, batch):
        entries = []
        for sample in batch:
            path = str(Path(sample['episode_path']).resolve())
            if path not in self.episodes:
                raise ValueError('Sample outside selected training episodes')
            entry = self.episodes[path]
            bundle = sample.get('trajectory_constraints')
            digest = hashlib.sha256(canonical_json(bundle).encode()).hexdigest() if bundle is not None else None
            if digest != entry[3]:
                raise ValueError('Embedded constraints changed or disappeared after dataset initialization')
            entries.append((*entry, sample))

        def evaluate(clean_actions):
            if clean_actions.ndim != 3 or clean_actions.shape[0] != len(entries):
                raise ValueError('Expected clean actions [batch,horizon,action_dim]')
            losses, weights, violations = [], [], defaultdict(list)
            from openwam.train.constraint_schema import KINDS
            for prediction, (decoder, constraints, convention, _, sample) in zip(clean_actions, entries, strict=True):
                if not constraints:
                    continue
                reference = torch.as_tensor(sample['action'], device=prediction.device).detach()
                initial = torch.as_tensor(sample['proprio'], device=prediction.device).detach()
                mask = torch.as_tensor(sample['action_mask'], device=prediction.device, dtype=torch.bool)
                state_mask = torch.as_tensor(sample['proprio_mask'], device=prediction.device, dtype=torch.bool)
                if reference.shape != prediction.shape or mask.shape != prediction.shape:
                    raise ValueError('Reference and action mask must match predicted chunk')
                if initial.shape != (1, prediction.shape[-1]) or state_mask.shape != initial.shape:
                    raise ValueError('Initial state and its dimension mask are required')
                h = self.execution_horizon or len(prediction)
                physical = decoder.decode(torch.cat((initial, prediction[:h])))
                ref = decoder.decode(torch.cat((initial, reference[:h])))
                valid = torch.cat((state_mask, mask[:h]))[:, decoder.indices]
                for c in constraints:
                    v = residual(physical, ref, valid, c, start_frame=int(sample['start_frame']), rotation_convention=convention)
                    if v is None:
                        continue
                    w = c['penalty']['weight']
                    losses.append(w*penalty(v, c['penalty'])); weights.append(w)
                    violations[c['type']].append(v.detach())
            numerator = torch.stack(losses).sum() if losses else clean_actions.reshape(-1)[:0].sum()
            denominator = clean_actions.new_tensor(sum(weights), dtype=torch.float32)
            local_weight = denominator.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(denominator)
                world_size = torch.distributed.get_world_size()
            loss = self.weight*numerator*world_size/denominator.clamp_min(1e-12)
            metrics = {'constraint_count': clean_actions.new_tensor(float(len(losses))), 'constraint_weight_sum': local_weight}
            for kind in KINDS:
                v = torch.cat(violations[kind]) if violations[kind] else clean_actions.new_empty(0)
                metrics.update({f'{kind}_count': v.new_tensor(float(v.numel())), f'{kind}_residual_sum': v.sum(),
                                f'{kind}_residual_max': v.max() if v.numel() else v.new_zeros(()),
                                f'{kind}_violation_count': (v > 0).float().sum()})
            return {'loss': loss, 'metrics': metrics}
        return evaluate


def build_constraint_objective(cfg, dataset):
    settings = cfg.training.get('constraints', {})
    weight = number(settings.get('weight', 0.0))
    if weight < 0:
        raise ValueError('Constraint weight must be nonnegative')
    if weight == 0:
        return None
    if cfg.dataloader.type != 'robotwin' or cfg.dataloader.variant != 'clean_50':
        raise ValueError('Embedded C2R constraints require RoboTwin clean_50')
    if cfg.model.architecture.framework != 'dual_system' or cfg.model.architecture.variant != 'joint_self_attn':
        raise ValueError('Constraint integration supports joint_self_attn only')
    if cfg.training.lambda_action <= 0:
        raise ValueError('Retain positive action FM weight')
    if settings.get('manifest'):
        raise ValueError('External manifests are obsolete; embed constraints in each HDF5 episode')
    return RoboTwinConstraintObjective(dataset, cfg.dataloader.dataset_dir, weight=weight,
                                       execution_horizon=settings.get('execution_horizon'))


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
    for kind in sorted(name.removesuffix("_residual_sum") for name in totals if name.endswith("_residual_sum")):
        count = totals[f"{kind}_count"]
        report[f"{kind}_count"] = count
        if count:
            report[f"{kind}_residual_mean"] = totals[f"{kind}_residual_sum"] / count
            report[f"{kind}_residual_max"] = totals[f"{kind}_residual_max"]
            report[f"{kind}_violation_rate"] = totals[f"{kind}_violation_count"] / count
    for name in ("constraint_candidate_count", "constraint_weight_sum"):
        if name in totals:
            report[name] = totals[name]
    value = losses["constraint"].detach().float().reshape(1)
    if accelerator is not None and accelerator.num_processes > 1:
        value = accelerator.gather(value)
    return {"loss_constraint": value.mean().item(), "constraint_metrics": report}
