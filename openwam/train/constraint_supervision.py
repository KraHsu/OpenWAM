"""Offline geometric constraints embedded in RoboTwin episodes.

No operation classes, external label manifest, simulator or dynamics model.
The callback receives the differentiable clean estimate, not a velocity field.
"""
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from openwam.train.constraint_calibration import timestep_factors
from openwam.train.constraint_coverage import chunk_candidate, coverage_report
from openwam.train.constraint_schema import canonical_json, number, validate_bundle
from openwam.train.meta_constraints import ANGLE_CENTER_POLICIES, penalty, residual

SIGMA_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)

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
    def __init__(self, dataset, dataset_root, *, weight, execution_horizon=None, require_full_coverage=True,
                 timestep_strategy='clean', sigma_floor=0.05, penalty_scale_overrides=None,
                 angle_center_backward='legacy', angle_center_log_per_constraint=True):
        self.weight = number(weight, positive=True)
        if angle_center_backward not in ANGLE_CENTER_POLICIES:
            raise ValueError('Unknown angular-center backward policy')
        self.angle_center_backward = angle_center_backward
        if type(angle_center_log_per_constraint) is not bool:
            raise ValueError('angle_center_log_per_constraint must be boolean')
        self.angle_center_log_per_constraint = angle_center_log_per_constraint
        self.angle_center_metric_keys, self.angle_center_metric_catalog = {}, {}
        self.angle_center_hint_constraints = 0
        if timestep_strategy not in ('clean', 'fm_matched'):
            raise ValueError('timestep_strategy must be clean or fm_matched')
        self.timestep_strategy = timestep_strategy
        self.sigma_floor = number(sigma_floor, positive=True)
        if self.sigma_floor > 1:
            raise ValueError('sigma_floor must be at most 1')
        from openwam.train.constraint_schema import KINDS
        self.penalty_scale_overrides = dict(penalty_scale_overrides or {})
        if set(self.penalty_scale_overrides) - set(KINDS):
            raise ValueError('Unknown constraint kind in penalty_scale_overrides')
        self.penalty_scale_overrides = {
            kind: number(scale, positive=True) for kind, scale in self.penalty_scale_overrides.items()
        }
        if execution_horizon is not None and (type(execution_horizon) is not int or execution_horizon < 1):
            raise ValueError('execution_horizon must be null or a positive integer')
        self.execution_horizon = execution_horizon
        self.require_full_coverage = require_full_coverage
        self.coverage = {}
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
                            if c['type']=='axis_angle_interval' and c['parameters']['bounds'][0]>0:
                                if angle_center_backward=='fixed_annotation_v1' and 'center_tangent_hint' not in c['parameters']:
                                    raise ValueError(f'Positive-lower angular constraint {c["id"]} lacks its fixed annotation tangent; regenerate explicitly, never infer one from a chunk')
                                self.angle_center_hint_constraints += int('center_tangent_hint' in c['parameters'])
                                if angle_center_backward!='legacy' and angle_center_log_per_constraint:
                                    identity=dict(episode=path.relative_to(root).as_posix(),constraint_id=c['id'])
                                    key='angle_center_c_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                                    self.angle_center_metric_keys[(str(path),c['id'])]=key
                                    self.angle_center_metric_catalog[key]=identity
                            constraints.append(c)
                    archived.append({'episode': path.relative_to(root).as_posix(), 'bundle': bundle})
                report = coverage_report(constraints, length)
                self.coverage[str(path)] = report
                if require_full_coverage and not report['complete']:
                    raise ValueError(f'Incomplete chunk-safe motion coverage: {path}; {report["covered"]}/{report["total"]} frames')
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
        result = {'embedded_constraints_sha256': self.store.sha256, 'weight': self.weight,
                'matched_constraints': self.matched_constraints, 'status_counts': dict(self.store.status_counts),
                'source_counts': dict(self.store.source_counts), 'objective_version': 'meta_intervals_v5_rigid_local_point',
                'timestep_strategy': self.timestep_strategy, 'sigma_floor': self.sigma_floor,
                'penalty_scale_overrides': self.penalty_scale_overrides,
                'sigma_bins': list(SIGMA_BINS), 'reduction': 'global_active_constraint_weight_mean',
                'require_full_coverage': self.require_full_coverage,
                'fully_covered_episodes': sum(r['complete'] for r in self.coverage.values()),
                'execution_horizon': self.execution_horizon, 'gradient_entry': 'single_step_clean_action_estimate',
                'decoder_sha256': hashlib.sha256(json.dumps(decoders).encode()).hexdigest()}
        if self.angle_center_backward != 'legacy':
            result.update(objective_version='meta_intervals_v6_fixed_angular_center',
                angle_center_backward=dict(policy=self.angle_center_backward,
                    selection='exact_cross_zero_dot_positive_and_lower_positive',
                    invalid_hint='reject_before_training; no world-axis fallback',
                    hint_constraints=self.angle_center_hint_constraints,
                    log_per_constraint=self.angle_center_log_per_constraint,
                    contribution_metric_catalog=self.angle_center_metric_catalog,
                    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('meta_constraints.py','constraint_schema.py','action_trajectory.py','constraint_supervision.py')}))
        return result

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
            horizon = min(self.execution_horizon or len(sample['action']), len(sample['action']))
            candidates = [c for c in entry[1] if chunk_candidate(c, int(sample['start_frame']), horizon)]
            entries.append((entry[0], candidates, entry[2], entry[3], sample))

        def evaluate(clean_actions, *, fm_sigma=None, fm_timestep_weight=None):
            if clean_actions.ndim != 3 or clean_actions.shape[0] != len(entries):
                raise ValueError('Expected clean actions [batch,horizon,action_dim]')
            if (fm_sigma is None) != (fm_timestep_weight is None):
                raise ValueError('Both FM sigma and timestep weight must be supplied together')
            if fm_sigma is None:
                if self.timestep_strategy != 'clean':
                    raise ValueError('fm_matched constraints require FM timestep context')
                factors = clean_actions.new_ones(len(entries), dtype=torch.float32)
                sigmas = None
            else:
                sigmas = torch.as_tensor(fm_sigma, device=clean_actions.device, dtype=torch.float32).detach()
                fm_weights = torch.as_tensor(fm_timestep_weight, device=clean_actions.device, dtype=torch.float32).detach()
                if sigmas.shape != (len(entries),) or fm_weights.shape != sigmas.shape:
                    raise ValueError('FM context must have shape [batch]')
                factors = timestep_factors(sigmas, fm_weights, strategy=self.timestep_strategy,
                                           sigma_floor=self.sigma_floor)
            # One device synchronization per batch, never one per constraint.
            sigma_bin_indices = None if sigmas is None else (sigmas*4).long().clamp_max(3).cpu().tolist()
            losses, weights, violations = [], [], defaultdict(list)
            from openwam.train.constraint_schema import KINDS
            zero = clean_actions.new_zeros((), dtype=torch.float32)
            contributions = {kind: zero.clone() for kind in KINDS}
            raw_contributions = {kind: zero.clone() for kind in KINDS}
            kind_weights = Counter()
            bin_losses = [zero.clone() for _ in range(len(SIGMA_BINS)-1)]
            bin_weights = [0.0 for _ in bin_losses]
            bin_counts = [0 for _ in bin_losses]
            center_metrics = {name:zero.clone() for name in ('angle_center_hit_count',
                'angle_center_hint_frame_count','angle_center_invalid_hint_count')}
            center_contributions = zero.clone()
            # Fixed dataset-wide keys keep DDP metric columns aligned across
            # ranks with different active chunks. Inactive entries are zero;
            # updates below are out of place, so the shared zero is not mutated.
            center_constraint_contributions = {key:zero for key in self.angle_center_metric_catalog}
            for sample_index, (prediction, (decoder, constraints, convention, _, sample)) in enumerate(
                    zip(clean_actions, entries, strict=True)):
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
                    v = residual(physical, ref, valid, c, start_frame=int(sample['start_frame']), rotation_convention=convention,
                                 angle_center_backward=self.angle_center_backward,diagnostics=center_metrics)
                    if v is None:
                        continue
                    w = c['penalty']['weight']
                    kind = c['type']
                    spec = dict(c['penalty'])
                    if kind in self.penalty_scale_overrides:
                        spec['scale'] = self.penalty_scale_overrides[kind]
                    raw_loss = w*penalty(v, spec)
                    weighted_loss = factors[sample_index]*raw_loss
                    if self.angle_center_backward=='fixed_annotation_v1' and c['type']=='axis_angle_interval' and c['parameters']['bounds'][0]>0:
                        center_contributions += weighted_loss.detach()
                        key=self.angle_center_metric_keys.get((str(Path(sample['episode_path']).resolve()),c['id']))
                        if key is not None:
                            center_constraint_contributions[key]=center_constraint_contributions[key]+weighted_loss.detach()
                    losses.append(weighted_loss)
                    weights.append(w)
                    violations[kind].append(v.detach())
                    raw_contributions[kind] += raw_loss.detach()
                    contributions[kind] += weighted_loss.detach()
                    kind_weights[kind] += w
                    if sigma_bin_indices is not None:
                        index = sigma_bin_indices[sample_index]
                        bin_losses[index] += weighted_loss.detach()
                        bin_weights[index] += w
                        bin_counts[index] += 1
            numerator = torch.stack(losses).sum() if losses else clean_actions.reshape(-1)[:0].sum()
            denominator = clean_actions.new_tensor(sum(weights), dtype=torch.float32)
            local_weight = denominator.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(denominator)
                world_size = torch.distributed.get_world_size()
            loss = self.weight*numerator*world_size/denominator.clamp_min(1e-12)
            metrics = {'constraint_count': zero.new_tensor(float(len(losses))),
                       'constraint_weight_sum': local_weight, 'constraint_objective_weight': zero.new_tensor(self.weight)}
            if self.angle_center_backward!='legacy':
                metrics.update(center_metrics,angle_center_policy_penalty_sum=center_contributions)
                metrics.update({key+'_penalty_sum':value for key,value in center_constraint_contributions.items()})
            for kind in KINDS:
                v = torch.cat(violations[kind]) if violations[kind] else clean_actions.new_empty(0)
                metrics.update({f'{kind}_count': v.new_tensor(float(v.numel())), f'{kind}_residual_sum': v.sum(),
                                f'{kind}_residual_max': v.max() if v.numel() else v.new_zeros(()),
                                f'{kind}_violation_count': (v > 0).float().sum(),
                                f'{kind}_penalty_sum': contributions[kind],
                                f'{kind}_raw_penalty_sum': raw_contributions[kind],
                                f'{kind}_weight_sum': zero.new_tensor(kind_weights[kind])})
            for index, (bin_loss, bin_weight, bin_count) in enumerate(zip(bin_losses, bin_weights, bin_counts)):
                metrics.update({f'sigma_bin_{index}_penalty_sum': bin_loss,
                                f'sigma_bin_{index}_weight_sum': zero.new_tensor(bin_weight),
                                f'sigma_bin_{index}_count': zero.new_tensor(bin_count)})
            return {'loss': loss, 'metrics': metrics}
        evaluate.accepts_fm_context = True
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
                                       execution_horizon=settings.get('execution_horizon'),
                                       require_full_coverage=settings.get('require_full_coverage', True),
                                       timestep_strategy=settings.get('timestep_strategy', 'clean'),
                                       sigma_floor=settings.get('sigma_floor', 0.05),
                                       penalty_scale_overrides=settings.get('penalty_scale_overrides', {}),
                                       angle_center_backward=settings.get('angle_center_backward','legacy'),
                                       angle_center_log_per_constraint=settings.get('angle_center_log_per_constraint',True))


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
    # Every rank uses the same objective coefficient; do not sum it as a count.
    objective_weight = values[:, names.index('constraint_objective_weight')].mean().item() if 'constraint_objective_weight' in names else 1.0
    weight_sum = totals.get('constraint_weight_sum', 0.0)
    denominator = max(weight_sum, 1e-12)
    for name in ('angle_center_hit_count','angle_center_hint_frame_count','angle_center_invalid_hint_count'):
        if name in totals:
            report[name]=totals[name]
    if 'angle_center_policy_penalty_sum' in totals:
        report['angle_center_policy_loss_contribution']=objective_weight*totals['angle_center_policy_penalty_sum']/denominator
    for name,value in totals.items():
        if name.startswith('angle_center_c_') and name.endswith('_penalty_sum'):
            report[name.removesuffix('_penalty_sum')+'_loss_contribution']=objective_weight*value/denominator
    for kind in sorted(name.removesuffix("_residual_sum") for name in totals if name.endswith("_residual_sum")):
        count = totals[f"{kind}_count"]
        report[f"{kind}_count"] = count
        if count:
            report[f"{kind}_residual_mean"] = totals[f"{kind}_residual_sum"] / count
            report[f"{kind}_residual_max"] = totals[f"{kind}_residual_max"]
            report[f"{kind}_violation_rate"] = totals[f"{kind}_violation_count"] / count
        if f'{kind}_penalty_sum' in totals:
            report[f'{kind}_loss_contribution'] = objective_weight*totals[f'{kind}_penalty_sum']/denominator
            report[f'{kind}_raw_loss_contribution'] = objective_weight*totals[f'{kind}_raw_penalty_sum']/denominator
            report[f'{kind}_weight_sum'] = totals[f'{kind}_weight_sum']
    for index in range(len(SIGMA_BINS)-1):
        prefix = f'sigma_bin_{index}'
        if f'{prefix}_count' in totals:
            report[f'{prefix}_count'] = totals[f'{prefix}_count']
            report[f'{prefix}_loss_contribution'] = objective_weight*totals[f'{prefix}_penalty_sum']/denominator
            report[f'{prefix}_weight_sum'] = totals[f'{prefix}_weight_sum']
    for name in ("constraint_candidate_count", "constraint_weight_sum"):
        if name in totals:
            report[name] = totals[name]
    value = losses["constraint"].detach().float().reshape(1)
    if accelerator is not None and accelerator.num_processes > 1:
        value = accelerator.gather(value)
    return {"loss_constraint": value.mean().item(), "constraint_metrics": report}
