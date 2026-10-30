"""Regression tests for FP32 FM labels/reconstruction with mixed-precision forward.

An ideal FP32 velocity should recover the original labels to FP32 roundoff.
A real BF16 prediction still has output quantization error; the fix does not
claim to eliminate that error or quantization of the model's forward input.
"""

import pytest
import torch

from openwam.model.action_backbone.scheduler import ActionScheduler
from tests.test_openwam_trainer import _make_fake_loss_inputs, _make_tiny_arch


class RecordingScheduler(ActionScheduler):
    def __init__(self, sigma):
        super().__init__()
        self.sigmas = torch.full((1000,), sigma, dtype=torch.float32)
        self.timesteps = self.sigmas * 1000
        self.linear_timesteps_weights = torch.ones(1000)

    def add_noise(self, original_samples, noise, sigma):
        self.original = original_samples
        self.noise = noise
        self.sigma = sigma
        self.noisy = super().add_noise(original_samples, noise, sigma)
        return self.noisy

    def training_target(self, original_samples, noise):
        self.target = super().training_target(original_samples, noise)
        return self.target


def actions_and_mask():
    # Include fine sub-BF16 details and both time padding and unused 80D slots.
    actions = torch.linspace(-0.912345, 0.876543, 2 * 4 * 80).reshape(2, 4, 80)
    valid = torch.zeros_like(actions, dtype=torch.bool)
    valid[:, :, :20] = True
    valid[0, 3:] = False
    valid[1, 2:] = False
    return actions, valid


def run_loss(monkeypatch, *, sigma, dtype, error=0.0, quantize_output=False, through_prepare=False,
             return_loss_terms=False):
    arch = _make_tiny_arch()
    arch._dtype = dtype
    scheduler = RecordingScheduler(sigma)
    arch.action_backbone.scheduler = scheduler
    reference, valid = actions_and_mask()
    delta = torch.where(valid, error, 0.25).requires_grad_()
    captured = {}

    def forward(noisy, timesteps, **kwargs):
        captured["forward_input"] = noisy
        captured["forward_timesteps"] = timesteps
        velocity = scheduler.target + delta
        if quantize_output:
            velocity = velocity.to(torch.bfloat16)
        captured["velocity"] = velocity
        return torch.zeros_like(kwargs["latents"]), velocity

    def callback(clean):
        captured["clean"] = clean
        captured["constraint"] = (clean - reference).square()[valid].mean()
        return {"loss": captured["constraint"]}

    monkeypatch.setattr(arch, "forward", forward)
    inputs = _make_fake_loss_inputs(B=2)
    if through_prepare:
        # Exercise the normal trainer entry, without decoding images or models.
        class IdentityTransform:
            def apply(self, sample):
                return sample

        arch._pipeline_transform_instance = IdentityTransform()
        monkeypatch.setattr(arch, "preprocess", lambda **kwargs: dict(inputs))
        samples = [
            dict(video=[], prompt="test", action=action, action_mask=mask)
            for action, mask in zip(reference, valid, strict=True)
        ]
        prepared = arch.prepare_inputs(samples)
        captured["prepared_actions"] = prepared["actions"]
        result = arch.compute_loss(action_loss_callback=callback, return_loss_terms=return_loss_terms, **prepared)
    else:
        result = arch.compute_loss(
            actions=reference,
            action_is_pad=~valid,
            action_loss_callback=callback,
            return_loss_terms=return_loss_terms,
            **inputs,
        )
    return result, captured, scheduler, reference, valid, delta


def test_offline_gradient_probe_can_use_actual_weighted_loss_graph(monkeypatch):
    from openwam.train.constraint_calibration import measure_loss_gradients

    result, _, _, _, _, delta = run_loss(
        monkeypatch, sigma=0.5, dtype=torch.float32, error=0.03, return_loss_terms=True
    )
    torch.testing.assert_close(sum(result['loss_terms'].values()), result['loss'])
    diagnostics = measure_loss_gradients(result['loss_terms'], [delta])
    assert diagnostics['norms']['action'] > 0 and diagnostics['norms']['constraint'] > 0
    assert delta.grad is None
    result['loss'].backward()
    assert torch.isfinite(delta.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sigma", [0.0, 1e-5, 0.54321, 1.0])
@pytest.mark.parametrize("through_prepare", [False, True])
def test_ideal_velocity_recovers_fp32_reference(monkeypatch, dtype, sigma, through_prepare):
    torch.manual_seed(404)
    result, captured, scheduler, reference, valid, _ = run_loss(
        monkeypatch, sigma=sigma, dtype=dtype, through_prepare=through_prepare
    )
    for tensor in (scheduler.original, scheduler.noise, scheduler.sigma, scheduler.noisy, scheduler.target):
        assert tensor.dtype == torch.float32
    torch.testing.assert_close(scheduler.original, reference, rtol=0, atol=0)
    torch.testing.assert_close(scheduler.sigma, torch.full((2, 1, 1), sigma), rtol=0, atol=0)
    assert captured["forward_input"].dtype == dtype
    assert captured["forward_timesteps"].dtype == dtype
    assert captured["clean"].dtype == torch.float32
    torch.testing.assert_close(captured["clean"][valid], reference[valid], rtol=0, atol=3e-7)
    assert result["loss_action"].item() == 0
    assert result["loss_constraint"].item() < 1e-13
    if through_prepare:
        assert captured["prepared_actions"].dtype == torch.float32


@pytest.mark.parametrize("sigma", [0.0, 0.54321, 1.0])
def test_constraint_gradient_reaches_only_valid_velocity_cells(monkeypatch, sigma):
    result, captured, _, reference, valid, delta = run_loss(
        monkeypatch, sigma=sigma, dtype=torch.bfloat16, error=0.03
    )
    expected = reference - sigma * delta
    torch.testing.assert_close(captured["clean"], expected, rtol=0, atol=4e-7)
    captured["constraint"].backward()
    assert torch.isfinite(delta.grad).all()
    assert torch.count_nonzero(delta.grad[~valid]) == 0
    if sigma:
        assert torch.count_nonzero(delta.grad[valid]) == valid.sum()
        expected_grad = 2 * sigma**2 * delta.detach()[valid] / valid.sum()
        torch.testing.assert_close(delta.grad[valid], expected_grad, rtol=3e-5, atol=1e-8)
    else:
        assert torch.count_nonzero(delta.grad) == 0
    torch.testing.assert_close(result["loss_action"], torch.tensor(0.03**2), rtol=1e-5, atol=1e-8)


def test_bf16_prediction_error_is_distinct_from_fm_arithmetic_error(monkeypatch):
    torch.manual_seed(404)
    _, captured, scheduler, reference, valid, _ = run_loss(
        monkeypatch, sigma=0.54321, dtype=torch.bfloat16, quantize_output=True
    )
    expected_output_error = -scheduler.sigma * (captured["velocity"].float() - scheduler.target)
    actual_error = captured["clean"] - reference
    torch.testing.assert_close(actual_error[valid], expected_output_error[valid], rtol=0, atol=3e-7)
    assert actual_error[valid].abs().max() > 1e-4

    # Reproduce the old full BF16 arithmetic with an otherwise ideal velocity:
    # merely casting the already rounded noisy sample/target back is too late.
    original_bf16 = reference.bfloat16()
    noise_bf16 = scheduler.noise.bfloat16()
    sigma_bf16 = scheduler.sigma.bfloat16()
    noisy_bf16 = (1 - sigma_bf16) * original_bf16 + sigma_bf16 * noise_bf16
    target_bf16 = noise_bf16 - original_bf16
    old_clean = noisy_bf16.float() - sigma_bf16.float() * target_bf16.float()
    assert (old_clean - reference)[valid].abs().max() > 1e-3


def test_outer_bf16_autocast_keeps_fm_chain_fp32(monkeypatch):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, captured, scheduler, reference, valid, _ = run_loss(
            monkeypatch, sigma=0.54321, dtype=torch.bfloat16, through_prepare=True
        )
    assert scheduler.noisy.dtype == scheduler.target.dtype == captured["clean"].dtype == torch.float32
    torch.testing.assert_close(captured["clean"][valid], reference[valid], rtol=0, atol=3e-7)


def test_opt_in_callback_receives_actual_fp32_sigma_and_indexed_weight(monkeypatch):
    arch = _make_tiny_arch()
    arch._dtype = torch.bfloat16
    scheduler = RecordingScheduler(0.54321)
    scheduler.linear_timesteps_weights = torch.full((1000,), 1.234567)
    arch.action_backbone.scheduler = scheduler
    reference, _ = actions_and_mask()
    captured = {}

    def forward(noisy, timesteps, **kwargs):
        return torch.zeros_like(kwargs["latents"]), scheduler.target

    def callback(clean, *, fm_sigma, fm_timestep_weight):
        captured.update(sigma=fm_sigma, weight=fm_timestep_weight)
        return {"loss": clean.sum() * 0}

    callback.accepts_fm_context = True
    monkeypatch.setattr(arch, "forward", forward)
    arch.compute_loss(actions=reference, action_loss_callback=callback, **_make_fake_loss_inputs(B=2))
    assert captured["sigma"].dtype == captured["weight"].dtype == torch.float32
    torch.testing.assert_close(captured["sigma"], torch.full((2,), 0.54321), rtol=0, atol=0)
    torch.testing.assert_close(captured["weight"], torch.full((2,), 1.234567), rtol=0, atol=0)
