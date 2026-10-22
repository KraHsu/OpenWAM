"""Tests for video/action loss helpers on BaseWAMArchitecture.

These tests exercise architecture._compute_video_loss and
architecture._compute_action_loss, which replaced the standalone
FlowMatchVideoActionLoss class.
"""

import pytest
import torch


def _make_arch():
    """Build a minimal architecture with a mock video backbone for loss tests."""
    from tests.test_openwam_trainer import _make_tiny_arch

    return _make_tiny_arch()


class _MockScheduler:
    num_train_timesteps = 1000
    linear_timesteps_weights = torch.ones(1000)
    timesteps = torch.linspace(0, 1, 1000)
    sigmas = torch.linspace(1, 0, 1000)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


def test_video_loss_basic():
    """Video loss produces a positive scalar."""
    arch = _make_arch()
    noise_pred = torch.randn(2, 16, 5, 4, 4)
    target = torch.randn(2, 16, 5, 4, 4)
    timestep_ids = torch.tensor([10, 20])

    loss = arch._compute_video_loss(noise_pred, target, timestep_ids, {}, device="cpu")
    assert loss.shape == ()
    assert loss.item() > 0


def test_action_loss_basic():
    """Action loss produces a positive scalar."""
    arch = _make_arch()
    noise_pred = torch.randn(2, 49, 14)
    target = torch.randn(2, 49, 14)
    timestep_ids = torch.tensor([5, 15])

    loss = arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")
    assert loss.shape == ()
    assert loss.item() > 0


def test_action_loss_single_sample():
    """Loss computation should handle B=1."""
    arch = _make_arch()
    noise_pred = torch.randn(1, 49, 14)
    target = torch.randn(1, 49, 14)
    timestep_ids = torch.tensor([10])

    loss = arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")
    assert loss.shape == ()


def test_action_loss_per_token_timestep():
    """_compute_action_loss rejects (B, T) timestep_ids in FastWAM-compatible mode."""
    arch = _make_arch()
    B, T, action_dim = 2, 49, 14
    noise_pred = torch.randn(B, T, action_dim)
    target = torch.randn(B, T, action_dim)
    timestep_ids = torch.randint(0, 1000, (B, T))

    with pytest.raises(ValueError, match="per-sample"):
        arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs={}, device="cpu")


def test_action_loss_per_token_timestep_with_pad():
    """Per-token timestep is rejected even when action padding is present."""
    arch = _make_arch()
    B, T, action_dim = 2, 10, 4
    noise_pred = torch.randn(B, T, action_dim)
    target = torch.randn(B, T, action_dim)
    timestep_ids = torch.randint(0, 1000, (B, T))
    action_is_pad = torch.zeros(B, T, dtype=torch.bool)
    action_is_pad[0, 7:] = True
    action_is_pad[1, 4:] = True

    inputs = {"action_is_pad": action_is_pad}
    with pytest.raises(ValueError, match="per-sample"):
        arch._compute_action_loss(noise_pred, target, timestep_ids, _MockScheduler(), inputs=inputs, device="cpu")


def test_constraint_callback_preserves_fm_terms_and_backpropagates():
    """Requires the normal OpenWAM CPU test environment, but no model weights."""
    from tests.test_openwam_trainer import _make_fake_loss_inputs

    arch = _make_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    actions = torch.randn(1, 5, 7)
    inputs = _make_fake_loss_inputs()
    torch.manual_seed(17)
    baseline = arch.compute_loss(actions=actions, **inputs)
    captured = {}

    def callback(clean):
        captured["clean"] = clean
        clean.retain_grad()
        loss = clean.square().mean()
        captured["loss"] = loss
        return {"loss": loss, "metrics": {"test_count": clean.new_tensor(1.0)}}

    torch.manual_seed(17)
    augmented = arch.compute_loss(actions=actions, action_loss_callback=callback, **inputs)
    torch.testing.assert_close(augmented["loss_video"], baseline["loss_video"], rtol=0, atol=0)
    torch.testing.assert_close(augmented["loss_action"], baseline["loss_action"], rtol=0, atol=0)
    torch.testing.assert_close(augmented["loss"], baseline["loss"] + captured["loss"])
    captured["loss"].backward()
    assert captured["clean"].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in arch.action_backbone.parameters())


def test_constraint_callback_clean_estimate_has_correct_flow_sign(monkeypatch):
    from tests.test_openwam_trainer import _make_fake_loss_inputs

    arch = _make_arch()
    scheduler = _MockScheduler()
    scheduler.sigmas = torch.full((1000,), 0.5)
    arch.action_backbone.scheduler = scheduler
    actions = torch.randn(1, 5, 7)
    delta = torch.full_like(actions, 0.2, requires_grad=True)

    def forward(noisy, action_timestep, **kwargs):
        velocity = (noisy - actions) / 0.5 + delta
        return torch.zeros_like(kwargs["latents"]), velocity

    monkeypatch.setattr(arch, "forward", forward)
    captured = {}

    def callback(clean):
        captured["clean"] = clean
        return {"loss": clean.sum()}

    result = arch.compute_loss(actions=actions, action_loss_callback=callback, **_make_fake_loss_inputs())
    torch.testing.assert_close(captured["clean"], actions - 0.5 * delta)
    # The constraint gradient must reach the velocity prediction with -sigma.
    captured["clean"].sum().backward()
    torch.testing.assert_close(delta.grad, torch.full_like(delta, -0.5))
    assert torch.isfinite(result["loss"])
