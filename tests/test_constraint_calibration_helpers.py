"""CPU tests for explicit time-weighting and side-effect-free gradient diagnostics."""

import math

import pytest
import torch

from openwam.train.constraint_calibration import measure_loss_gradients, timestep_factors


def test_clean_factors_are_one_and_detached():
    sigma = torch.tensor([0.0, 0.1, 1.0], requires_grad=True)
    weights = torch.tensor([0.0, 1.5, 2.0], requires_grad=True)
    result = timestep_factors(sigma, weights)
    torch.testing.assert_close(result, torch.ones(3))
    assert result.dtype == torch.float32 and not result.requires_grad


def test_fm_matched_factors_are_finite_at_endpoints_and_explicit_floor():
    sigma = torch.tensor([0.0, 0.01, 0.05, 0.5, 1.0])
    weights = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    result = timestep_factors(sigma, weights, strategy="fm_matched", sigma_floor=0.05)
    torch.testing.assert_close(result, torch.tensor([0.0, 400.0, 800.0, 12.0, 4.0]))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[2:] * sigma[2:].square(), weights[2:])
    assert timestep_factors(torch.tensor([0.0]), torch.tensor([1.0]), strategy="fm_matched").item() == pytest.approx(400)


def test_fm_matching_cancels_quadratic_sigma_curvature_above_floor():
    velocity = torch.tensor([0.3, 0.3, 0.3], requires_grad=True)
    sigma = torch.tensor([0.1, 0.5, 1.0])
    weights = torch.tensor([0.7, 1.3, 0.4])
    clean_error = -sigma * velocity
    geometry = (timestep_factors(sigma, weights, strategy="fm_matched") * clean_error.square()).sum()
    fm = (weights * velocity.square()).sum()
    torch.testing.assert_close(torch.autograd.grad(geometry, velocity)[0], torch.autograd.grad(fm, velocity)[0])


@pytest.mark.parametrize("sigma", [[-0.1], [1.1], [math.nan], [math.inf]])
def test_invalid_sigma_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        timestep_factors(torch.tensor(sigma), torch.ones(1))


@pytest.mark.parametrize("weights", [[-0.1], [math.nan], [math.inf]])
def test_invalid_fm_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="fm_weight"):
        timestep_factors(torch.ones(1), torch.tensor(weights))


@pytest.mark.parametrize("floor", [0.0, -0.1, 1.1, math.nan, math.inf, True])
def test_invalid_floor_is_rejected(floor):
    with pytest.raises(ValueError, match="sigma_floor"):
        timestep_factors(torch.ones(1), torch.ones(1), sigma_floor=floor)


def test_bad_shape_dtype_and_strategy_are_rejected():
    with pytest.raises(ValueError, match="shapes"):
        timestep_factors(torch.ones(2, 1), torch.ones(2, 1))
    with pytest.raises(ValueError, match="shapes"):
        timestep_factors(torch.ones(2), torch.ones(1))
    with pytest.raises(TypeError, match="floating"):
        timestep_factors(torch.ones(2, dtype=torch.long), torch.ones(2))
    with pytest.raises(ValueError, match="strategy"):
        timestep_factors(torch.ones(1), torch.ones(1), strategy="automatic")


def test_conflicting_weighted_gradients_and_existing_grad_buffers_are_preserved():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    frozen = torch.tensor([4.0])
    parameter.grad = torch.tensor([7.0, 8.0])
    unused.grad = torch.tensor([9.0])
    buffers = [parameter.grad, unused.grad]
    snapshots = [buffer.clone() for buffer in buffers]
    x = parameter.square().sum()
    terms = {"action": x, "constraint": -0.5 * x, "constant": torch.tensor(3.0)}
    report = measure_loss_gradients(terms, [parameter, unused, frozen])
    assert report["scope"] == "provided_parameters_only"
    assert report["parameter_count"] == 3 and report["trainable_parameter_count"] == 2
    assert report["norms"]["action"] == pytest.approx(math.sqrt(20))
    assert report["norms"]["constraint"] == pytest.approx(math.sqrt(5))
    first = report["pairs"][0]
    assert first["cosine"] == pytest.approx(-1.0)
    assert first["norm_ratio"] == pytest.approx(2.0)
    assert first["cosine_valid"] and first["ratio_valid"]
    for pair in report["pairs"][1:]:
        assert pair["cosine"] is None and not pair["cosine_valid"]
        assert pair["norm_ratio"] is None and not pair["ratio_valid"]
    assert parameter.grad is buffers[0] and unused.grad is buffers[1]
    for original, snapshot in zip(buffers, snapshots, strict=True):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
    # Graph remains intact and the eventual training backward still works.
    (terms["action"] + terms["constraint"]).backward()
    torch.testing.assert_close(parameter.grad, snapshots[0] + parameter.detach())
    torch.testing.assert_close(unused.grad, snapshots[1])


def test_small_network_diagnostics_match_independent_vector_gradients():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    parameters = tuple(model.parameters())
    output = model(torch.tensor([[0.3, -0.7], [0.8, 0.1]]))
    terms = {"action": output.square().mean(), "constraint": 0.1 * (output - 1).square().mean()}
    vectors = {
        name: torch.cat([g.flatten() for g in torch.autograd.grad(term, parameters, retain_graph=True)])
        for name, term in terms.items()
    }
    report = measure_loss_gradients(terms, parameters)
    for name, vector in vectors.items():
        assert report["norms"][name] == pytest.approx(float(vector.norm()), rel=1e-6)
    cosine = torch.nn.functional.cosine_similarity(vectors["action"], vectors["constraint"], dim=0)
    assert report["pairs"][0]["cosine"] == pytest.approx(float(cosine), rel=1e-6)
    assert all(parameter.grad is None for parameter in parameters)


def test_empty_unused_and_zero_gradient_cases_have_no_nan():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    unrelated = torch.tensor(2.0, requires_grad=True)
    for parameters in [[], [parameter]]:
        report = measure_loss_gradients({"zero": parameter.sum() * 0, "unused": unrelated.square()}, parameters)
        assert report["norms"] == {"zero": 0.0, "unused": 0.0}
        assert report["pairs"][0]["cosine"] is None
        assert report["pairs"][0]["norm_ratio"] is None
    assert measure_loss_gradients({}, [parameter])["pairs"] == []
    report = measure_loss_gradients({"zero": parameter.sum() * 0, "nonzero": parameter.sum()}, [parameter])
    assert report["pairs"][0]["norm_ratio"] == 0
    assert report["pairs"][0]["ratio_valid"]
    assert report["pairs"][0]["cosine"] is None


def test_invalid_gradient_diagnostic_inputs_are_rejected():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="duplicate"):
        measure_loss_gradients({"loss": parameter.sum()}, [parameter, parameter])
    with pytest.raises(TypeError, match="scalar"):
        measure_loss_gradients({"loss": parameter}, [parameter])
    with pytest.raises(ValueError, match="non-finite loss"):
        measure_loss_gradients({"loss": parameter.sum() * math.inf}, [parameter])
    zero = torch.tensor(0.0, requires_grad=True)
    with pytest.raises(ValueError, match="non-finite gradient"):
        measure_loss_gradients({"loss": zero.sqrt()}, [zero])
