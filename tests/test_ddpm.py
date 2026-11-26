"""Tests for the DDPM implementation."""

import numpy as np
import pytest
import torch

from ddpm import (
    EpsilonModel,
    NoiseSchedule,
    SinusoidalTimeEmbedding,
    make_two_moons,
    mean_nn_distance,
    sample,
    train,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_data_shape_and_standardisation():
    x = make_two_moons(1000, np.random.default_rng(0))
    assert x.shape == (1000, 2)
    assert torch.allclose(x.mean(0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(x.std(0, unbiased=False), torch.ones(2), atol=1e-4)


def test_schedule_is_monotone():
    s = NoiseSchedule(T=200)
    assert torch.all(s.betas[1:] > s.betas[:-1])
    assert torch.all(s.alphas_cumprod[1:] < s.alphas_cumprod[:-1])
    assert 0 < s.alphas_cumprod[-1] < s.alphas_cumprod[0] < 1


def test_scaled_schedule_destroys_the_signal():
    """With the 1000/T rescaling, x_T is essentially pure noise, which is
    what sampling assumes when it starts from N(0, I)."""
    assert NoiseSchedule(T=200).alphas_cumprod[-1] < 1e-3
    # Without it, the original betas leave about 13 % of alpha_bar at T = 200.
    unscaled = NoiseSchedule(T=200, scale_to_T=False).alphas_cumprod[-1]
    assert unscaled == pytest.approx(0.132, abs=0.005)
    # And the rescaled 200-step schedule matches the 1000-step reference.
    ref = NoiseSchedule(T=1000).alphas_cumprod[-1]
    assert NoiseSchedule(T=200).alphas_cumprod[-1] == pytest.approx(float(ref), rel=0.5)


def test_betas_stay_below_one_for_small_T():
    """1000/T rescaling must not produce beta >= 1 (alpha = 0) when T is small."""
    for T in (5, 20, 50):
        s = NoiseSchedule(T=T)
        assert s.betas.max() < 1.0
        assert torch.all(s.alphas > 0)


def test_closed_form_matches_step_by_step_forward_process():
    """Applying x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) eps repeatedly must
    give the same mean and variance as the one-shot formula q_sample."""
    s = NoiseSchedule(T=50)
    x0 = torch.tensor([1.0, -2.0])
    n = 200_000
    x = x0.expand(n, 2).clone()
    t_check = 30
    for t in range(t_check + 1):
        x = torch.sqrt(s.alphas[t]) * x + torch.sqrt(s.betas[t]) * torch.randn_like(x)

    one_shot = s.q_sample(x0.expand(n, 2), torch.full((n,), t_check), torch.randn(n, 2))
    assert torch.allclose(x.mean(0), one_shot.mean(0), atol=0.01)
    assert torch.allclose(x.var(0), one_shot.var(0), atol=0.01)
    assert torch.allclose(x.mean(0), s.sqrt_acp[t_check] * x0, atol=0.01)


def test_time_embedding_shape_and_distinct_steps():
    emb = SinusoidalTimeEmbedding(64)(torch.tensor([0, 1, 199]))
    assert emb.shape == (3, 64)
    assert not torch.allclose(emb[0], emb[1])


def test_model_output_shape():
    model = EpsilonModel()
    out = model(torch.randn(16, 2), torch.randint(0, 200, (16,)))
    assert out.shape == (16, 2)


def test_training_reduces_loss():
    data = make_two_moons(1000, np.random.default_rng(0))
    model = EpsilonModel(hidden=64)
    losses = train(model, data, NoiseSchedule(T=50), steps=300, batch_size=256, log_every=0)
    assert np.mean(losses[-50:]) < np.mean(losses[:50])


def test_sampling_shape_snapshots_and_reproducibility():
    s = NoiseSchedule(T=20)
    model = EpsilonModel(hidden=32)
    torch.manual_seed(1)
    x_a, snaps = sample(model, s, 50, record_every=5)
    torch.manual_seed(1)
    x_b, _ = sample(model, s, 50, record_every=5)
    assert x_a.shape == (50, 2) and torch.isfinite(x_a).all()
    assert torch.equal(x_a, x_b)
    assert [t for t, _ in snaps] == [19, 15, 10, 5, 0]


def test_mean_nn_distance():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert mean_nn_distance(a, a) == 0.0
    assert mean_nn_distance(a, np.array([[0.0, 1.0]])) == pytest.approx((1 + 2 ** 0.5) / 2)


def test_demo_smoke_run():
    import ddpm_demo

    m = ddpm_demo.main(["--steps", "30", "--no-figure"])
    for key in ("gen_to_real", "real_to_gen", "real_to_real"):
        assert np.isfinite(m[key])


def test_parameter_count_matches_readme():
    assert sum(p.numel() for p in EpsilonModel().parameters()) == 149_250
