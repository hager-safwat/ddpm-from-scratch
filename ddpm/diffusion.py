"""Training objective, ancestral sampling and an evaluation metric."""

import numpy as np
import torch
from torch import nn


def train(model, data, schedule, steps=15000, batch_size=512, lr=1e-3, log_every=3000):
    """Train eps_theta with the simple DDPM loss.

    Each step: pick random data points and random timesteps, jump straight
    to x_t with the closed-form forward process, and regress the exact noise
    that was injected (MSE). That single objective is all a DDPM needs.

    Returns the list of per-step losses.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch_size,))
        x0 = data[idx]
        t = torch.randint(0, schedule.T, (batch_size,))
        noise = torch.randn_like(x0)
        loss = loss_fn(model(schedule.q_sample(x0, t, noise), t), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:>6d}/{steps}   loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def sample(model, schedule, n, data_dim=2, record_every=None):
    """DDPM ancestral sampling from pure Gaussian noise.

    Walk from t = T-1 down to 0. At each step compute the posterior mean of
    x_{t-1} from the predicted noise, then (except at t = 0) add fresh noise
    with variance beta_t.

    Returns the final samples and a list of (t, x_t) snapshots.
    """
    x = torch.randn(n, data_dim)
    snapshots = []
    for t in reversed(range(schedule.T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps = model(x, t_batch)
        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        acp_t = schedule.alphas_cumprod[t]
        mean = (x - beta_t / torch.sqrt(1 - acp_t) * eps) / torch.sqrt(alpha_t)
        x = mean + torch.sqrt(beta_t) * torch.randn_like(x) if t > 0 else mean
        if record_every is not None and (t % record_every == 0 or t == schedule.T - 1):
            snapshots.append((t, x.clone()))
    return x, snapshots


def mean_nn_distance(a, b, chunk=1000):
    """Average distance from each point of ``a`` to its nearest point in ``b``.

    Used in both directions: generated -> real measures precision (do samples
    land on the data?), real -> generated measures coverage (is every part of
    the data reached?). Computed in chunks to keep memory small.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mins = []
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk]
        d = np.sqrt(((block[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        mins.append(d.min(axis=1))
    return float(np.concatenate(mins).mean())
