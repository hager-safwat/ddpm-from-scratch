"""
A denoising diffusion probabilistic model (DDPM) from scratch
=============================================================

A minimal, self-contained implementation of the DDPM of Ho et al. (2020),
trained to generate samples from a 2-D toy distribution (two interleaving
moons). Everything runs on CPU in well under a minute.

The point of keeping it in 2-D is that you can *see* the whole method: the
forward process that gradually turns data into noise, the network that
learns to predict that noise, and the reverse process that turns noise back
into data. The exact same machinery scales to images -- only the noise-
prediction network changes (an MLP here, a U-Net for images).

Two ideas do all the work:

  1. Forward process (fixed, no learning). Starting from a real point x0,
     add Gaussian noise in T small steps until nothing but noise remains.
     A closed form lets us jump straight to any step t:

         x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * eps,   eps ~ N(0, I)

     where acp_t is the cumulative product of (1 - beta) up to step t.

  2. Reverse process (learned). Train a network eps_theta(x_t, t) to predict
     the noise eps that was added. Once it can do that, we can start from
     pure noise and walk backwards step by step, subtracting the predicted
     noise a little at a time, to produce a fresh sample.

All data is synthetic. This is a mechanism demo, not a benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
device = "cpu"


# ---------------------------------------------------------------------------
# 1. The target distribution
# ---------------------------------------------------------------------------
# Two interleaving half-moons. It is a classic "hard" 2-D shape: two curved,
# non-convex clusters. If the model reproduces it, it has genuinely learned a
# non-trivial distribution, not just a blob.
def make_two_moons(n, noise=0.06):
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * rng.uniform(0, 1, n1)          # upper moon
    t2 = np.pi * rng.uniform(0, 1, n2)          # lower moon
    moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    moon2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    data = np.concatenate([moon1, moon2], axis=0)
    data += rng.normal(0, noise, data.shape)
    # Standardise to roughly zero-mean / unit-scale, which the diffusion
    # maths assumes (the endpoint of the forward process is standard normal).
    data = (data - data.mean(0)) / data.std(0)
    return torch.tensor(data, dtype=torch.float32)


data = make_two_moons(4000)


# ---------------------------------------------------------------------------
# 2. The noise schedule
# ---------------------------------------------------------------------------
# beta_t is how much noise we add at step t. A small linear ramp is the
# original choice. From it we precompute alpha and its cumulative product,
# which is all the forward process needs.
T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)          # acp_t
sqrt_acp = torch.sqrt(alphas_cumprod)
sqrt_one_minus_acp = torch.sqrt(1.0 - alphas_cumprod)


# ---------------------------------------------------------------------------
# 3. The noise-prediction network
# ---------------------------------------------------------------------------
# The network must know *which* noise level it is looking at, so we feed the
# timestep t through a sinusoidal embedding (the same trick as Transformer
# positional encodings) before concatenating it with the noisy point.
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: integer timesteps, shape (B,). Returns (B, dim).
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsilonModel(nn.Module):
    """Predicts the noise added to a point, given the point and the timestep."""

    def __init__(self, t_dim=64, hidden=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        temb = self.time_embed(t)
        return self.net(torch.cat([x, temb], dim=-1))


model = EpsilonModel().to(device)


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
# For each step we: pick random data points, pick random timesteps, use the
# closed-form forward process to jump straight to x_t, and train the network
# to recover the exact noise we injected. That single objective is all a DDPM
# needs.
def q_sample(x0, t, noise):
    """Forward process: produce x_t from x0 in one shot."""
    return sqrt_acp[t][:, None] * x0 + sqrt_one_minus_acp[t][:, None] * noise


opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch_size = 512
steps = 15000

for step in range(steps):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    x0 = data[idx]
    t = torch.randint(0, T, (batch_size,))            # random noise levels
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    pred = model(x_t, t)
    loss = loss_fn(pred, noise)                        # predict the noise
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 3000 == 0:
        print(f"step {step + 1:>6d}/{steps}   loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 5. Sampling (the reverse process)
# ---------------------------------------------------------------------------
# Start from pure Gaussian noise and walk backwards from t = T-1 to 0. At each
# step, use the predicted noise to estimate a slightly cleaner point, then
# (except at the very last step) add a little fresh noise back -- this is
# standard DDPM ancestral sampling.
@torch.no_grad()
def sample(n, record_every=None):
    x = torch.randn(n, 2)
    snapshots = []
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps = model(x, t_batch)
        beta_t = betas[t]
        alpha_t = alphas[t]
        acp_t = alphas_cumprod[t]
        # Posterior mean of x_{t-1} given x_t and the predicted noise.
        mean = (x - beta_t / torch.sqrt(1 - acp_t) * eps) / torch.sqrt(alpha_t)
        if t > 0:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        else:
            x = mean
        if record_every is not None and (t % record_every == 0 or t == T - 1):
            snapshots.append((t, x.clone()))
    return x, snapshots


generated, snapshots = sample(4000, record_every=T // 4)


# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 4.6))

ax0 = fig.add_subplot(1, 3, 1)
ax0.scatter(*data.numpy().T, s=4, alpha=0.4, color="#1f4e79")
ax0.set_title("Target distribution\n(real two-moons data)")

ax1 = fig.add_subplot(1, 3, 2)
ax1.scatter(*generated.numpy().T, s=4, alpha=0.4, color="#b5651d")
ax1.set_title("Generated by the DDPM\n(starting from pure noise)")

# Reverse trajectory: a handful of snapshots from noise (faint) to samples.
ax2 = fig.add_subplot(1, 3, 3)
snaps = sorted(snapshots, key=lambda s: -s[0])         # high t (noise) first
cmap = plt.cm.viridis
for i, (t, pts) in enumerate(snaps):
    ax2.scatter(*pts.numpy().T, s=3, alpha=0.35,
                color=cmap(i / max(1, len(snaps) - 1)),
                label=f"t={t}")
ax2.set_title("Reverse process\n(noise -> data, dark to light)")
ax2.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.9)

for a in (ax0, ax1, ax2):
    a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")

fig.suptitle("A DDPM trained from scratch learns to turn Gaussian noise into "
             "the target two-moons distribution.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figure.png", dpi=130)
print("\nSaved figure.png")
