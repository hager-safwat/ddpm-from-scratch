"""Noise schedule and the closed-form forward process."""

import torch


class NoiseSchedule:
    """Linear beta schedule, with every quantity the forward and reverse
    processes need precomputed.

    The original DDPM uses betas from 1e-4 to 0.02 over T = 1000 steps. With
    fewer steps the same betas do not destroy the signal: for T = 200 the
    final alpha_bar is about 0.13, so x_T still contains 36 % of x0 while
    sampling starts from pure noise. ``scale_to_T=True`` (the default)
    multiplies the betas by 1000 / T, which keeps the total amount of noise
    the same as in the 1000-step schedule (final alpha_bar about 4e-5).

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    beta_start, beta_end : float
        Betas of the reference 1000-step schedule.
    scale_to_T : bool
        Rescale the betas by 1000 / T as explained above.
    """

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, scale_to_T=True):
        scale = 1000.0 / T if scale_to_T else 1.0
        self.T = T
        # For very small T the rescaled betas would reach 1 (alpha = 0, and
        # sampling would divide by zero), so they are capped at 0.999 as in
        # Nichol and Dhariwal (2021). For T = 200 the cap is not active.
        self.betas = torch.linspace(beta_start * scale, beta_end * scale, T).clamp(max=0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)      # alpha_bar_t
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """Forward process in one shot:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.

        x0, noise: (B, D). t: integer steps, shape (B,).
        """
        return self.sqrt_acp[t][:, None] * x0 + self.sqrt_one_minus_acp[t][:, None] * noise
