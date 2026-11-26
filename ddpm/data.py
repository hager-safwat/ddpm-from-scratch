"""Target distribution: two interleaving half-moons in 2-D."""

import numpy as np
import torch


def make_two_moons(n, rng, noise=0.06):
    """Sample n points from two interleaving half-moons, standardised.

    It is a classic "hard" 2-D shape: two curved, non-convex clusters. If the
    model reproduces it, it has learned a non-trivial distribution, not a blob.

    Parameters
    ----------
    n : int
        Number of points (split evenly between the two moons).
    rng : np.random.Generator
        Source of randomness, passed in explicitly for reproducibility.
    noise : float
        Standard deviation of the Gaussian jitter added before standardising.

    Returns
    -------
    torch.Tensor, shape (n, 2), float32, roughly zero mean and unit scale.
    """
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * rng.uniform(0, 1, n1)          # upper moon
    t2 = np.pi * rng.uniform(0, 1, n2)          # lower moon
    moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    moon2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    data = np.concatenate([moon1, moon2], axis=0)
    data += rng.normal(0, noise, data.shape)
    # Standardise: the end point of the forward process is a standard normal,
    # so the data should live on a comparable scale.
    data = (data - data.mean(0)) / data.std(0)
    return torch.tensor(data, dtype=torch.float32)
