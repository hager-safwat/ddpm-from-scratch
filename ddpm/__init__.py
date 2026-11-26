"""A minimal DDPM (Ho et al., 2020) on 2-D data, written from scratch."""

from .data import make_two_moons
from .diffusion import mean_nn_distance, sample, train
from .model import EpsilonModel, SinusoidalTimeEmbedding
from .schedule import NoiseSchedule

__version__ = "0.2.0"

__all__ = [
    "EpsilonModel",
    "NoiseSchedule",
    "SinusoidalTimeEmbedding",
    "make_two_moons",
    "mean_nn_distance",
    "sample",
    "train",
]
