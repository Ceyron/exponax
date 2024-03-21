"""
Collection of routines to create (randomized) initial conditions. Each type of
initial condition has a random generator. If the initial condition can expressed
in closed form, there is also a parameterized function for it.

By default, all IC generators are **single-channel**. They can be queried in
arbitrary number of spatial dimensions but always only return a single channel.
To create a multi-channel IC, e.g., for the Burgers equation, use the
`MultiChannelIC` and `RandomMultiChannelICGenerator` classes.
"""
from ._base_ic import BaseIC, BaseRandomICGenerator
from ._clamping import ClampingICGenerator
from ._diffused_noise import DiffusedNoise
from ._discontinuities import Discontinuities, RandomDiscontinuities
from ._gaussian_blob import GaussianBlobs, RandomGaussianBlobs
from ._gaussian_random_field import GaussianRandomField
from ._multi_channel import MultiChannelIC, RandomMultiChannelICGenerator
from ._scaled import ScaledIC, ScaledICGenerator
from ._sine_waves_1d import RandomSineWaves1d, SineWaves1d
from ._truncated_fourier_series import RandomTruncatedFourierSeries

__all__ = [
    "BaseIC",
    "BaseRandomICGenerator",
    "ClampingICGenerator",
    "Discontinuities",
    "DiffusedNoise",
    "GaussianBlobs",
    "GaussianRandomField",
    "MultiChannelIC",
    "RandomDiscontinuities",
    "RandomGaussianBlobs",
    "RandomMultiChannelICGenerator",
    "RandomTruncatedFourierSeries",
    "ScaledIC",
    "ScaledICGenerator",
    "SineWaves1d",
    "RandomSineWaves1d",
]
