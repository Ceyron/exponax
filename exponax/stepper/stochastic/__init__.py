"""
Stochastic stepper sub-package for Exponax.

Public API
----------
StochasticAllenCahn
    Exponential Euler-Maruyama integrator for the stochastic
    Allen-Cahn SPDE with additive or multiplicative (gradient-type)
    noise.
"""

from ._stochastic_allen_cahn import StochasticAllenCahn

__all__ = [
    "StochasticAllenCahn",
]
