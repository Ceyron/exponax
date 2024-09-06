# Validation

Since Fourier-pseudo spectral ETDRK methods are exact for linear bandlimited
problems (on periodic domains), this can be automatically validated to machine precision and is done in `tests/test_validation.py`. This folder contains additional validation notebooks for specific problems.

Additionally, run the script `qualitative rollouts` to produce a set
visualizations (1D -> spatio-temporal, 2D & 3D -> animations) of the
trajectories of the pre-built solvers. References to this can be found at:
https://github.com/Ceyron/exponax_qualitative_rollouts