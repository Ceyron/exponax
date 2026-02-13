"""
Comprehensive tests for the exponax.viz visualization module.

Tests cover:
- Static plot functions (1D, spatio-temporal, 2D) with all argument variations
- Faceted static plot functions with facet_over_channels True/False
- Animation functions (1D, spatio-temporal, 2D) with all argument variations
- Faceted animation functions with facet_over_channels True/False
- Input validation (wrong ndim raises ValueError)
- NotImplementedError for unimplemented functions
- Volume utilities (zigzag_alpha, triangle_wave) that don't require GPU

3D functions (plot_state_3d, plot_state_3d_facet, animate_state_3d,
animate_state_3d_facet, plot_spatio_temporal_2d, plot_spatio_temporal_2d_facet,
volume_render_state_3d) are skipped because they require the vape4d package.
"""

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

import exponax as ex
from exponax.viz._volume import triangle_wave, zigzag_alpha

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(0)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after every test to prevent memory leaks."""
    yield
    plt.close("all")


# ===========================================================================
# 1. Static plot functions
# ===========================================================================


# ---- plot_state_1d --------------------------------------------------------


class TestPlotState1d:
    def test_basic_single_channel(self):
        state = jax.random.normal(KEY, (1, 64))
        fig = ex.viz.plot_state_1d(state)
        assert isinstance(fig, Figure)

    def test_basic_multi_channel(self):
        state = jax.random.normal(KEY, (5, 64))
        fig = ex.viz.plot_state_1d(state)
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        state = jax.random.normal(KEY, (3, 64))
        fig = ex.viz.plot_state_1d(state, labels=["u", "v", "w"])
        assert isinstance(fig, Figure)

    def test_custom_vlim(self):
        state = jax.random.normal(KEY, (1, 64))
        fig = ex.viz.plot_state_1d(state, vlim=(-5.0, 5.0))
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        state = jax.random.normal(KEY, (1, 64))
        fig = ex.viz.plot_state_1d(state, domain_extent=2 * jnp.pi)
        assert isinstance(fig, Figure)

    def test_with_ax(self):
        state = jax.random.normal(KEY, (2, 64))
        fig, ax = plt.subplots()
        result = ex.viz.plot_state_1d(state, ax=ax)
        # When ax is provided, returns plot objects, not figure
        assert result is not None
        assert not isinstance(result, Figure)

    def test_custom_axis_labels(self):
        state = jax.random.normal(KEY, (1, 64))
        fig = ex.viz.plot_state_1d(state, xlabel="x", ylabel="u(x)")
        assert isinstance(fig, Figure)

    def test_wrong_ndim_raises(self):
        state_1d = jax.random.normal(KEY, (64,))
        with pytest.raises(ValueError, match="two-axis"):
            ex.viz.plot_state_1d(state_1d)

        state_3d = jax.random.normal(KEY, (1, 64, 64))
        with pytest.raises(ValueError, match="two-axis"):
            ex.viz.plot_state_1d(state_3d)


# ---- plot_spatio_temporal -------------------------------------------------


class TestPlotSpatioTemporal:
    def test_basic(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj)
        assert isinstance(fig, Figure)

    def test_custom_vlim(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj, vlim=(-3.0, 3.0))
        assert isinstance(fig, Figure)

    def test_custom_cmap(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj, cmap="viridis")
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj, domain_extent=2 * jnp.pi)
        assert isinstance(fig, Figure)

    def test_with_dt(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj, dt=0.1)
        assert isinstance(fig, Figure)

    def test_with_dt_and_include_init(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal(trj, dt=0.1, include_init=True)
        assert isinstance(fig, Figure)

    def test_with_ax(self):
        trj = jax.random.normal(KEY, (20, 1, 64))
        fig, ax = plt.subplots()
        result = ex.viz.plot_spatio_temporal(trj, ax=ax)
        assert result is not None
        assert not isinstance(result, Figure)

    def test_multi_channel_uses_first(self):
        """Multi-channel input should work (only first channel is plotted)."""
        trj = jax.random.normal(KEY, (20, 3, 64))
        fig = ex.viz.plot_spatio_temporal(trj)
        assert isinstance(fig, Figure)

    def test_wrong_ndim_raises(self):
        trj_2d = jax.random.normal(KEY, (20, 64))
        with pytest.raises(ValueError):
            ex.viz.plot_spatio_temporal(trj_2d)


# ---- plot_state_2d --------------------------------------------------------


class TestPlotState2d:
    def test_basic(self):
        state = jax.random.normal(KEY, (1, 32, 32))
        fig = ex.viz.plot_state_2d(state)
        assert isinstance(fig, Figure)

    def test_custom_vlim(self):
        state = jax.random.normal(KEY, (1, 32, 32))
        fig = ex.viz.plot_state_2d(state, vlim=(-2.0, 2.0))
        assert isinstance(fig, Figure)

    def test_custom_cmap(self):
        state = jax.random.normal(KEY, (1, 32, 32))
        fig = ex.viz.plot_state_2d(state, cmap="coolwarm")
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        state = jax.random.normal(KEY, (1, 32, 32))
        fig = ex.viz.plot_state_2d(state, domain_extent=1.0)
        assert isinstance(fig, Figure)

    def test_with_ax(self):
        state = jax.random.normal(KEY, (1, 32, 32))
        fig, ax = plt.subplots()
        result = ex.viz.plot_state_2d(state, ax=ax)
        assert result is not None
        assert not isinstance(result, Figure)

    def test_multi_channel_uses_first(self):
        """Multi-channel input should work (wraps BC on full state, imshow on first)."""
        state = jax.random.normal(KEY, (3, 32, 32))
        fig = ex.viz.plot_state_2d(state)
        assert isinstance(fig, Figure)

    def test_wrong_ndim_raises(self):
        state_2d = jax.random.normal(KEY, (32, 32))
        with pytest.raises(ValueError, match="three-axis"):
            ex.viz.plot_state_2d(state_2d)


# ===========================================================================
# 2. Faceted static plot functions
# ===========================================================================


# ---- plot_state_1d_facet --------------------------------------------------


class TestPlotState1dFacet:
    def test_basic(self):
        states = jax.random.normal(KEY, (4, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_multi_channel(self):
        states = jax.random.normal(KEY, (4, 3, 64))
        fig = ex.viz.plot_state_1d_facet(states, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        states = jax.random.normal(KEY, (4, 2, 64))
        fig = ex.viz.plot_state_1d_facet(states, labels=["u", "v"], grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_with_titles(self):
        states = jax.random.normal(KEY, (4, 1, 64))
        fig = ex.viz.plot_state_1d_facet(
            states, titles=["a", "b", "c", "d"], grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_custom_vlim(self):
        states = jax.random.normal(KEY, (4, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, vlim=(-3.0, 3.0), grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        states = jax.random.normal(KEY, (4, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, domain_extent=2 * jnp.pi, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_custom_figsize(self):
        states = jax.random.normal(KEY, (4, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, grid=(2, 2), figsize=(8, 6))
        assert isinstance(fig, Figure)

    def test_fewer_states_than_grid(self):
        """Grid (2,2)=4 slots but only 2 states -> extra axes removed."""
        states = jax.random.normal(KEY, (2, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_single_subplot(self):
        """Grid (1,1)=1 slot with one state."""
        states = jax.random.normal(KEY, (1, 1, 64))
        fig = ex.viz.plot_state_1d_facet(states, grid=(1, 1))
        assert isinstance(fig, Figure)

    def test_wrong_ndim_raises(self):
        states_2d = jax.random.normal(KEY, (1, 64))
        with pytest.raises(ValueError, match="three-axis"):
            ex.viz.plot_state_1d_facet(states_2d)


# ---- plot_spatio_temporal_facet -------------------------------------------


class TestPlotSpatioTemporalFacet:
    def test_facet_over_channels(self):
        """facet_over_channels=True: input (T, C, N)."""
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=True, grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_facet_over_batches(self):
        """facet_over_channels=False: input (B, T, 1, N)."""
        trjs = jax.random.normal(KEY, (4, 20, 1, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=False, grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_with_titles(self):
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs,
            facet_over_channels=True,
            titles=["ch0", "ch1", "ch2", "ch3"],
            grid=(2, 2),
        )
        assert isinstance(fig, Figure)

    def test_with_dt(self):
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=True, dt=0.05, grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=True, domain_extent=1.0, grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_with_include_init(self):
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs,
            facet_over_channels=True,
            dt=0.1,
            include_init=True,
            grid=(2, 2),
        )
        assert isinstance(fig, Figure)

    def test_custom_cmap(self):
        trjs = jax.random.normal(KEY, (20, 4, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=True, cmap="viridis", grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_single_subplot_channel(self):
        """Single channel faceted: (T, 1, N)."""
        trjs = jax.random.normal(KEY, (20, 1, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=True, grid=(1, 1)
        )
        assert isinstance(fig, Figure)

    def test_single_subplot_batch(self):
        """Single batch faceted: (1, T, 1, N)."""
        trjs = jax.random.normal(KEY, (1, 20, 1, 64))
        fig = ex.viz.plot_spatio_temporal_facet(
            trjs, facet_over_channels=False, grid=(1, 1)
        )
        assert isinstance(fig, Figure)

    def test_wrong_ndim_channels_mode_raises(self):
        trjs_4d = jax.random.normal(KEY, (4, 20, 1, 64))
        with pytest.raises(ValueError, match="three-axis"):
            ex.viz.plot_spatio_temporal_facet(trjs_4d, facet_over_channels=True)

    def test_wrong_ndim_batch_mode_raises(self):
        trjs_3d = jax.random.normal(KEY, (20, 4, 64))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_spatio_temporal_facet(trjs_3d, facet_over_channels=False)


# ---- plot_state_2d_facet --------------------------------------------------


class TestPlotState2dFacet:
    def test_facet_over_channels(self):
        """facet_over_channels=True: input (C, N, N)."""
        states = jax.random.normal(KEY, (4, 32, 32))
        fig = ex.viz.plot_state_2d_facet(states, facet_over_channels=True, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_facet_over_batches(self):
        """facet_over_channels=False: input (B, 1, N, N)."""
        states = jax.random.normal(KEY, (4, 1, 32, 32))
        fig = ex.viz.plot_state_2d_facet(states, facet_over_channels=False, grid=(2, 2))
        assert isinstance(fig, Figure)

    def test_with_titles(self):
        states = jax.random.normal(KEY, (4, 32, 32))
        fig = ex.viz.plot_state_2d_facet(
            states,
            facet_over_channels=True,
            titles=["ch0", "ch1", "ch2", "ch3"],
            grid=(2, 2),
        )
        assert isinstance(fig, Figure)

    def test_custom_cmap(self):
        states = jax.random.normal(KEY, (4, 32, 32))
        fig = ex.viz.plot_state_2d_facet(
            states, facet_over_channels=True, cmap="coolwarm", grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_with_domain_extent(self):
        states = jax.random.normal(KEY, (4, 32, 32))
        fig = ex.viz.plot_state_2d_facet(
            states,
            facet_over_channels=True,
            domain_extent=2 * jnp.pi,
            grid=(2, 2),
        )
        assert isinstance(fig, Figure)

    def test_custom_vlim(self):
        states = jax.random.normal(KEY, (4, 32, 32))
        fig = ex.viz.plot_state_2d_facet(
            states, facet_over_channels=True, vlim=(-2.0, 2.0), grid=(2, 2)
        )
        assert isinstance(fig, Figure)

    def test_single_subplot_channel(self):
        states = jax.random.normal(KEY, (1, 32, 32))
        fig = ex.viz.plot_state_2d_facet(states, facet_over_channels=True, grid=(1, 1))
        assert isinstance(fig, Figure)

    def test_single_subplot_batch(self):
        states = jax.random.normal(KEY, (1, 1, 32, 32))
        fig = ex.viz.plot_state_2d_facet(states, facet_over_channels=False, grid=(1, 1))
        assert isinstance(fig, Figure)

    def test_wrong_ndim_channels_mode_raises(self):
        states_4d = jax.random.normal(KEY, (4, 1, 32, 32))
        with pytest.raises(ValueError, match="three-axis"):
            ex.viz.plot_state_2d_facet(states_4d, facet_over_channels=True)

    def test_wrong_ndim_batch_mode_raises(self):
        states_3d = jax.random.normal(KEY, (4, 32, 32))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_state_2d_facet(states_3d, facet_over_channels=False)


# ===========================================================================
# 3. Animation functions
# ===========================================================================


# ---- animate_state_1d -----------------------------------------------------


class TestAnimateState1d:
    def test_basic_single_channel(self):
        trj = jax.random.normal(KEY, (10, 1, 64))
        ani = ex.viz.animate_state_1d(trj)
        assert isinstance(ani, FuncAnimation)

    def test_basic_multi_channel(self):
        trj = jax.random.normal(KEY, (10, 3, 64))
        ani = ex.viz.animate_state_1d(trj)
        assert isinstance(ani, FuncAnimation)

    def test_with_vlim(self):
        trj = jax.random.normal(KEY, (10, 1, 64))
        ani = ex.viz.animate_state_1d(trj, vlim=(-5.0, 5.0))
        assert isinstance(ani, FuncAnimation)

    def test_with_domain_extent(self):
        trj = jax.random.normal(KEY, (10, 1, 64))
        ani = ex.viz.animate_state_1d(trj, domain_extent=2 * jnp.pi)
        assert isinstance(ani, FuncAnimation)

    def test_with_dt(self):
        trj = jax.random.normal(KEY, (10, 1, 64))
        ani = ex.viz.animate_state_1d(trj, dt=0.1)
        assert isinstance(ani, FuncAnimation)

    def test_with_include_init(self):
        trj = jax.random.normal(KEY, (10, 1, 64))
        ani = ex.viz.animate_state_1d(trj, dt=0.1, include_init=True)
        assert isinstance(ani, FuncAnimation)

    def test_frame_count(self):
        trj = jax.random.normal(KEY, (15, 1, 64))
        ani = ex.viz.animate_state_1d(trj)
        assert ani._save_count == 15


# ---- animate_spatio_temporal ----------------------------------------------


class TestAnimateSpatioTemporal:
    def test_basic(self):
        """Input shape: (S, T, 1, N) - S outer time steps."""
        trjs = jax.random.normal(KEY, (5, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs)
        assert isinstance(ani, FuncAnimation)

    def test_with_vlim(self):
        trjs = jax.random.normal(KEY, (5, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs, vlim=(-2.0, 2.0))
        assert isinstance(ani, FuncAnimation)

    def test_with_cmap(self):
        trjs = jax.random.normal(KEY, (5, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs, cmap="viridis")
        assert isinstance(ani, FuncAnimation)

    def test_with_domain_extent(self):
        trjs = jax.random.normal(KEY, (5, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs, domain_extent=1.0)
        assert isinstance(ani, FuncAnimation)

    def test_with_dt_and_include_init(self):
        trjs = jax.random.normal(KEY, (5, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs, dt=0.05, include_init=True)
        assert isinstance(ani, FuncAnimation)

    def test_frame_count(self):
        trjs = jax.random.normal(KEY, (8, 20, 1, 64))
        ani = ex.viz.animate_spatio_temporal(trjs)
        assert ani._save_count == 8

    def test_wrong_ndim_raises(self):
        trjs_3d = jax.random.normal(KEY, (20, 1, 64))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.animate_spatio_temporal(trjs_3d)


# ---- animate_state_2d -----------------------------------------------------


class TestAnimateState2d:
    def test_basic(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj)
        assert isinstance(ani, FuncAnimation)

    def test_with_vlim(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj, vlim=(-3.0, 3.0))
        assert isinstance(ani, FuncAnimation)

    def test_with_cmap(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj, cmap="inferno")
        assert isinstance(ani, FuncAnimation)

    def test_with_domain_extent(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj, domain_extent=2.0)
        assert isinstance(ani, FuncAnimation)

    def test_with_dt(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj, dt=0.1)
        assert isinstance(ani, FuncAnimation)

    def test_with_include_init(self):
        trj = jax.random.normal(KEY, (10, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj, dt=0.1, include_init=True)
        assert isinstance(ani, FuncAnimation)

    def test_frame_count(self):
        trj = jax.random.normal(KEY, (12, 1, 32, 32))
        ani = ex.viz.animate_state_2d(trj)
        assert ani._save_count == 12

    def test_wrong_ndim_raises(self):
        trj_3d = jax.random.normal(KEY, (1, 32, 32))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.animate_state_2d(trj_3d)


# ===========================================================================
# 4. Faceted animation functions
# ===========================================================================


# ---- animate_state_1d_facet -----------------------------------------------


class TestAnimateState1dFacet:
    def test_basic(self):
        """Input shape: (B, T, C, N)."""
        trj = jax.random.normal(KEY, (4, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_multi_channel(self):
        trj = jax.random.normal(KEY, (4, 10, 3, 64))
        ani = ex.viz.animate_state_1d_facet(trj, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_with_titles(self):
        trj = jax.random.normal(KEY, (4, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(
            trj, titles=["a", "b", "c", "d"], grid=(2, 2)
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_labels(self):
        trj = jax.random.normal(KEY, (4, 10, 2, 64))
        ani = ex.viz.animate_state_1d_facet(trj, labels=["u", "v"], grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_with_dt(self):
        trj = jax.random.normal(KEY, (4, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, dt=0.1, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_with_include_init(self):
        trj = jax.random.normal(KEY, (4, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, dt=0.1, include_init=True, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_with_domain_extent(self):
        trj = jax.random.normal(KEY, (4, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, domain_extent=2 * jnp.pi, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_single_subplot(self):
        trj = jax.random.normal(KEY, (1, 10, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, grid=(1, 1))
        assert isinstance(ani, FuncAnimation)

    def test_frame_count(self):
        trj = jax.random.normal(KEY, (4, 15, 1, 64))
        ani = ex.viz.animate_state_1d_facet(trj, grid=(2, 2))
        assert ani._save_count == 15

    def test_wrong_ndim_raises(self):
        trj_3d = jax.random.normal(KEY, (10, 1, 64))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.animate_state_1d_facet(trj_3d)


# ---- animate_state_2d_facet -----------------------------------------------


class TestAnimateState2dFacet:
    def test_facet_over_channels(self):
        """facet_over_channels=True: input (T, C, N, N)."""
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(trj, facet_over_channels=True, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_facet_over_batches(self):
        """facet_over_channels=False: input (B, T, 1, N, N)."""
        trj = jax.random.normal(KEY, (4, 10, 1, 16, 16))
        ani = ex.viz.animate_state_2d_facet(trj, facet_over_channels=False, grid=(2, 2))
        assert isinstance(ani, FuncAnimation)

    def test_with_titles_channels(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj,
            facet_over_channels=True,
            titles=["ch0", "ch1", "ch2", "ch3"],
            grid=(2, 2),
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_titles_batches(self):
        trj = jax.random.normal(KEY, (4, 10, 1, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj,
            facet_over_channels=False,
            titles=["b0", "b1", "b2", "b3"],
            grid=(2, 2),
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_vlim(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj, facet_over_channels=True, vlim=(-2.0, 2.0), grid=(2, 2)
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_cmap(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj, facet_over_channels=True, cmap="coolwarm", grid=(2, 2)
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_domain_extent(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj, facet_over_channels=True, domain_extent=1.0, grid=(2, 2)
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_dt(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj, facet_over_channels=True, dt=0.1, grid=(2, 2)
        )
        assert isinstance(ani, FuncAnimation)

    def test_with_include_init(self):
        trj = jax.random.normal(KEY, (10, 4, 16, 16))
        ani = ex.viz.animate_state_2d_facet(
            trj,
            facet_over_channels=True,
            dt=0.1,
            include_init=True,
            grid=(2, 2),
        )
        assert isinstance(ani, FuncAnimation)

    def test_single_channel_facet(self):
        trj = jax.random.normal(KEY, (10, 1, 16, 16))
        ani = ex.viz.animate_state_2d_facet(trj, facet_over_channels=True, grid=(1, 1))
        assert isinstance(ani, FuncAnimation)

    def test_single_batch_facet(self):
        trj = jax.random.normal(KEY, (1, 10, 1, 16, 16))
        ani = ex.viz.animate_state_2d_facet(trj, facet_over_channels=False, grid=(1, 1))
        assert isinstance(ani, FuncAnimation)

    def test_wrong_ndim_channels_mode_raises(self):
        trj_5d = jax.random.normal(KEY, (4, 10, 1, 16, 16))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.animate_state_2d_facet(trj_5d, facet_over_channels=True)

    def test_wrong_ndim_batch_mode_raises(self):
        trj_4d = jax.random.normal(KEY, (10, 4, 16, 16))
        with pytest.raises(ValueError, match="five-axis"):
            ex.viz.animate_state_2d_facet(trj_4d, facet_over_channels=False)


# ===========================================================================
# 5. NotImplementedError tests
# ===========================================================================


class TestNotImplemented:
    def test_animate_spatio_temporal_2d(self):
        with pytest.raises(NotImplementedError):
            ex.viz.animate_spatio_temporal_2d()

    def test_animate_spatio_temporal_facet_channels(self):
        trjs = jax.random.normal(KEY, (5, 20, 4, 64))
        with pytest.raises(NotImplementedError):
            ex.viz.animate_spatio_temporal_facet(trjs, facet_over_channels=True)

    def test_animate_spatio_temporal_facet_batches(self):
        trjs = jax.random.normal(KEY, (4, 5, 20, 1, 64))
        with pytest.raises(NotImplementedError):
            ex.viz.animate_spatio_temporal_facet(trjs, facet_over_channels=False)

    def test_animate_spatio_temporal_2d_facet(self):
        with pytest.raises(NotImplementedError):
            ex.viz.animate_spatio_temporal_2d_facet()


# ===========================================================================
# 6. Volume utilities (no GPU required)
# ===========================================================================


class TestZigzagAlpha:
    def test_with_linear_segmented_colormap(self):
        cmap = plt.get_cmap("RdBu_r")
        assert isinstance(cmap, LinearSegmentedColormap)
        modified = zigzag_alpha(cmap)
        assert isinstance(modified, LinearSegmentedColormap)
        assert "alpha" in modified._segmentdata

    def test_with_listed_colormap(self):
        # Create a simple ListedColormap
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cmap = ListedColormap(colors, name="test_listed")
        modified = zigzag_alpha(cmap)
        assert isinstance(modified, ListedColormap)
        # Each color should now have an alpha channel appended
        for c in modified.colors:
            assert len(c) == 4

    def test_with_min_alpha(self):
        cmap = plt.get_cmap("RdBu_r")
        modified = zigzag_alpha(cmap, min_alpha=0.3)
        assert isinstance(modified, LinearSegmentedColormap)

    def test_invalid_cmap_type_raises(self):
        with pytest.raises(TypeError, match="ListedColormap.*LinearSegmentedColormap"):
            zigzag_alpha("not_a_colormap")


class TestTriangleWave:
    def test_basic_values(self):
        # At x=0, triangle_wave should be 0
        assert float(triangle_wave(0.0, 1.0)) == pytest.approx(0.0, abs=1e-6)

    def test_midpoint_at_quarter_period(self):
        # At x=p/4, triangle_wave should be 0.5
        assert float(triangle_wave(0.25, 1.0)) == pytest.approx(0.5, abs=1e-6)

    def test_peak_at_half_period(self):
        # At x=p/2, triangle_wave should be 1.0 (peak)
        assert float(triangle_wave(0.5, 1.0)) == pytest.approx(1.0, abs=1e-6)

    def test_periodicity(self):
        x = jnp.array(0.3)
        p = 1.0
        val1 = triangle_wave(x, p)
        val2 = triangle_wave(x + p, p)
        assert float(val1) == pytest.approx(float(val2), abs=1e-6)

    def test_vectorized(self):
        x = jnp.linspace(0, 2, 100)
        result = triangle_wave(x, 1.0)
        assert result.shape == (100,)
        # All values should be in [0, 1]
        assert float(jnp.min(result)) >= -1e-6
        assert float(jnp.max(result)) <= 1.0 + 1e-6


# ===========================================================================
# 7. 3D functions - skip (require vape4d)
# ===========================================================================


@pytest.mark.skip(reason="Requires vape4d GPU package")
class TestPlotState3d:
    def test_basic(self):
        state = jax.random.normal(KEY, (1, 16, 16, 16))
        fig = ex.viz.plot_state_3d(state)
        assert isinstance(fig, Figure)


@pytest.mark.skip(reason="Requires vape4d GPU package")
class TestPlotState3dFacet:
    def test_facet_over_channels(self):
        states = jax.random.normal(KEY, (4, 16, 16, 16))
        fig = ex.viz.plot_state_3d_facet(states, facet_over_channels=True, grid=(2, 2))
        assert isinstance(fig, Figure)


# ===========================================================================
# 8. Input validation for 3D functions (ndim checks happen before vape4d)
# ===========================================================================


class TestInputValidation3d:
    def test_plot_state_3d_wrong_ndim(self):
        state_3d = jax.random.normal(KEY, (1, 16, 16))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_state_3d(state_3d)

    def test_plot_spatio_temporal_2d_wrong_ndim(self):
        trj_3d = jax.random.normal(KEY, (10, 1, 16))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_spatio_temporal_2d(trj_3d)

    def test_plot_state_3d_facet_wrong_ndim_channels(self):
        states = jax.random.normal(KEY, (1, 16, 16))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_state_3d_facet(states, facet_over_channels=True)

    def test_plot_state_3d_facet_wrong_ndim_batches(self):
        states = jax.random.normal(KEY, (1, 16, 16, 16))
        with pytest.raises(ValueError, match="five-axis"):
            ex.viz.plot_state_3d_facet(states, facet_over_channels=False)

    def test_animate_state_3d_wrong_ndim(self):
        trj = jax.random.normal(KEY, (10, 1, 16, 16))
        with pytest.raises(ValueError, match="five-axis"):
            ex.viz.animate_state_3d(trj)

    def test_animate_state_3d_facet_wrong_ndim_channels(self):
        trj = jax.random.normal(KEY, (10, 1, 16, 16))
        with pytest.raises(ValueError, match="five-axis"):
            ex.viz.animate_state_3d_facet(trj, facet_over_channels=True)

    def test_animate_state_3d_facet_wrong_ndim_batches(self):
        trj = jax.random.normal(KEY, (10, 1, 16, 16, 16))
        with pytest.raises(ValueError, match="six-axis"):
            ex.viz.animate_state_3d_facet(trj, facet_over_channels=False)

    def test_plot_spatio_temporal_2d_facet_wrong_ndim_channels(self):
        trjs = jax.random.normal(KEY, (4, 1, 16, 16, 16))
        with pytest.raises(ValueError, match="four-axis"):
            ex.viz.plot_spatio_temporal_2d_facet(trjs, facet_over_channels=True)

    def test_plot_spatio_temporal_2d_facet_wrong_ndim_batches(self):
        trjs = jax.random.normal(KEY, (10, 4, 16, 16))
        with pytest.raises(ValueError, match="five-axis"):
            ex.viz.plot_spatio_temporal_2d_facet(trjs, facet_over_channels=False)
