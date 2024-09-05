import jax
import matplotlib.pyplot as plt

import exponax as ex


def test_plot_state_1d():
    state = jax.random.normal(
        jax.random.PRNGKey(0),
        (10, 100),
    )

    fig = ex.viz.plot_state_1d(state)
    plt.close(fig)


def test_plot_spatio_temporal():
    trj = jax.random.normal(jax.random.PRNGKey(0), (100, 1, 64))

    fig = ex.viz.plot_spatio_temporal(trj)
    plt.close(fig)


def test_plot_state_2d():
    state = jax.random.normal(jax.random.PRNGKey(0), (1, 100, 100))

    fig = ex.viz.plot_state_2d(state)
    plt.close(fig)


# # Requires a GPU and therefore cannot easily be tested on GitHub Actions

# def test_plot_state_3d():
#     state = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 32))

#     fig = ex.viz.plot_state_3d(state)
#     plt.close(fig)
