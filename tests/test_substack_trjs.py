import jax
import jax.numpy as jnp

import exponax as ex


def test_substack_trjs():
    simple_trj = jnp.array([1, 2, 3, 4, 5, 6])
    substacked_trjs = ex.stack_sub_trajectories(simple_trj, 3)
    correct_substacked_trjs = jnp.array(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
        ]
    )

    assert substacked_trjs.shape == correct_substacked_trjs.shape
    assert jnp.allclose(substacked_trjs, correct_substacked_trjs)


def test_substack_trjs_pytree_wrong_shapes():
    # The leave arrays have different leading dimensions.
    simple_pytree = {
        "a": jnp.array([1, 2, 3, 4, 5, 6]),
        "b": jnp.array(
            [
                1,
                2,
                3,
                4,
                5,
            ]
        ),
    }

    try:
        ex.stack_sub_trajectories(simple_pytree, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")


def test_substack_trjs_pytree():
    simple_pytree = {
        "a": jnp.array([1, 2, 3, 4, 5, 6]),
        "b": jnp.array([10, 11, 12, 13, 14, 15]),
    }

    substacked_pytree = ex.stack_sub_trajectories(simple_pytree, 3)

    correct_substacked_pytree = {
        "a": jnp.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
            ]
        ),
        "b": jnp.array(
            [
                [10, 11, 12],
                [11, 12, 13],
                [12, 13, 14],
                [13, 14, 15],
            ]
        ),
    }

    assert substacked_pytree.keys() == correct_substacked_pytree.keys()
    assert jnp.allclose(substacked_pytree["a"], correct_substacked_pytree["a"])
    assert jnp.allclose(substacked_pytree["b"], correct_substacked_pytree["b"])


def test_substack_trjs_more_rollout_than_possible():
    simple_trj = jnp.array([1, 2, 3, 4, 5, 6])

    try:
        ex.stack_sub_trajectories(simple_trj, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError.")


def test_substack_trjs_as_much_rollout_as_elements():
    simple_trj = jnp.array([1, 2, 3, 4, 5, 6])

    substacked_trjs = ex.stack_sub_trajectories(simple_trj, 6)

    correct_substacked_trjs = simple_trj.reshape((1, 6))

    assert substacked_trjs.shape == correct_substacked_trjs.shape
    assert jnp.allclose(substacked_trjs, correct_substacked_trjs)


def test_substack_trjs_higher_tensors():
    shape = (6, 1, 5)
    sample_trj = jax.random.normal(
        jax.random.PRNGKey(0),
        shape,
    )

    substacked_trjs = ex.stack_sub_trajectories(sample_trj, 3)

    corrected_substacked_shape = (4, 3, 1, 5)

    assert substacked_trjs.shape == corrected_substacked_shape
