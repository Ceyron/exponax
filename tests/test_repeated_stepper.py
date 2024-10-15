import jax
import pytest

import exponax as ex


def test_repeated_stepper():
    DOMAIN_EXTENT = 1.0
    NUM_POINTS = 81
    DT = 0.1
    NUM_REPEATS = 10

    burgers_stepper = ex.stepper.Burgers(1, DOMAIN_EXTENT, NUM_POINTS, DT)

    burgers_stepper_repeated = ex.RepeatedStepper(burgers_stepper, NUM_REPEATS)

    burgers_stepper_repeated_manually = ex.repeat(burgers_stepper, NUM_REPEATS)

    u_0 = ex.ic.RandomTruncatedFourierSeries(1, max_one=True)(
        NUM_POINTS, key=jax.random.PRNGKey(0)
    )

    u_final = burgers_stepper_repeated(u_0)
    u_final_manually = burgers_stepper_repeated_manually(u_0)

    # Need a looser rel tolerance because Burgers is a decaying phenomenon,
    # hence the expected/reference state has low magnitude after 10 steps.
    assert u_final == pytest.approx(u_final_manually, rel=1e-3)
