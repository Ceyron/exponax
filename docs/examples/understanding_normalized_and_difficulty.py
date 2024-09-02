"""
This is a streamlit app.
"""
import jax
import matplotlib.pyplot as plt
import streamlit as st

import exponax as ex

jax.config.update("jax_platform_name", "cpu")

with st.sidebar:
    num_points = st.slider("Number of points", 16, 256, 48)
    num_steps = st.slider("Number of steps", 1, 300, 50)
    num_modes_init = st.slider("Number of modes in the initial condition", 1, 40, 5)
    num_substeps = st.slider("Number of substeps", 1, 100, 1)

    use_difficulty = st.toggle("Use difficulty", value=True)

    overall_scale = st.slider("Overall scale", 0.1, 50.0, 1.0)

    a_0_cols = st.columns(3)
    with a_0_cols[0]:
        a_0_mantissa = st.slider("a_0 mantissa", 0.0, 10.0, 0.0)
    with a_0_cols[1]:
        a_0_exponent = st.slider("a_0 exponent", -5, 5, 0)
    with a_0_cols[2]:
        a_0_sign = st.select_slider("a_0 sign", options=["-", "+"])
    a_0 = float(f"{a_0_sign}{a_0_mantissa}e{a_0_exponent}")

    a_1_cols = st.columns(3)
    with a_1_cols[0]:
        a_1_mantissa = st.slider("a_1 mantissa", 0.0, 10.0, 0.1)
    with a_1_cols[1]:
        a_1_exponent = st.slider("a_1 exponent", -5, 5, 0)
    with a_1_cols[2]:
        a_1_sign = st.select_slider("a_1 sign", options=["-", "+"])
    a_1 = float(f"{a_1_sign}{a_1_mantissa}e{a_1_exponent}")

    a_2_cols = st.columns(3)
    with a_2_cols[0]:
        a_2_mantissa = st.slider("a_2 mantissa", 0.0, 10.0, 0.0)
    with a_2_cols[1]:
        a_2_exponent = st.slider("a_2 exponent", -5, 5, 0)
    with a_2_cols[2]:
        a_2_sign = st.select_slider("a_2 sign", options=["-", "+"])
    a_2 = float(f"{a_2_sign}{a_2_mantissa}e{a_2_exponent}")

    a_3_cols = st.columns(3)
    with a_3_cols[0]:
        a_3_mantissa = st.slider("a_3 mantissa", 0.0, 10.0, 0.0)
    with a_3_cols[1]:
        a_3_exponent = st.slider("a_3 exponent", -5, 5, 0)
    with a_3_cols[2]:
        a_3_sign = st.select_slider("a_3 sign", options=["-", "+"])
    a_3 = float(f"{a_3_sign}{a_3_mantissa}e{a_3_exponent}")

    a_4_cols = st.columns(3)
    with a_4_cols[0]:
        a_4_mantissa = st.slider("a_4 mantissa", 0.0, 10.0, 0.0)
    with a_4_cols[1]:
        a_4_exponent = st.slider("a_4 exponent", -5, 5, 0)
    with a_4_cols[2]:
        a_4_sign = st.select_slider("a_4 sign", options=["-", "+"])
    a_4 = float(f"{a_4_sign}{a_4_mantissa}e{a_4_exponent}")

    b_0_cols = st.columns(3)
    with b_0_cols[0]:
        b_0_mantissa = st.slider("b_0 mantissa", 0.0, 10.0, 0.0)
    with b_0_cols[1]:
        b_0_exponent = st.slider("b_0 exponent", -5, 5, 0)
    with b_0_cols[2]:
        b_0_sign = st.select_slider("b_0 sign", options=["-", "+"])
    b_0 = float(f"{b_0_sign}{b_0_mantissa}e{b_0_exponent}")

    b_1_cols = st.columns(3)
    with b_1_cols[0]:
        b_1_mantissa = st.slider("b_1 mantissa", 0.0, 10.0, 0.0)
    with b_1_cols[1]:
        b_1_exponent = st.slider("b_1 exponent", -5, 5, 0)
    with b_1_cols[2]:
        b_1_sign = st.select_slider("b_1 sign", options=["-", "+"])
    b_1 = float(f"{b_1_sign}{b_1_mantissa}e{b_1_exponent}")

    b_2_cols = st.columns(3)
    with b_2_cols[0]:
        b_2_mantissa = st.slider("b_2 mantissa", 0.0, 10.0, 0.0)
    with b_2_cols[1]:
        b_2_exponent = st.slider("b_2 exponent", -5, 5, 0)
    with b_2_cols[2]:
        b_2_sign = st.select_slider("b_2 sign", options=["-", "+"])
    b_2 = float(f"{b_2_sign}{b_2_mantissa}e{b_2_exponent}")

    # a_0 = st.slider("a_0", -10.0, 10.0, 0.0)
    # a_1 = st.slider("a_1", -10.0, 10.0, 0.1)
    # a_2 = st.slider("a_2", -10.0, 10.0, 0.0)
    # a_3 = st.slider("a_3", -10.0, 10.0, 0.0)
    # a_4 = st.slider("a_4", -10.0, 10.0, 0.0)
    # b_0 = st.slider("b_0", -10.0, 10.0, 0.0)
    # b_1 = st.slider("b_1", -10.0, 10.0, 0.0)
    # b_2 = st.slider("b_2", -10.0, 10.0, 0.0)

linear_tuple = (a_0, a_1, a_2, a_3, a_4)
nonlinear_tuple = (b_0, b_1, b_2)

linear_tuple = tuple([overall_scale * x for x in linear_tuple])
nonlinear_tuple = tuple([overall_scale * x for x in nonlinear_tuple])

if use_difficulty:
    stepper = ex.RepeatedStepper(
        ex.normalized.DifficultyGeneralNonlinearStepper(
            1,
            num_points,
            linear_difficulties=tuple(x / num_substeps for x in linear_tuple),
            nonlinear_difficulties=tuple(x / num_substeps for x in nonlinear_tuple),
        ),
        num_substeps,
    )
else:
    stepper = ex.RepeatedStepper(
        ex.normalized.NormalizedGeneralNonlinearStepper(
            1,
            num_points,
            normalized_coefficients_linear=tuple(
                x / num_substeps for x in linear_tuple
            ),
            normalized_coefficients_nonlinear=tuple(
                x / num_substeps for x in nonlinear_tuple
            ),
        ),
        num_substeps,
    )

ic_gen = ex.ic.RandomSineWaves1d(1, cutoff=num_modes_init, max_one=True)
u_0 = ic_gen(num_points, key=jax.random.PRNGKey(0))

trj = ex.rollout(stepper, num_steps, include_init=True)(u_0)

v_range = st.slider("Colorbar range", 0.1, 10.0, 1.0)

fig, ax = plt.subplots()
ax.imshow(
    trj[:, 0, :].T,
    aspect="auto",
    vmin=-v_range,
    vmax=v_range,
    cmap="RdBu_r",
    origin="lower",
)

st.write(f"Linear: {linear_tuple}")
st.write(f"Nonlinear: {nonlinear_tuple}")

st.pyplot(fig)
