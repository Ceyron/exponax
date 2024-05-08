"""
This is a streamlit app.
"""
import base64
import dataclasses
import io
import json
import random
from dataclasses import dataclass
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from IPython.display import DisplayObject
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

import exponax as ex

st.set_page_config(layout="wide")
jax.config.update("jax_platform_name", "cpu")

with st.sidebar:
    dimension_type = st.select_slider(
        "Number of Spatial Dimensions (ST=Spatio-Temporal plot)",
        options=["1d ST", "1d", "2d", "2d ST", "3d"],
    )
    num_points = st.slider("Number of points", 16, 256, 48)
    num_steps = st.slider("Number of steps", 1, 300, 50)
    num_modes_init = st.slider("Number of modes in the initial condition", 1, 40, 5)
    num_substeps = st.slider("Number of substeps", 1, 100, 1)

    overall_scale = st.slider("Overall scale", 0.1, 10.0, 1.0)

    preset_mode = st.selectbox(
        "Select a preset",
        [
            "None",
            "Burgers (single-channel hack)",
            "KdV (viscous, single-channel-hack)",
            "KdV (hyper-viscous, single-channel-hack)",
            "KS (conservative, single-channel-hack)",
            "KS (combustion)",
        ],
    )

    a_0_default_mantisssa = 0.0
    a_0_default_exponent = 0
    a_0_default_sign = "-"

    a_1_default_mantisssa = 0.0
    a_1_default_exponent = 0
    a_1_default_sign = "-"

    a_2_default_mantisssa = 0.0
    a_2_default_exponent = 0
    a_2_default_sign = "-"

    a_3_default_mantisssa = 0.0
    a_3_default_exponent = 0
    a_3_default_sign = "-"

    a_4_default_mantisssa = 0.0
    a_4_default_exponent = 0
    a_4_default_sign = "-"

    b_0_default_mantisssa = 0.0
    b_0_default_exponent = 0
    b_0_default_sign = "-"

    b_1_default_mantisssa = 0.0
    b_1_default_exponent = 0
    b_1_default_sign = "-"

    b_2_default_mantisssa = 0.0
    b_2_default_exponent = 0
    b_2_default_sign = "-"
    if preset_mode == "None":
        a_1_default_mantisssa = 0.1
        a_1_default_exponent = 0
        a_1_default_sign = "-"

    elif preset_mode == "Burgers (single-channel hack)":
        a_2_default_mantisssa = 0.15
        a_2_default_exponent = 1
        a_2_default_sign = "+"

        b_1_default_mantisssa = 0.2
        b_1_default_exponent = 1
        b_1_default_sign = "-"

    elif preset_mode == "KdV (viscous, single-channel-hack)":
        a_2_default_mantisssa = 0.2
        a_2_default_exponent = 1
        a_2_default_sign = "+"

        a_3_default_mantisssa = 0.14
        a_3_default_exponent = 2
        a_3_default_sign = "-"

        b_1_default_mantisssa = 0.2
        b_1_default_exponent = 1
        b_1_default_sign = "-"
    elif preset_mode == "KdV (hyper-viscous, single-channel-hack)":
        a_3_default_mantisssa = 0.14
        a_3_default_exponent = 2
        a_3_default_sign = "-"

        a_4_default_mantisssa = 0.9
        a_4_default_exponent = 1
        a_4_default_sign = "-"

        b_1_default_mantisssa = 0.2
        b_1_default_exponent = 1
        b_1_default_sign = "-"

    elif preset_mode == "KS (conservative, single-channel-hack)":
        a_2_default_mantisssa = 0.2
        a_2_default_exponent = 1
        a_2_default_sign = "-"

        a_4_default_mantisssa = 0.15
        a_4_default_exponent = 2
        a_4_default_sign = "-"

        b_1_default_mantisssa = 0.1
        b_1_default_exponent = 1
        b_1_default_sign = "-"
    elif preset_mode == "KS (combustion)":
        a_2_default_mantisssa = 0.12
        a_2_default_exponent = 1
        a_2_default_sign = "-"

        a_4_default_mantisssa = 0.15
        a_4_default_exponent = 2
        a_4_default_sign = "-"

        b_2_default_mantisssa = 0.6
        b_2_default_exponent = 1
        b_2_default_sign = "-"

    use_difficulty = st.toggle("Use difficulty", value=True)

    a_0_cols = st.columns(3)
    with a_0_cols[0]:
        a_0_mantissa = st.slider("a_0 mantissa", 0.0, 1.0, a_0_default_mantisssa)
    with a_0_cols[1]:
        a_0_exponent = st.slider("a_0 exponent", -5, 5, a_0_default_exponent)
    with a_0_cols[2]:
        a_0_sign = st.select_slider("a_0 sign", options=["-", "+"])
    a_0 = float(f"{a_0_sign}{a_0_mantissa}e{a_0_exponent}")

    a_1_cols = st.columns(3)
    with a_1_cols[0]:
        a_1_mantissa = st.slider("a_1 mantissa", 0.0, 1.0, a_1_default_mantisssa)
    with a_1_cols[1]:
        a_1_exponent = st.slider("a_1 exponent", -5, 5, a_0_default_exponent)
    with a_1_cols[2]:
        a_1_sign = st.select_slider(
            "a_1 sign", options=["-", "+"], value=a_1_default_sign
        )
    a_1 = float(f"{a_1_sign}{a_1_mantissa}e{a_1_exponent}")

    a_2_cols = st.columns(3)
    with a_2_cols[0]:
        a_2_mantissa = st.slider("a_2 mantissa", 0.0, 1.0, a_2_default_mantisssa)
    with a_2_cols[1]:
        a_2_exponent = st.slider("a_2 exponent", -5, 5, a_2_default_exponent)
    with a_2_cols[2]:
        a_2_sign = st.select_slider(
            "a_2 sign", options=["-", "+"], value=a_2_default_sign
        )
    a_2 = float(f"{a_2_sign}{a_2_mantissa}e{a_2_exponent}")

    a_3_cols = st.columns(3)
    with a_3_cols[0]:
        a_3_mantissa = st.slider("a_3 mantissa", 0.0, 1.0, a_3_default_mantisssa)
    with a_3_cols[1]:
        a_3_exponent = st.slider("a_3 exponent", -5, 5, a_3_default_exponent)
    with a_3_cols[2]:
        a_3_sign = st.select_slider(
            "a_3 sign", options=["-", "+"], value=a_3_default_sign
        )
    a_3 = float(f"{a_3_sign}{a_3_mantissa}e{a_3_exponent}")

    a_4_cols = st.columns(3)
    with a_4_cols[0]:
        a_4_mantissa = st.slider("a_4 mantissa", 0.0, 1.0, a_4_default_mantisssa)
    with a_4_cols[1]:
        a_4_exponent = st.slider("a_4 exponent", -5, 5, a_4_default_exponent)
    with a_4_cols[2]:
        a_4_sign = st.select_slider(
            "a_4 sign", options=["-", "+"], value=a_4_default_sign
        )
    a_4 = float(f"{a_4_sign}{a_4_mantissa}e{a_4_exponent}")

    b_0_cols = st.columns(3)
    with b_0_cols[0]:
        b_0_mantissa = st.slider("b_0 mantissa", 0.0, 1.0, b_0_default_mantisssa)
    with b_0_cols[1]:
        b_0_exponent = st.slider("b_0 exponent", -5, 5, b_0_default_exponent)
    with b_0_cols[2]:
        b_0_sign = st.select_slider(
            "b_0 sign", options=["-", "+"], value=b_0_default_sign
        )
    b_0 = float(f"{b_0_sign}{b_0_mantissa}e{b_0_exponent}")

    b_1_cols = st.columns(3)
    with b_1_cols[0]:
        b_1_mantissa = st.slider("b_1 mantissa", 0.0, 1.0, b_1_default_mantisssa)
    with b_1_cols[1]:
        b_1_exponent = st.slider("b_1 exponent", -5, 5, b_1_default_exponent)
    with b_1_cols[2]:
        b_1_sign = st.select_slider(
            "b_1 sign", options=["-", "+"], value=b_1_default_sign
        )
    b_1 = float(f"{b_1_sign}{b_1_mantissa}e{b_1_exponent}")

    b_2_cols = st.columns(3)
    with b_2_cols[0]:
        b_2_mantissa = st.slider("b_2 mantissa", 0.0, 1.0, b_2_default_mantisssa)
    with b_2_cols[1]:
        b_2_exponent = st.slider("b_2 exponent", -5, 5, b_2_default_exponent)
    with b_2_cols[2]:
        b_2_sign = st.select_slider(
            "b_2 sign", options=["-", "+"], value=b_2_default_sign
        )
    b_2 = float(f"{b_2_sign}{b_2_mantissa}e{b_2_exponent}")

    # elif preset_mode == "Burgers (single-channel hack)":
    #     use_difficulty = True

    #     a_0 = 0.0
    #     a_1 = 0.0
    #     a_2 = 1.5
    #     a_3 = 0.0
    #     a_4 = 0.0
    #     b_0 = 0.0
    #     b_1 = -2.0
    #     b_2 = 0.0


linear_tuple = (a_0, a_1, a_2, a_3, a_4)
nonlinear_tuple = (b_0, b_1, b_2)

linear_tuple = tuple([overall_scale * x for x in linear_tuple])
nonlinear_tuple = tuple([overall_scale * x for x in nonlinear_tuple])

if dimension_type in ["1d ST", "1d"]:
    num_spatial_dims = 1
elif dimension_type in ["2d ST", "2d"]:
    num_spatial_dims = 2
elif dimension_type == "3d":
    num_spatial_dims = 3

if use_difficulty:
    stepper = ex.RepeatedStepper(
        ex.normalized.DifficultyGeneralNonlinearStepper(
            num_spatial_dims,
            num_points,
            linear_difficulties=tuple(x / num_substeps for x in linear_tuple),
            nonlinear_difficulties=tuple(x / num_substeps for x in nonlinear_tuple),
        ),
        num_substeps,
    )
else:
    stepper = ex.RepeatedStepper(
        ex.normalized.NormlizedGeneralNonlinearStepper(
            num_spatial_dims,
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

if num_spatial_dims == 1:
    ic_gen = ex.ic.RandomSineWaves1d(
        num_spatial_dims, cutoff=num_modes_init, max_one=True
    )
else:
    ic_gen = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims, cutoff=num_modes_init, max_one=True
    )
u_0 = ic_gen(num_points, key=jax.random.PRNGKey(0))

trj = ex.rollout(stepper, num_steps, include_init=True)(u_0)


v_range = st.slider("Value range", 0.1, 10.0, 1.0)


st.write(f"Linear: {linear_tuple}   Nonlinear: {nonlinear_tuple}")


TEMPLATE_IFRAME = """
    <div>
        <iframe id="{canvas_id}" src="https://keksboter.github.io/v4dv/?inline" width="{canvas_width}" height="{canvas_height}" frameBorder="0" sandbox="allow-same-origin allow-scripts"></iframe>
    </div>
    <script>

        window.addEventListener(
            "message",
            (event) => {{
                if (event.data !== "ready") {{
                    return;
                }}
                let data_decoded = Uint8Array.from(atob("{data_code}"), c => c.charCodeAt(0));
                let cmap_decoded = Uint8Array.from(atob("{cmap_code}"), c => c.charCodeAt(0));
                const iframe = document.getElementById("{canvas_id}");
                if (iframe === null) return;
                iframe.contentWindow.postMessage({{
                    volume: data_decoded,
                    cmap: cmap_decoded,
                    settings: {settings_json}
                }},
                "*");
            }},
            false,
        );
    </script>
"""


@dataclass(unsafe_hash=True)
class ViewerSettings:
    width: int
    height: int
    background_color: tuple
    show_colormap_editor: bool
    show_volume_info: bool
    vmin: Optional[float]
    vmax: Optional[float]


def show(
    data: np.ndarray,
    colormap,
    width: int = 800,
    height: int = 600,
    background_color=(0.0, 0.0, 0.0, 1.0),
    show_colormap_editor=False,
    show_volume_info=False,
    vmin=None,
    vmax=None,
):
    return VolumeRenderer(
        data,
        colormap,
        ViewerSettings(
            width,
            height,
            background_color,
            show_colormap_editor,
            show_volume_info,
            vmin,
            vmax,
        ),
    )


class VolumeRenderer(DisplayObject):
    def __init__(self, data: np.ndarray, colormap, settings: ViewerSettings):
        super(VolumeRenderer, self).__init__(
            data={"volume": data, "cmap": colormap, "settings": settings}
        )

    def _repr_html_(self):
        data = self.data["volume"]
        colormap = self.data["cmap"]
        settings = self.data["settings"]
        buffer = io.BytesIO()
        np.save(buffer, data.astype(np.float32))
        data_code = base64.b64encode(buffer.getvalue())

        buffer2 = io.BytesIO()
        colormap_data = colormap(np.linspace(0, 1, 256)).astype(np.float32)
        np.save(buffer2, colormap_data)
        cmap_code = base64.b64encode(buffer2.getvalue())

        canvas_id = f"v4dv_canvas_{str(random.randint(0,2**32))}"
        html_code = TEMPLATE_IFRAME.format(
            canvas_id=canvas_id,
            data_code=data_code.decode("utf-8"),
            cmap_code=cmap_code.decode("utf-8"),
            canvas_width=settings.width,
            canvas_height=settings.height,
            settings_json=json.dumps(dataclasses.asdict(settings)),
        )
        return html_code

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()


def felix_cmap_hack(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        for i, a in enumerate(cmap.colors):
            a.append(2 * abs(i / cmap.N - 0.5))
    elif isinstance(cmap, LinearSegmentedColormap):
        cmap._segmentdata["alpha"] = np.array(
            [[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )
    return cmap


if dimension_type == "1d ST":
    ex.viz.plot_spatio_temporal(trj, vlim=(-v_range, v_range))
    fig = plt.gcf()
    st.pyplot(fig)
elif dimension_type == "1d":
    ani = ex.viz.animate_state_1d(trj, vlim=(-v_range, v_range))
    components.html(ani.to_jshtml(), height=800)
elif dimension_type == "2d":
    ani = ex.viz.animate_state_2d(trj, vlim=(-v_range, v_range))
    components.html(ani.to_jshtml(), height=800)
elif dimension_type == "2d ST":
    trj_rearranged = trj.transpose(1, 0, 2, 3)[None]
    components.html(
        show(
            trj_rearranged,
            plt.get_cmap("RdBu_r"),
            width=1500,
            height=800,
            show_colormap_editor=True,
            show_volume_info=True,
        ).__html__(),
        height=800,
    )
elif dimension_type == "3d":
    components.html(
        show(
            trj,
            plt.get_cmap("RdBu_r"),
            width=1500,
            height=800,
            show_colormap_editor=True,
            show_volume_info=True,
        ).__html__(),
        height=800,
    )
