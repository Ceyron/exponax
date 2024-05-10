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
    st.title("Exponax Dynamics Brewer")
    dimension_type = st.select_slider(
        "Number of Spatial Dimensions (ST=Spatio-Temporal plot)",
        options=["1d ST", "1d", "2d", "2d ST", "3d"],
    )
    num_points = st.slider("Number of points", 16, 256, 48)
    num_steps = st.slider("Number of steps", 1, 300, 50)
    num_modes_init = st.slider(
        "Number of modes in the initial condition (not GS)", 1, 40, 5
    )
    num_substeps = st.slider("Number of substeps", 1, 100, 1)

    v_range = st.slider("Value range", 0.1, 10.0, 1.0)

    st.divider()

    rd_type = st.select_slider(
        "Type of Equation", ["FisherKPP", "GrayScott", "SwiftHohenberg"]
    )

    domain_extent_cols = st.columns(3)
    with domain_extent_cols[0]:
        domain_extent_mantissa = st.slider("domain_extent mantissa", 0.0, 1.0, 0.1)
    with domain_extent_cols[1]:
        domain_extent_exponent = st.slider("domain_extent exponent", -5, 5, 1)
    domain_extent_sign = "+"
    domain_extent = float(
        f"{domain_extent_sign}{domain_extent_mantissa}e{domain_extent_exponent}"
    )

    dt_cols = st.columns(3)
    with dt_cols[0]:
        dt_mantissa = st.slider("dt mantissa", 0.0, 1.0, 0.5)
    with dt_cols[1]:
        dt_exponent = st.slider("dt exponent", -5, 5, 0)
    dt_sign = "+"
    dt = float(f"{dt_sign}{dt_mantissa}e{dt_exponent}")

    if rd_type == "FisherKPP":
        diffusivity_cols = st.columns(3)
        with diffusivity_cols[0]:
            diffusivity_mantissa = st.slider("diffusivity mantissa", 0.0, 1.0, 0.1)
        with diffusivity_cols[1]:
            diffusivity_exponent = st.slider("diffusivity exponent", -5, 5, -2)
        diffusivity_sign = "+"
        diffusivity = float(
            f"{diffusivity_sign}{diffusivity_mantissa}e{diffusivity_exponent}"
        )

        reactivity_cols = st.columns(3)
        with reactivity_cols[0]:
            reactivity_mantissa = st.slider("reactivity mantissa", 0.0, 1.0, 0.1)
        with reactivity_cols[1]:
            reactivity_exponent = st.slider("reactivity exponent", -5, 5, 0)
        reactivity_sign = "+"
        reactivity = float(
            f"{reactivity_sign}{reactivity_mantissa}e{reactivity_exponent}"
        )
    elif rd_type == "GrayScott":
        feed_rate = st.slider("feed_rate", 0.00, 0.08, 0.008, 0.004, format="%.3f")
        kill_rate = st.slider("kill_rate", 0.03, 0.07, 0.046, 0.002, format="%.3f")

        st.write("Only displays the second channel")
    elif rd_type == "SwiftHohenberg":
        reactivity = st.slider("reactivity", 0.0, 1.0, 0.7)
        critical_number = st.slider("critical_number", 0.0, 1.0, 1.0)

    # st.write(f"dt: {dt}")
    # st.write(f"diffusivity: {diffusivity}")
    # st.write(f"domain_extent: {domain_extent}")

    # st.write(f"Linear: {linear_tuple}")
    # st.write(f"Nonlinear: {nonlinear_tuple}")

if dimension_type in ["1d ST", "1d"]:
    num_spatial_dims = 1
elif dimension_type in ["2d ST", "2d"]:
    num_spatial_dims = 2
elif dimension_type == "3d":
    num_spatial_dims = 3

if rd_type == "FisherKPP":
    stepper = ex.RepeatedStepper(
        ex.reaction.FisherKPP(
            num_spatial_dims,
            domain_extent,
            num_points,
            dt / num_substeps,
            diffusivity=diffusivity,
            reactivity=reactivity,
        ),
        num_substeps,
    )
elif rd_type == "GrayScott":
    stepper = ex.RepeatedStepper(
        ex.reaction.GrayScott(
            num_spatial_dims,
            domain_extent,
            num_points,
            dt / num_substeps,
            feed_rate=feed_rate,
            kill_rate=kill_rate,
        ),
        num_substeps,
    )
elif rd_type == "SwiftHohenberg":
    stepper = ex.RepeatedStepper(
        ex.reaction.SwiftHohenberg(
            num_spatial_dims,
            domain_extent,
            num_points,
            dt / num_substeps,
            reactivity=reactivity,
            critical_number=critical_number,
        ),
        num_substeps,
    )

if rd_type == "GrayScott":
    ic_gen = ex.ic.RandomMultiChannelICGenerator(
        [
            ex.ic.RandomGaussianBlobs(num_spatial_dims, one_complement=True),
            ex.ic.RandomGaussianBlobs(num_spatial_dims),
        ]
    )
else:
    if num_spatial_dims == 1:
        ic_gen = ex.ic.RandomSineWaves1d(
            num_spatial_dims, cutoff=num_modes_init, max_one=True
        )
    else:
        ic_gen = ex.ic.RandomTruncatedFourierSeries(
            num_spatial_dims, cutoff=num_modes_init, max_one=True
        )

    if rd_type == "FisherKPP":
        ic_gen = ex.ic.ClampingICGenerator(ic_gen)

u_0 = ic_gen(num_points, key=jax.random.PRNGKey(0))

trj = ex.rollout(stepper, num_steps, include_init=True)(u_0)

if rd_type == "GrayScott":
    trj = trj[:, 1:]


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
