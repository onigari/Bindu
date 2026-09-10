"""
Run with:
    streamlit run app.py
"""
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from PIL import Image

from imgutil.core import *
from imgutil.histograms import *
from imgutil.frequency import *


# ---------------------------------------------------------------------------
# Synthetic sample images (self-contained -- no bundled asset files needed)
# ---------------------------------------------------------------------------

def make_gradient_circle_image(size: int = 256) -> np.ndarray:
    """Same generator used in the demo scripts: RGB gradient plus a white circle."""
    x = np.linspace(0, 255, size)
    y = np.linspace(0, 255, size)
    xx, yy = np.meshgrid(x, y)
    r = xx
    g = yy
    b = 255 - (xx + yy) / 2
    img = np.stack([r, g, b], axis=-1)

    yy_idx, xx_idx = np.ogrid[:size, :size]
    circle_mask = (xx_idx - size * 0.7) ** 2 + (yy_idx - size * 0.3) ** 2 < (size * 0.15) ** 2
    img[circle_mask] = [255, 255, 255]
    return img.astype(np.float64)


def make_stripes_image(size: int = 256) -> np.ndarray:
    """High-frequency vertical stripes, good for later frequency-domain demos too."""
    x = np.arange(size)
    stripe = (np.sin(x * np.pi / 4) > 0).astype(np.float64) * 255
    stripe_2d = np.tile(stripe, (size, 1))
    img = np.stack([stripe_2d, np.roll(stripe_2d, 8, axis=1), np.full((size, size), 128.0)], axis=-1)
    return img.astype(np.float64)


def make_radial_image(size: int = 256) -> np.ndarray:
    """Smooth radial gradient -- almost all low-frequency content, good filtering contrast."""
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size / 2, size / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist_norm = np.clip(dist / (size / 2) * 255, 0, 255)
    img = np.stack([255 - dist_norm, dist_norm, np.full((size, size), 180.0)], axis=-1)
    return img.astype(np.float64)


SAMPLE_IMAGES = {
    "Gradient + circle": make_gradient_circle_image,
    "Stripes": make_stripes_image,
    "Radial gradient": make_radial_image,
}


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_fft(channel: np.ndarray) -> np.ndarray:
    """Raw FFT for one channel, memoized on the channel's bytes only.

    Kept separate from log-scale display formatting: log_scale only
    changes how the magnitude is *displayed*, not the FFT itself, so it
    must not be part of this cache key or every checkbox toggle forces a
    full fft2 recompute for nothing.
    """
    return compute_fft(channel)


@st.cache_resource(show_spinner=False)
def _shared_colorbar_figure(vmin: float, vmax: float, cmap_name: str = "viridis"):
    """One small shared matplotlib colorbar legend, cached.

    Built once per (vmin, vmax, cmap) combination via st.cache_resource
    (matplotlib Figures aren't picklable/hashable the way st.cache_data
    wants, so cache_resource is the correct cache for this), then reused
    across reruns instead of rebuilt on every checkbox toggle. This is
    the only piece of real matplotlib in this rendering path -- the
    spectrum panels themselves stay on the fast numpy path.
    """
    fig, ax = plt.subplots(figsize=(0.9, 3.6))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_name),
        cax=ax,
    )
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def _shared_normalized_spectrum_to_rgb(spectrum: np.ndarray, vmin: float, vmax: float,
                                        cmap_name: str = "viridis") -> np.ndarray:
    """Map a spectrum to RGB, normalized against a shared (vmin, vmax) range
    instead of each panel's own min/max.

    A single shared colorbar is only a truthful legend if every panel was
    colored against the same range -- per-panel auto-scaling would make
    the shared colorbar lie for panels other than the one that set it.
    """
    norm = (spectrum - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(spectrum)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Read an uploaded file (via Streamlit's uploader) into a float64 RGB array.

    Uses PIL instead of cv2.imread since the uploaded file is in-memory
    bytes, not a filesystem path -- cv2.imread only reads from disk.
    """
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    return np.array(image).astype(np.float64)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

TOOLS = [
    "Color channel analyzer and histogram",
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def run_color_channel_analyzer(img: np.ndarray, channels: dict) -> None:
    """Stage A + B: channel separation and histograms."""
    st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width=400)

    st.divider()

    # --- Stage A: channel separation ---
    st.subheader("Color Channels")
    channel_view_mode = st.radio("Channel display style", ["Grayscale", "Tinted color"], horizontal=True)

    col_r, col_g, col_b = st.columns(3)
    cols = [col_r, col_g, col_b]
    names = ["R", "G", "B"]

    # 1. Render all three images instantly
    for col, name in zip(cols, names):
        with col:
            if channel_view_mode == "Grayscale":
                display_img = channel_as_grayscale_image(channels[name])
            else:
                display_img = channel_as_color_image(channels[name], name)
            
            st.image(display_img, caption=f"{name} channel", width="stretch")

    # 2. Generate and render the matplotlib histograms afterwards
    for col, name in zip(cols, names):
        with col:
            hist_fig = plot_single_channel_histogram(channels[name], name)
            st.pyplot(hist_fig)
            plt.close(hist_fig)

    st.divider()

    # --- Stage B: histograms ---
    st.subheader("Combined Histogram")

    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        show_combined = st.checkbox("Show combined histogram", value=True)
    with hist_col2:
        show_luminance = st.checkbox("Show luminance histogram", value=False)

    fig = plot_channel_histograms(
        channels,
        title="Combined Histogram",
        show_combined=show_combined,
        show_luminance=show_luminance,
    )
    st.pyplot(fig, width="content")
    plt.close(fig)

    st.divider()

    # --- Stage C: frequency content ---
    # This whole block is its own fragment: toggling the checkboxes below
    # reruns only this function, not the channel images/histograms above
    # (which is what caused the visible lag before).
    render_frequency_content(channels, names)


@st.fragment
def render_frequency_content(channels: dict, names: list) -> None:
    """Stage C: frequency content -- isolated as a fragment.

    Without @st.fragment, Streamlit reruns the *entire* script on every
    checkbox click, so toggling "Log-scale magnitude" was also
    re-rendering the channel images and every histogram above it before
    it ever got to redrawing the spectra -- that's the lag. Scoping this
    block to a fragment means only this function reruns when its own
    widgets change.

    Spectrum panels are plain numpy->RGB images (fast, no per-panel
    Figure), paired with a single shared matplotlib colorbar legend that's
    cached via st.cache_resource so it isn't rebuilt on every toggle.
    Every panel is normalized to the same (vmin, vmax) range so that one
    shared legend is an accurate read for all of them.
    """
    st.subheader("Frequency Content")

    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        log_scale = st.checkbox("Log-scale magnitude", value=True)
    with freq_col2:
        show_luminance_spectrum = st.checkbox("Show combined luminance spectrum", value=True)

    # Compute all spectra up front (cheap; the real cost was always the
    # rendering step, not the FFT itself thanks to _cached_fft).
    spectra = {}
    for name in names:
        fft_result = _cached_fft(channels[name])
        spectra[name] = compute_magnitude_spectrum(fft_result, log_scale=log_scale)

    if show_luminance_spectrum:
        luminance = 0.299 * channels["R"] + 0.587 * channels["G"] + 0.114 * channels["B"]
        lum_fft = _cached_fft(luminance)
        spectra["Luminance"] = compute_magnitude_spectrum(lum_fft, log_scale=log_scale)

    panel_names = list(spectra.keys())
    n_panels = len(panel_names)

    # Shared (vmin, vmax) across all panels so the one colorbar is accurate
    # for every panel, not just whichever one happened to set the range.
    global_vmin = min(float(s.min()) for s in spectra.values())
    global_vmax = max(float(s.max()) for s in spectra.values())

    legend_col, *panel_cols = st.columns([1] + [4] * n_panels)
    with legend_col:
        st.caption("Magnitude")
        cb_fig = _shared_colorbar_figure(round(global_vmin, 4), round(global_vmax, 4))
        st.pyplot(cb_fig, width="stretch")

    for col, name in zip(panel_cols, panel_names):
        with col:
            rgb = _shared_normalized_spectrum_to_rgb(spectra[name], global_vmin, global_vmax)
            st.image(rgb, caption=f"{name} spectrum", width="stretch")

    # Frequency-domain energy per channel (how much "information" each channel carries)
    energy_report = compare_channel_energy(channels)
    energy_cols = st.columns(3)
    for col, name in zip(energy_cols, names):
        with col:
            st.metric(
                label=f"{name} energy share",
                value=f"{energy_report[name]['fraction']:.1%}",
            )


TOOL_RUNNERS = {
    "Color channel analyzer and histogram": run_color_channel_analyzer,
}


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bindu - Image Tools", layout="wide")
# st.title("imgutil")
# st.caption("Interactive companion to the imgutil image-processing package")

# --- Image source selection ---
st.sidebar.header("Image source")
source_mode = st.sidebar.radio("Load from", ["Upload", "Sample gallery"])

img = None

if source_mode == "Upload":
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        img = load_uploaded_image(uploaded_file)
    else:
        st.info("Upload an image, or switch to the sample gallery in the sidebar.")

else:
    sample_name = st.sidebar.selectbox("Sample image", list(SAMPLE_IMAGES.keys()))
    img = SAMPLE_IMAGES[sample_name]()

# st.sidebar.caption(f"Image shape: {img.shape[0]} x {img.shape[1]}")

# --- Tool selection ---
st.sidebar.divider()
st.sidebar.header("Tools")
selected_tool = st.sidebar.radio("Choose a tool", TOOLS, label_visibility="collapsed")


if img is None:
    st.stop()


channels = split_channels(img)

# --- Main content ---
TOOL_RUNNERS[selected_tool](img, channels)