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
from imgutil.filtering import (
    gaussian_blur_channel, gaussian_blur,
    box_blur_channel, box_blur,
    laplacian_sharpen_channel, laplacian_sharpen,
    unsharp_mask_channel, unsharp_mask,
    filter_channels, compare_perchannel_vs_merged,
)


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
    return compute_fft(channel)


@st.cache_resource(show_spinner=False)
def _shared_colorbar_figure(vmin: float, vmax: float, cmap_name: str = "viridis"):
    fig, ax = plt.subplots(figsize=(1.1, 4.4))
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


def make_single_channel_filtered_image(channels: dict, chan_fn, channel_name: str, **kwargs) -> np.ndarray:
    """Apply chan_fn to just one channel, leave the other two untouched, and
    recombine into a full RGB image.

    Useful for visualizing *which* channel a filter's effect actually shows
    up in -- e.g. filtering only R will only change the red content of the
    merged image, so any blur/sharpen artifacts you see are entirely
    attributable to that channel.
    """
    modified = dict(channels)  # shallow copy: unfiltered channels stay shared
    modified[channel_name] = chan_fn(channels[channel_name], **kwargs)
    return merge_channels(modified)


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------
FILTER_CONFIGS = {
    "Gaussian blur": {
        "channel_fn": gaussian_blur_channel,
        "merged_fn": gaussian_blur,
        "params": [
            {"name": "sigma", "label": "Sigma", "min": 0.1, "max": 15.0, "default": 3.0, "step": 0.1},
        ],
    },
    "Box blur": {
        "channel_fn": box_blur_channel,
        "merged_fn": box_blur,
        "params": [
            {"name": "ksize", "label": "Kernel size", "min": 1, "max": 31, "default": 7, "step": 2, "int": True},
        ],
    },
    "Unsharp mask": {
        "channel_fn": unsharp_mask_channel,
        "merged_fn": unsharp_mask,
        "params": [
            {"name": "sigma", "label": "Blur sigma", "min": 0.1, "max": 15.0, "default": 2.0, "step": 0.1},
            {"name": "amount", "label": "Amount", "min": 0.0, "max": 5.0, "default": 1.5, "step": 0.1},
        ],
    },
    "Laplacian sharpen": {
        "channel_fn": laplacian_sharpen_channel,
        "merged_fn": laplacian_sharpen,
        "params": [
            {"name": "ksize", "label": "Kernel size", "min": 1, "max": 15, "default": 3, "step": 2, "int": True},
            {"name": "scale", "label": "Scale", "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1},
        ],
    },
}


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

TOOLS = [
    "Color channel analyzer and histogram",
    "Partial image reconstruction",
    "Filtering",
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

    render_frequency_content(channels, names)


@st.fragment
def render_frequency_content(channels: dict, names: list) -> None:
    st.subheader("Frequency Content")

    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        log_scale = st.checkbox("Log-scale magnitude", value=True)
    with freq_col2:
        show_luminance_spectrum = st.checkbox("Show combined luminance spectrum", value=True)

    channel_spectra = {}
    for name in names:
        fft_result = _cached_fft(channels[name])
        channel_spectra[name] = compute_magnitude_spectrum(fft_result, log_scale=log_scale)

    luminance_spectrum = None
    if show_luminance_spectrum:
        luminance = 0.299 * channels["R"] + 0.587 * channels["G"] + 0.114 * channels["B"]
        lum_fft = _cached_fft(luminance)
        luminance_spectrum = compute_magnitude_spectrum(lum_fft, log_scale=log_scale)

    all_spectra = list(channel_spectra.values()) + (
        [luminance_spectrum] if luminance_spectrum is not None else []
    )
    global_vmin = min(float(s.min()) for s in all_spectra)
    global_vmax = max(float(s.max()) for s in all_spectra)

    # --- Row 1: R/G/B spectra, with the colorbar legend on the right ---
    n_channels = len(names)
    *panel_cols, legend_col = st.columns([4] * n_channels + [1])

    for col, name in zip(panel_cols, names):
        with col:
            rgb = _shared_normalized_spectrum_to_rgb(channel_spectra[name], global_vmin, global_vmax)
            st.image(rgb, caption=f"{name} spectrum", width="stretch")

    with legend_col:
        # Blank caption-height placeholder so the colorbar image itself
        # starts at the same vertical offset as the spectrum images
        # (which each have a real caption above their image).
        st.caption("\u200b")
        cb_fig = _shared_colorbar_figure(round(global_vmin, 4), round(global_vmax, 4))
        st.pyplot(cb_fig, width="stretch")

    # --- Row 2: energy share per channel, directly under the spectra ---
    st.markdown(
        """
        <style>
        .energy-share-label {
            font-size: 0.75rem;
            color: rgba(120, 120, 120, 0.9);
            margin-bottom: 0.1rem;
        }
        .energy-share-value {
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.1;
            margin-bottom: 1.0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    energy_report = compare_channel_energy(channels)
    *energy_cols, _energy_legend_spacer = st.columns([4] * n_channels + [1])
    for col, name in zip(energy_cols, names):
        with col:
            st.markdown(
                f"""
                <div class="energy-share-label">{name} energy share</div>
                <div class="energy-share-value">{energy_report[name]['fraction']:.1%}</div>
                """,
                unsafe_allow_html=True,
            )

    # --- Row 3: combined luminance spectrum, below everything else ---
    if luminance_spectrum is not None:
        lum_width, rest_width = 4, 4 * (n_channels - 1) + 1  # one panel-width + leftover blank
        lum_col, _ = st.columns([lum_width, rest_width])
        with lum_col:
            rgb = _shared_normalized_spectrum_to_rgb(luminance_spectrum, global_vmin, global_vmax)
            st.image(rgb, caption="Luminance spectrum", width="stretch")


def run_partial_reconstruction(img: np.ndarray, channels: dict) -> None:
    st.subheader("Partial Image Reconstruction")
    st.caption("Combine one or two RGB channels using their frequency content. Omitted channels are set to zero.")
    selected = st.multiselect(
        "Channels to reconstruct", ["R", "G", "B"], default=["R", "G"],
        max_selections=2,
    )
    if not selected:
        st.info("Select one or two channels to reconstruct an image.")
        return

    selected = [name for name in "RGB" if name in selected]
    spectra = {name: _cached_fft(channels[name]) for name in selected}
    reconstructed = reconstruct_partial_image(spectra)
    # Round inverse-FFT residue before conversion to avoid losing one intensity level.
    preview = np.rint(np.clip(reconstructed, 0, 255)).astype(np.uint8)
    original_col, result_col = st.columns(2)
    with original_col:
        st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width="stretch")
    with result_col:
        st.image(preview, caption=f"Reconstructed: {' + '.join(selected)}", width="stretch")

    png = io.BytesIO()
    Image.fromarray(preview).save(png, format="PNG")
    st.download_button(
        "Download reconstructed image", data=png.getvalue(),
        file_name=f"reconstructed_{''.join(selected)}.png", mime="image/png",
    )


def run_spatial_filtering(img: np.ndarray, channels: dict) -> None:
    st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width=400)

    st.divider()

    st.subheader("Filter Settings")
    filter_name = st.selectbox("Filter", list(FILTER_CONFIGS.keys()))
    config = FILTER_CONFIGS[filter_name]

    kwargs = {}
    param_cols = st.columns(len(config["params"]))
    for col, param in zip(param_cols, config["params"]):
        with col:
            if param.get("int"):
                value = st.slider(param["label"], param["min"], param["max"], param["default"], step=param["step"],)
                kwargs[param["name"]] = int(value)
            else:
                kwargs[param["name"]] = st.slider(param["label"], param["min"], param["max"], param["default"], step=param["step"],)

    chan_fn = config["channel_fn"]
    merged_fn = config["merged_fn"]

    r_only_result = make_single_channel_filtered_image(channels, chan_fn, "R", **kwargs)
    g_only_result = make_single_channel_filtered_image(channels, chan_fn, "G", **kwargs)
    b_only_result = make_single_channel_filtered_image(channels, chan_fn, "B", **kwargs)
    merged_result = merged_fn(img, **kwargs)

    stats = compare_perchannel_vs_merged(img, channels, chan_fn, **kwargs)

    st.divider()

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            np.clip(r_only_result, 0, 255).astype(np.uint8),
            caption=f"{filter_name} (R only)",
            width="stretch",
        )
    with col2:
        st.image(
            np.clip(g_only_result, 0, 255).astype(np.uint8),
            caption=f"{filter_name} (G only)",
            width="stretch",
        )
    with col3:
        st.image(
            np.clip(b_only_result, 0, 255).astype(np.uint8),
            caption=f"{filter_name} (B only)",
            width="stretch",
        )
    st.image(
            np.clip(merged_result, 0, 255).astype(np.uint8),
            caption=f"{filter_name} (all channels)",
            width=400,
        )

    st.divider()


TOOL_RUNNERS = {
    "Color channel analyzer and histogram": run_color_channel_analyzer,
    "Partial image reconstruction": run_partial_reconstruction,
    "Filtering": run_spatial_filtering,
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
