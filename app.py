"""
Run with:
    streamlit run app.py
"""
import io

import numpy as np
import streamlit as st
from PIL import Image

from imgutil.core import *
from imgutil.histograms import *


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
    for col, name in zip([col_r, col_g, col_b], ["R", "G", "B"]):
        with col:
            if channel_view_mode == "Grayscale":
                display_img = channel_as_grayscale_image(channels[name])
            else:
                display_img = channel_as_color_image(channels[name], name)
            st.image(display_img, caption=f"{name} channel", width="stretch")

            hist_fig = plot_single_channel_histogram(channels[name], name)
            st.pyplot(hist_fig)

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

st.sidebar.caption(f"Image shape: {img.shape[0]} x {img.shape[1]}")

# --- Tool selection ---
st.sidebar.divider()
st.sidebar.header("Tools")
selected_tool = st.sidebar.radio("Choose a tool", TOOLS, label_visibility="collapsed")


if img is None:
    st.stop()


channels = split_channels(img)

# --- Main content ---
TOOL_RUNNERS[selected_tool](img, channels)
