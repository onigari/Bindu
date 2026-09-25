"""
Run with:
    streamlit run app.py
"""
import cv2
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from animated_histograms import animated_histogram, animated_spectrum, prepare_spectrum_channels
from ui import apply_style, render_brand, render_header, TOOL_DETAILS

from imgutil.core import *
from imgutil.color_spaces import rgb_to_ycbcr
from imgutil.compression import compress_image, decompress_image
from imgutil.lossless import compress_png, decompress_png, encode_bmp
from imgutil.restoration import restore_photo
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
    "Old photo (AI-generated example)": lambda: load_image_rgb(
        str(Path(__file__).parent / "inputs" / "old_photo_example.png")),
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
    """Use the existing OpenCV dependency for general image-file import."""
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read this image file.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


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
    "Color space comparison",
    "Compression and decompression",
    "Lossless PNG compression",
    "Filtering",
    "Old Photo Restoration",
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_ready(label, source, settings=None, key="run"):
    """Require explicit approval again whenever the input or settings change."""
    data = source.tobytes() if isinstance(source, np.ndarray) else source
    signature = hashlib.sha256(data + repr(settings).encode()).hexdigest()
    state_key = f"approved_{selected_tool}_{key}"
    if st.session_state.get(state_key) != signature:
        st.session_state.pop(state_key, None)
    if st.button(label, key=f"{selected_tool}_{key}", type="primary"):
        st.session_state[state_key] = signature
    return st.session_state.get(state_key) == signature


def image_picker(tool):
    st.subheader("Your image")
    source_mode = st.radio("Load from", ["Upload", "Sample gallery"],
                           key=f"{tool}_source", horizontal=True)
    img = None
    if source_mode == "Upload":
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"],
                                    key=f"{tool}_upload")
        if uploaded is not None:
            try:
                img = load_uploaded_image(uploaded)
            except ValueError as exc:
                st.error(str(exc))
    else:
        sample = st.selectbox("Sample image", list(SAMPLE_IMAGES), key=f"{tool}_sample")
        try:
            img = SAMPLE_IMAGES[sample]()
        except (ValueError, OSError) as exc:
            st.error(f"Could not load the sample: {exc}")
    if img is None:
        # Clearing the source must also clear permission to process it.
        for key in list(st.session_state):
            if key.startswith(f"approved_{tool}_"):
                del st.session_state[key]
        st.info("Upload an image or choose a sample to get started.")
    return img


def run_color_channel_analyzer(img: np.ndarray, channels: dict) -> None:
    """Stage A + B: channel separation and histograms."""
    channel_view_mode = st.radio("Channel display style", ["Grayscale", "Tinted color"], horizontal=True)
    show_combined = st.checkbox("Show combined histogram", value=True)
    show_luminance = st.checkbox("Show luminance histogram", value=False)
    log_scale = st.checkbox("Log-scale magnitude", value=True)
    show_luminance_spectrum = st.checkbox("Show combined luminance spectrum", value=True)
    if not tool_ready("Analyze image", img, (channel_view_mode, show_combined, show_luminance, log_scale, show_luminance_spectrum)):
        return
    with st.expander("Source image", expanded=False):
        st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width=400)

    st.divider()

    # --- Stage A: channel separation ---
    st.subheader("Color Channels")
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

    # Draw the histogram curves in the browser without blocking processing.
    for col, name in zip(cols, names):
        with col:
            animated_histogram({name: channels[name]}, f"{name} histogram")

    st.divider()

    # --- Stage B: histograms ---
    st.subheader("Combined Histogram")

    animated_histogram(channels, "Combined Histogram", combined=show_combined,
                       luminance=show_luminance)

    st.divider()

    render_frequency_content(channels, names, log_scale, show_luminance_spectrum)


def render_frequency_content(channels: dict, names: list, log_scale: bool, show_luminance_spectrum: bool) -> None:
    st.subheader("Frequency Content")

    sources = dict(channels)
    if show_luminance_spectrum:
        sources["Luminance"] = 0.299 * channels["R"] + 0.587 * channels["G"] + 0.114 * channels["B"]
    prepared, bound = prepare_spectrum_channels(sources)
    n_channels = len(names)
    for col, name in zip(st.columns(n_channels), names):
        with col:
            animated_spectrum(prepared[name], f"{name} spectrum", bound=bound, log_scale=log_scale)
    st.caption("Spectra are calculated live from source rows. Zero frequency is centered; each new row changes the whole spectrum.")
    maximum = float(np.log1p(bound) if log_scale else bound)
    st.caption(f"Shared magnitude scale: 0 to {maximum:.2f}.")

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

    if show_luminance_spectrum:
        with st.columns(n_channels)[0]:
            animated_spectrum(prepared["Luminance"], "Luminance spectrum", bound=bound, log_scale=log_scale)


def run_partial_reconstruction(img: np.ndarray, channels: dict) -> None:
    st.subheader("Partial Image Reconstruction")
    st.caption("Reconstruct using only a subset of RGB channels.")

    st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width=400)

    st.divider()

    if not tool_ready("Reconstruct image", img):
        return

    # Pre-compute FFTs once
    channel_ffts = {name: _cached_fft(channels[name]) for name in "RGB"}

    def _render_combo(selected: list, col) -> None:
        spectra = {name: channel_ffts[name] for name in selected}
        reconstructed = reconstruct_partial_image(spectra)
        preview = np.rint(np.clip(reconstructed, 0, 255)).astype(np.uint8)
        label = " + ".join(selected)
        with col:
            st.image(preview, caption=label, width="stretch")

    # --- Row 1: all 2-channel combinations ---
    st.markdown("**Two channels**")
    two_channel_combos = [["R", "G"], ["R", "B"], ["G", "B"]]
    cols = st.columns(3)
    for combo, col in zip(two_channel_combos, cols):
        _render_combo(combo, col)

    # --- Row 2: individual channels ---
    st.markdown("**One channel**")
    one_channel_combos = [["R"], ["G"], ["B"]]
    cols = st.columns(3)
    for combo, col in zip(one_channel_combos, cols):
        _render_combo(combo, col)


def run_color_space_comparison(img: np.ndarray, channels: dict) -> None:
    st.subheader("RGB vs YCbCr")
    st.image(np.clip(img, 0, 255).astype(np.uint8), caption="Original", width=400)
    st.caption(
        "RGB stores red, green, and blue intensities. YCbCr separates luma (Y) "
        "from blue-difference (Cb) and red-difference (Cr) color information. "
        "This comparison uses full-range values from 0 to 255, with neutral chroma at 128. "
        "Conversion alone does not compress the image."
    )
    center = st.checkbox("Remove channel mean before FFT", value=True, key="color_space_center")
    if not tool_ready("Compare color spaces", img, center):
        return
    converted = rgb_to_ycbcr(img)
    ycbcr = {name: converted[:, :, i] for i, name in enumerate(("Y", "Cb", "Cr"))}
    spaces = [("RGB", channels), ("YCbCr", ycbcr)]
    st.caption("All channel previews use the same grayscale scale. Mid-gray Cb/Cr indicates neutral color.")
    for label, space in spaces:
        st.markdown(f"**{label} channels**")
        for col, (name, channel) in zip(st.columns(3), space.items()):
            with col:
                st.image(np.rint(channel).astype(np.uint8), caption=name, width="stretch")

    st.subheader("Channel Histograms")
    for label, space in spaces:
        st.markdown(f"**{label} histograms**")
        for col, (name, channel) in zip(st.columns(3), space.items()):
            with col:
                animated_histogram({name: channel}, f"{name} histogram")

    st.subheader("Frequency Content")
    st.caption(
        "Removing the mean highlights spatial detail and removes constant offsets, including neutral chroma. "
        "All six log-magnitude spectra share one color scale."
    )
    sources = {name: channel for _, space in spaces for name, channel in space.items()}
    prepared, bound = prepare_spectrum_channels(sources, center=center)
    for label, space in spaces:
        st.markdown(f"**{label} spectra**")
        for col, name in zip(st.columns(3), space):
            with col:
                animated_spectrum(prepared[name], f"{name} spectrum", bound=bound)
    st.caption(f"Shared magnitude scale: 0 to {np.log1p(bound):.2f} (log scale). Live preview up to 128 pixels per side.")



def _png_bytes(img: np.ndarray) -> bytes:
    return compress_png(img)


def run_compression(img, channels) -> None:
    st.subheader("Compression and Decompression")
    mode = st.radio("Operation", ["Compress", "Decompress"], horizontal=True)
    if mode == "Decompress":
        st.caption("Upload a Fourier .npz file downloaded from this tool. The original image is not required.")
        uploaded = st.file_uploader("Compressed image", type=["npz"], key="compressed_image")
        if uploaded is None:
            st.session_state.pop(f"approved_{selected_tool}_decompress", None)
            return
        if not tool_ready("Decompress image", uploaded.getvalue(), key="decompress"):
            return
        try:
            restored = decompress_image(uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        st.image(restored, caption="Decompressed image", width="stretch")
        st.caption(f"{restored.shape[1]} × {restored.shape[0]} pixels")
        st.download_button("Download decompressed PNG", _png_bytes(restored), "decompressed.png", "image/png", on_click="ignore")
        return

    img = image_picker(selected_tool)
    if img is None:
        return
    st.caption(
        "Keep the strongest Fourier harmonics in each RGB channel until the selected energy target is reached. "
        "Lower targets usually create smaller files with more detail loss. Removed information cannot be recovered."
    )
    with st.form("compression_settings"):
        energy = st.slider("Energy to retain per channel (%)", 1.0, 100.0, 99.0, 0.1)
        submitted = st.form_submit_button("Compress and preview")
    if not submitted:
        return
    try:
        with st.spinner("Compressing and reconstructing image..."):
            archive, stats = compress_image(img, energy)
            restored = decompress_image(archive)
    except ValueError as exc:
        st.error(str(exc))
        return
    original = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
    original_png = _png_bytes(original)
    mse = float(np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original", width="stretch")
    with right:
        st.image(restored, caption="Reconstructed from compressed file", width="stretch")
    size_col, ratio_col, mse_col, psnr_col = st.columns(4)
    size_col.metric("Archive size", f"{len(archive) / 1024:,.1f} KiB")
    ratio_col.metric("Raw RGB / archive", f"{original.nbytes / len(archive):.2f}×")
    mse_col.metric("MSE", f"{mse:.3f}")
    psnr_col.metric("PSNR", "∞ (identical)" if mse == 0 else f"{psnr:.2f} dB")
    st.caption(
        f"Raw RGB: {original.nbytes / 1024:,.1f} KiB · Original encoded as PNG: {len(original_png) / 1024:,.1f} KiB. "
        "A ratio below 1 means the archive is larger than raw RGB. Fourier archives can be larger than PNG/JPEG, "
        "especially at high retention. Energy retention is not a file-size percentage."
    )
    st.dataframe(stats, hide_index=True)
    with st.expander("Absolute pixel difference (amplified 4×)"):
        difference = np.abs(original.astype(np.int16) - restored.astype(np.int16))
        st.image(np.clip(difference * 4, 0, 255).astype(np.uint8), caption="Difference ×4", width="stretch")
    st.download_button("Download compressed file", archive, "compressed_image.npz", "application/octet-stream", on_click="ignore")
    st.download_button("Download reconstructed PNG", _png_bytes(restored), "reconstructed.png", "image/png", on_click="ignore")


def run_lossless_compression(img, channels) -> None:
    st.subheader("Lossless PNG Compression and Decompression")
    mode = st.radio("Operation", ["Compress", "Decompress"], horizontal=True, key="lossless_mode")
    if mode == "Decompress":
        st.caption("Upload an 8-bit RGB PNG from this tool. No original image is needed.")
        uploaded = st.file_uploader("Lossless PNG", type=["png"], key="lossless_upload")
        if uploaded is None:
            st.session_state.pop(f"approved_{selected_tool}_decompress", None)
            return
        if not tool_ready("Decompress PNG", uploaded.getvalue(), key="decompress"):
            return
        try:
            restored = decompress_png(uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        st.image(restored, caption="Decoded RGB pixels", width="stretch")
        st.caption(f"{restored.shape[1]} × {restored.shape[0]} pixels · {restored.nbytes:,} raw RGB bytes")
        st.download_button("Download uncompressed BMP", encode_bmp(restored), "decompressed.bmp", "image/bmp", on_click="ignore")
        return

    img = image_picker(selected_tool)
    if img is None:
        return
    st.caption(
        "Our custom PNG encoder uses reversible row filters, LZ77, and Huffman coding to preserve every 8-bit RGB pixel. "
        "Higher compression levels spend more effort reducing file size without changing image quality. "
        "The app converts uploads to RGB and rounds sample values to 8-bit pixels before encoding; "
        "source metadata, transparency, and original file bytes are not preserved."
    )
    with st.form("lossless_settings"):
        level = st.slider("PNG compression level", 0, 9, 9)
        submitted = st.form_submit_button("Compress and verify")
    if not submitted:
        return
    original = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
    try:
        with st.spinner("Encoding and verifying pixels with the custom PNG codec..."):
            encoded = compress_png(original, level)
            restored = decompress_png(encoded)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not np.array_equal(original, restored):
        st.error("Pixel verification failed. No download was generated.")
        return
    st.success("Verified: every decoded RGB pixel matches exactly. MSE = 0.")
    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original 8-bit RGB", width="stretch")
    with right:
        st.image(restored, caption="Decoded PNG — identical pixels", width="stretch")
    raw_col, png_col, ratio_col = st.columns(3)
    raw_col.metric("Raw RGB size", f"{original.nbytes:,} bytes")
    png_col.metric("PNG size", f"{len(encoded):,} bytes")
    ratio_col.metric("Raw RGB / PNG", f"{original.nbytes / len(encoded):.2f}×")
    st.caption("A ratio below 1 means the PNG is larger than raw RGB. Small or noisy images may not shrink; PNG may also be larger than JPEG.")
    st.download_button("Download lossless PNG", encoded, "lossless.png", "image/png", on_click="ignore")


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

    if not tool_ready("Apply filter", img, (filter_name, kwargs)):
        return

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


@st.cache_data(show_spinner=False, max_entries=3)
def _restore_preview(img, settings):
    restored, mask = restore_photo(img, **settings)
    return restored, mask, compress_png(restored, level=1)


def run_old_photo_restoration(img, channels) -> None:
    st.subheader("Old Photo Restoration")
    st.caption("Switch each step on or off. Click Restore photo to apply your settings to the original photo.")
    if img.shape[0] * img.shape[1] > 4_000_000:
        st.error("Please use a photo with at most 4 million pixels.")
        return
    settings = {}
    left, right = st.columns(2)
    with left:
        settings["lines"] = st.toggle("Remove white stripes / scan lines", key="restore_lines")
        settings["line_mode"] = st.selectbox("Line removal method",
            ["Automatic white stripes", "Periodic banding (FFT)"], disabled=not settings["lines"])
        settings["line_threshold"] = st.slider("Stripe threshold (lower detects fainter lines)",
            1, 40, 5, disabled=not settings["lines"] or settings["line_mode"] != "Automatic white stripes")
        settings["period"] = st.slider("Scan-line spacing (pixels)", 2.0, 100.0, 12.0, 0.5,
                                        disabled=not settings["lines"] or settings["line_mode"] != "Periodic banding (FFT)")
        settings["line_strength"] = st.slider("Line removal strength", 0.0, 1.0, 1.0, 0.05,
                                               disabled=not settings["lines"])
        st.caption("Automatic mode repairs thin white stripes, including slight scan skew. Use FFT for evenly repeating bands. Inspect the mask for real horizontal details.")
        settings["tint"] = st.toggle("Correct yellow tint / color cast", key="restore_tint")
        settings["tint_strength"] = st.slider("Color correction strength", 0.0, 1.0, 0.6, 0.05,
                                               disabled=not settings["tint"])
        st.caption("Automatic channel balancing assumes the scene averages toward neutral; reduce strength for naturally warm scenes.")
        settings["contrast"] = st.toggle("Recover faded contrast", key="restore_contrast")
        settings["contrast_strength"] = st.slider("Contrast recovery strength", 0.0, 1.0, 0.6, 0.05,
                                                   disabled=not settings["contrast"])
    with right:
        settings["denoise"] = st.toggle("Reduce grain / noise", key="restore_denoise")
        settings["noise_strength"] = st.slider("Noise reduction strength", 1.0, 20.0, 7.0, 1.0,
                                                disabled=not settings["denoise"])
        settings["scratches"] = st.toggle("Repair white dots / scratches", key="restore_scratches")
        settings["scratch_threshold"] = st.slider("Scratch threshold (higher selects less)", 10, 150, 30,
                                                   disabled=not settings["scratches"])
        settings["scratch_size"] = st.select_slider("Scratch detection width (pixels)", [3, 5, 7, 11, 15, 21],
                                                    value=11, disabled=not settings["scratches"])
        settings["repair_dark_scratches"] = st.checkbox("Also repair dark scratches", disabled=not settings["scratches"])
        st.caption("Automatic repair can mistake eyes, hair, or texture for scratches. Inspect the mask below and disable this step if needed.")
    st.caption("Order: scan lines → scratches → grain → color balance → contrast. Missing features cannot be recovered exactly.")
    if not tool_ready("Restore photo", img, settings):
        return
    try:
        with st.spinner("Restoring photo and preparing PNG..."):
            restored, mask, png = _restore_preview(img, settings)
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return
    original = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
    before, after = st.columns(2)
    before.image(original, caption="Original", width="stretch")
    after.image(restored, caption="Restored preview", width="stretch")
    if not any(settings[k] for k in ("lines", "tint", "contrast", "denoise", "scratches")):
        st.info("All steps are off: the output matches the original RGB pixels.")
    if settings["scratches"] or (settings["lines"] and settings["line_mode"] == "Automatic white stripes"):
        with st.expander("Inspect stripe and scratch repair mask", expanded=False):
            st.image(mask, caption="White pixels are selected for repair", clamp=True, width="stretch")
            st.caption(f"Selected {np.mean(mask > 0):.2%} of pixels.")
    with st.expander("Before / after wipe comparison"):
        position = st.slider("Original on left · restored on right", 0, 100, 50)
        split = round(original.shape[1] * position / 100)
        wipe = restored.copy()
        wipe[:, :split] = original[:, :split]
        st.image(wipe, width="stretch")
    st.download_button("Download restored PNG", png, "restored_photo.png", "image/png", on_click="ignore")


TOOL_RUNNERS = {
    "Color channel analyzer and histogram": run_color_channel_analyzer,
    "Partial image reconstruction": run_partial_reconstruction,
    "Color space comparison": run_color_space_comparison,
    "Compression and decompression": run_compression,
    "Lossless PNG compression": run_lossless_compression,
    "Filtering": run_spatial_filtering,
    "Old Photo Restoration": run_old_photo_restoration,
}


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bindu · Image workspace", page_icon="◉", layout="wide")
apply_style()
with st.sidebar:
    render_brand()

# --- Tool selection ---
st.sidebar.subheader("Explore tools")
selected_tool = st.sidebar.radio("Choose a tool", TOOLS, format_func=lambda tool: TOOL_DETAILS[tool][0], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("A little curiosity. A new perspective.")
st.sidebar.caption("Bindu · Color, detail & discovery")

render_header(selected_tool)
if st.session_state.get("active_tool") != selected_tool:
    for key in list(st.session_state):
        if key.startswith("approved_"):
            del st.session_state[key]
    st.session_state["active_tool"] = selected_tool

if selected_tool in ("Compression and decompression", "Lossless PNG compression"):
    TOOL_RUNNERS[selected_tool](None, None)
else:
    img = image_picker(selected_tool)
    if img is not None:
        TOOL_RUNNERS[selected_tool](img, split_channels(img))
