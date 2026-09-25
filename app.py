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
from animated_histograms import (
    animated_histogram, animated_spectrum, animated_reconstruction,
    prepare_spectrum_channels,
)
from filter_animation import animate_filter
from comparison import render_comparison
from ui import apply_style, render_brand, render_header, TOOL_DETAILS

from imgutil.core import *
from imgutil.color_spaces import rgb_to_ycbcr
from imgutil.compression import compress_image, decompress_image
from imgutil.lossless import compress_png, decompress_png, encode_bmp
from imgutil.wavelet import compress_wavelet, decompress_wavelet
from imgutil.restoration import restore_photo
from imgutil.noise import add_noise, remove_noise, rgb_pixels
from imgutil.histograms import *
from imgutil.frequency import *
from imgutil.filtering import (
    gaussian_blur_channel, gaussian_blur,
    box_blur_channel, box_blur,
    laplacian_sharpen_channel, laplacian_sharpen,
    unsharp_mask_channel, unsharp_mask,
    filter_channels, compare_perchannel_vs_merged, filter_kernel,
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
    "Lossless wavelet compression",
    "Filtering",
    "White Scratch Removal",
    "Noise addition and removal",
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Rebuilt on each Streamlit run so comparisons always match the current result.
page_comparisons = []


def queue_comparison(before, after, *, after_label, before_label="Original"):
    page_comparisons.append((before, after, before_label, after_label))


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
    with st.sidebar:
        st.subheader("Your image")
        source_mode = st.radio("Load from", ["Upload", "Sample gallery"],
                               key="workspace_source", horizontal=True)
        img = None
        if source_mode == "Upload":
            uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"],
                                        key="workspace_upload")
            if uploaded is not None:
                try:
                    img = load_uploaded_image(uploaded)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            sample = st.selectbox("Sample image", list(SAMPLE_IMAGES), key="workspace_sample")
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
    channels = split_channels(img)
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
            queue_comparison(img, display_img, after_label=f"{name} channel")

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

    st.caption("Live inverse Fourier reconstruction uses a preview up to 128 pixels per side. Each step adds a frequency row to the image; scrub backward to remove it.")
    if not tool_ready("Reconstruct image", img):
        return
    channels = split_channels(img)
    prepared, _ = prepare_spectrum_channels(channels)
    # The browser inverse transform consumes unshifted complex coefficients.
    channel_ffts = {name: np.fft.fft2(channel) for name, channel in prepared.items()}

    def _render_combo(selected: list, col) -> None:
        with col:
            result = np.stack([channels[name] if name in selected else np.zeros_like(channels[name])
                               for name in ("R", "G", "B")], axis=-1)
            queue_comparison(img, result, after_label=" + ".join(selected))
            animated_reconstruction({name: channel_ffts[name] for name in selected}, " + ".join(selected))

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
    channels = split_channels(img)
    converted = rgb_to_ycbcr(img)
    ycbcr = {name: converted[:, :, i] for i, name in enumerate(("Y", "Cb", "Cr"))}
    spaces = [("RGB", channels), ("YCbCr", ycbcr)]
    st.caption("All channel previews use the same grayscale scale. Mid-gray Cb/Cr indicates neutral color.")
    for label, space in spaces:
        st.markdown(f"**{label} channels**")
        for col, (name, channel) in zip(st.columns(3), space.items()):
            with col:
                st.image(np.clip(np.rint(channel), 0, 255).astype(np.uint8), caption=name, width="stretch")
                queue_comparison(img, np.rint(channel), after_label=f"{label} · {name}")

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


def compare_reference(reference, restored):
    if reference is None:
        st.caption("Add an original image above to enable the before / after slider.")
        return
    try:
        original = load_uploaded_image(reference)
    except ValueError as exc:
        st.error(str(exc))
        return
    if original.shape[:2] != restored.shape[:2]:
        st.warning("The reference must have the same width and height as the decoded image.")
        return
    queue_comparison(original, restored, after_label="Decoded image")


def run_compression(img, channels) -> None:
    st.subheader("Compression and Decompression")
    mode = st.radio("Operation", ["Compress", "Decompress"], horizontal=True)
    if mode == "Decompress":
        st.caption("Upload a Fourier .npz file downloaded from this tool. The original image is not required.")
        with st.sidebar:
            st.subheader("Your compressed image")
            uploaded = st.file_uploader("Compressed image", type=["npz"], key="compressed_image")
            reference = st.file_uploader("Original image for comparison (optional)", type=["png", "jpg", "jpeg", "bmp"], key="fourier_reference")
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
        compare_reference(reference, restored)
        st.caption(f"{restored.shape[1]} × {restored.shape[0]} pixels")
        st.download_button("Download decompressed PNG", _png_bytes(restored), "decompressed.png", "image/png", on_click="ignore")
        return

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
    queue_comparison(original, restored, after_label="Fourier reconstruction")
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
    st.subheader("Lossless Wavelet Compression and Decompression")
    mode = st.radio("Operation", ["Compress", "Decompress"], horizontal=True, key="lossless_mode")
    if mode == "Decompress":
        st.caption("Upload a wavelet .iwv archive from this tool, or a legacy 8-bit RGB PNG. No original image is needed.")
        with st.sidebar:
            st.subheader("Your compressed image")
            uploaded = st.file_uploader("Lossless archive", type=["iwv", "png"], key="lossless_upload")
            reference = st.file_uploader("Original image for comparison (optional)", type=["png", "jpg", "jpeg", "bmp"], key="lossless_reference")
        if uploaded is None:
            st.session_state.pop(f"approved_{selected_tool}_decompress", None)
            return
        if not tool_ready("Decompress image", uploaded.getvalue(), key="decompress"):
            return
        try:
            data = uploaded.getvalue()
            restored = decompress_png(data) if data.startswith(b'\x89PNG\r\n\x1a\n') else decompress_wavelet(data)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.image(restored, caption="Decoded RGB pixels", width="stretch")
        compare_reference(reference, restored)
        st.caption(f"{restored.shape[1]} × {restored.shape[0]} pixels · {restored.nbytes:,} raw RGB bytes")
        st.download_button("Download uncompressed BMP", encode_bmp(restored), "decompressed.bmp", "image/bmp", on_click="ignore")
        return

    if img is None:
        return
    st.caption(
        "Reversible integer Haar wavelets separate the image into averages and detail signals. "
        "Every coefficient is retained and compressed with LZ77 and Huffman coding, preserving every 8-bit RGB pixel. "
        "Higher compression levels spend more effort reducing file size without changing image quality. "
        "The app converts uploads to RGB and rounds sample values to 8-bit pixels before encoding; "
        "source metadata, transparency, and original file bytes are not preserved."
    )
    with st.form("lossless_settings"):
        levels = st.slider("Wavelet decomposition levels", 1, 8, 4)
        level = st.slider("Compression effort", 0, 9, 9)
        st.caption("More wavelet levels analyze larger image regions; they do not change quality or guarantee a smaller file. Downloads use the custom .iwv format.")
        submitted = st.form_submit_button("Compress and verify")
    if not submitted:
        return
    original = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
    try:
        with st.spinner("Applying integer wavelets, compressing, and verifying every pixel..."):
            encoded = compress_wavelet(original, level=level, levels=levels)
            restored = decompress_wavelet(encoded)
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
        st.image(restored, caption="Decoded wavelet archive — identical pixels", width="stretch")
    queue_comparison(original, restored, after_label="Decoded wavelet · identical pixels")
    raw_col, png_col, ratio_col = st.columns(3)
    raw_col.metric("Raw RGB size", f"{original.nbytes:,} bytes")
    png_col.metric("Wavelet archive size", f"{len(encoded):,} bytes")
    ratio_col.metric("Raw RGB / archive", f"{original.nbytes / len(encoded):.2f}×")
    st.caption("A ratio below 1 means the archive is larger than raw RGB. Small or noisy images may grow. Open .iwv files with this tool's Decompress operation.")
    st.download_button("Download lossless wavelet archive", encoded, "lossless.iwv", "application/octet-stream", on_click="ignore")


def render_kernel_visualizer(filter_name, params):
    kernel = filter_kernel(filter_name, **params)
    size = kernel.shape[0]
    center = size // 2
    st.subheader("Kernel visualizer")
    st.caption("Updates with your settings. Click Apply filter below to process the image.")
    left, right = st.columns([2, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        signed = bool(np.any(kernel < 0))
        limit = float(np.max(np.abs(kernel))) or 1.0
        plot = ax.imshow(kernel, cmap="RdBu_r" if signed else "viridis",
                         vmin=-limit if signed else 0, vmax=limit, interpolation="nearest")
        if size <= 9:
            ax.set_xticks(range(size), labels=range(-center, center + 1))
            ax.set_yticks(range(size), labels=range(-center, center + 1))
            for row in range(size):
                for col in range(size):
                    rgba = plot.cmap(plot.norm(kernel[row, col]))
                    brightness = .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2]
                    ax.text(col, row, f"{kernel[row, col]:.3g}", ha="center", va="center",
                            fontsize=7, color="black" if brightness > .55 else "white")
        else:
            ax.set_xticks([0, center, size - 1], labels=[-center, 0, center])
            ax.set_yticks([0, center, size - 1], labels=[-center, 0, center])
        ax.set_xlabel("Column offset from center")
        ax.set_ylabel("Row offset from center")
        ax.set_title(f"{filter_name}: {size} x {size}")
        fig.colorbar(plot, ax=ax, label="Weight")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.metric("Kernel size", f"{size} x {size}")
        st.metric("Sum of weights", f"{kernel.sum():.6g}")
        st.metric("Center weight", f"{kernel[center, center]:.6g}")
        explanations = {
            "Gaussian blur": "Nearby pixels contribute more than distant pixels. Kernel size is chosen automatically from sigma by OpenCV.",
            "Box blur": "Every pixel in the neighborhood has the same weight; the output is their average.",
            "Unsharp mask": "Effective kernel: (1 + amount) times the center impulse, minus amount times the Gaussian kernel.",
            "Laplacian sharpen": "Effective kernel: center impulse minus scale times the Laplacian kernel. Size 1 uses a 3 x 3 Laplacian stencil.",
        }
        st.write(explanations[filter_name])
        st.caption("The center cell aligns with the output pixel. The same weights are used for each selected channel. Image edges use OpenCV's reflected border handling.")
    with st.expander("All kernel weights"):
        st.dataframe(kernel, width="stretch")
        st.download_button("Download kernel CSV", "\n".join(",".join(f"{value:.17g}" for value in row) for row in kernel),
                           "filter_kernel.csv", "text/csv", on_click="ignore")


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

    channels = split_channels(img)
    render_kernel_visualizer(filter_name, kwargs)

    animate_filter(img, filter_kernel(filter_name, **kwargs))

    chan_fn = config["channel_fn"]
    merged_fn = config["merged_fn"]

    r_only_result = make_single_channel_filtered_image(channels, chan_fn, "R", **kwargs)
    g_only_result = make_single_channel_filtered_image(channels, chan_fn, "G", **kwargs)
    b_only_result = make_single_channel_filtered_image(channels, chan_fn, "B", **kwargs)
    merged_result = merged_fn(img, **kwargs)

    stats = compare_perchannel_vs_merged(img, channels, chan_fn, **kwargs)

    st.divider()

    st.subheader("Result")
    for col, name, result in zip(st.columns(3), ("R", "G", "B"),
                                 (r_only_result, g_only_result, b_only_result)):
        with col:
            st.image(np.clip(result, 0, 255).astype(np.uint8), caption=f"{filter_name} ({name} only)", width="stretch")
            queue_comparison(img, result, after_label=f"{filter_name} ({name} only)")
    st.image(np.clip(merged_result, 0, 255).astype(np.uint8), caption=f"{filter_name} (all channels)", width=400)
    queue_comparison(img, merged_result, after_label=f"{filter_name} (all channels)")

    st.divider()


@st.cache_data(show_spinner=False, max_entries=3)
def _restore_preview(img, settings):
    restored, mask = restore_photo(img, **settings)
    return restored, mask, compress_png(restored, level=1)


def run_white_scratch_removal(img, channels) -> None:
    st.caption("Repair small white dots and bright scratches using surrounding pixels.")
    if img.shape[0] * img.shape[1] > 4_000_000:
        st.error("Please use a photo with at most 4 million pixels.")
        return
    settings = {"scratches": True, "repair_dark_scratches": False}
    left, right = st.columns(2)
    with left:
        settings["scratch_threshold"] = st.slider(
            "Scratch threshold (higher selects less)", 10, 150, 30)
    with right:
        settings["scratch_size"] = st.select_slider(
            "Scratch detection width (pixels)", [3, 5, 7, 11, 15, 21], value=11)
    st.caption("Choose a detection width wider than the marks. Inspect the repair mask to check that real details are not selected.")
    if not tool_ready("Repair white dots / scratches", img, settings):
        return
    try:
        with st.spinner("Repairing white dots and scratches..."):
            restored, mask, png = _restore_preview(img, settings)
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return
    original = np.rint(np.clip(img, 0, 255)).astype(np.uint8)
    before, after = st.columns(2)
    before.image(original, caption="Original", width="stretch")
    after.image(restored, caption="White scratch removal preview", width="stretch")
    if not mask.any():
        st.info("No white dots or scratches detected with these settings.")
    with st.expander("Inspect scratch repair mask", expanded=False):
        st.image(mask, caption="White pixels are selected for repair", clamp=True, width="stretch")
        st.caption(f"Selected {np.mean(mask > 0):.2%} of pixels.")
    queue_comparison(original, restored, after_label="White scratch removal")
    st.download_button("Download repaired PNG", png, "white_scratch_removed.png", "image/png", on_click="ignore")


@st.cache_data(show_spinner=False, max_entries=3)
def _noise_preview(img, mode, kind, amount, seed, method, size, strength):
    original = rgb_pixels(img)
    noisy = add_noise(original, kind, amount, seed) if mode != "Remove noise" else original
    result = remove_noise(noisy, method, size, strength) if mode != "Add noise" else noisy
    return original, noisy, result


def run_noise_tool(img, channels):
    mode = st.radio("Operation", ["Add noise", "Remove noise", "Add then remove"],
                    horizontal=True, key="noise_mode")
    kind, amount, seed = "Gaussian", 20.0, 42
    method, size, strength = "Median", 3, 10.0
    if mode != "Remove noise":
        st.subheader("Add noise")
        kind = st.selectbox("Noise type", ["Gaussian", "Salt and pepper"])
        if kind == "Gaussian":
            amount = st.slider("Noise standard deviation (pixel levels)", 0.0, 80.0, 20.0, 1.0)
            st.caption("Adds independent brightness variation to each RGB channel.")
        else:
            amount = st.slider("Pixels affected (%)", 0.0, 50.0, 5.0, 0.5)
            st.caption("Randomly replaces pixels with black or white; the percentage is approximate.")
        seed = int(st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1))
        st.caption("Keep the same seed for repeatable noise, or change it for a different pattern.")
    if mode != "Add noise":
        st.subheader("Remove noise")
        method = st.selectbox("Denoising method", ["Median", "Gaussian blur", "Non-local means"])
        if method == "Non-local means":
            strength = st.slider("Denoising strength", 0.0, 30.0, 10.0, 1.0)
        else:
            size = st.select_slider("Filter width (pixels)", [3, 5, 7, 9, 11], value=3)
        st.caption("Median filtering suits salt-and-pepper noise. Gaussian blur smooths noise and detail. Non-local means uses similar patches; stronger settings may soften texture.")
    settings = (mode, kind, amount, seed, method, size, strength)
    action = {"Add noise": "Add noise", "Remove noise": "Remove noise", "Add then remove": "Add and remove noise"}[mode]
    if not tool_ready(action, img, settings, key="noise"):
        return
    try:
        with st.spinner("Processing noise..."):
            original, noisy, result = _noise_preview(img, *settings)
    except (ValueError, cv2.error) as exc:
        st.error(str(exc))
        return
    previews = [("Original", original)]
    if mode == "Add then remove":
        previews.append(("Added noise", noisy))
    previews.append(("Added noise" if mode == "Add noise" else "Denoised", result))
    for col, (label, pixels) in zip(st.columns(len(previews)), previews):
        col.image(pixels, caption=label, width="stretch")
    if mode == "Add then remove":
        st.download_button("Download noisy PNG", _png_bytes(noisy), "noisy.png", "image/png", on_click="ignore")
        queue_comparison(original, noisy, after_label="Added noise")
        # Use the actual noisy input as the before image for removal.
        queue_comparison(noisy, result, before_label="Noisy input", after_label="Denoised")
        st.caption("In the removal comparison, the before image is the noisy input.")
    else:
        queue_comparison(original, result, after_label=previews[-1][0])
    name = "noisy" if mode == "Add noise" else "denoised"
    st.download_button(f"Download {name} PNG", _png_bytes(result), f"{name}.png", "image/png", on_click="ignore")


TOOL_RUNNERS = {
    "Color channel analyzer and histogram": run_color_channel_analyzer,
    "Partial image reconstruction": run_partial_reconstruction,
    "Color space comparison": run_color_space_comparison,
    "Compression and decompression": run_compression,
    "Lossless wavelet compression": run_lossless_compression,
    "Filtering": run_spatial_filtering,
    "White Scratch Removal": run_white_scratch_removal,
    "Noise addition and removal": run_noise_tool,
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

render_header(selected_tool)
if st.session_state.get("active_tool") != selected_tool:
    for key in list(st.session_state):
        if key.startswith("approved_"):
            del st.session_state[key]
    st.session_state["active_tool"] = selected_tool

# Keep the shared uploader mounted, including in decompression mode, so
# Streamlit retains its file when navigating between tools.
img = image_picker(selected_tool)
if img is not None or selected_tool in ("Compression and decompression", "Lossless wavelet compression"):
    TOOL_RUNNERS[selected_tool](img, None)


if page_comparisons:
    st.divider()
    st.subheader("Before / after wipe comparison")
    st.caption("Drag the divider to compare each input image with its result.")
    for before, after, before_label, label in page_comparisons:
        render_comparison(before, after, before_label=before_label, after_label=label)


st.sidebar.divider()
st.sidebar.caption("Bindu · Image workspace")
