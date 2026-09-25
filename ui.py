"""Presentation helpers for the Bindu workspace."""
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


TOOL_DETAILS = {
    "Color channel analyzer and histogram": ("Channel analyzer", "Explore the colors behind your image. Compare RGB channels, histograms, and frequency content."),
    "Partial image reconstruction": ("Reconstruction", "Rebuild your image from selected color channels and see how each contributes."),
    "Color space comparison": ("Color spaces", "Look at the same image through RGB and YCbCr color spaces."),
    "Compression and decompression": ("Fourier compression", "Find the balance between retained frequency detail and archive size."),
    "Lossless PNG compression": ("Lossless PNG", "Compress and recover your image with exact preservation of its 8-bit RGB pixels."),
    "Filtering": ("Image filters", "Fine-tune texture and detail with blur and sharpening controls."),
    "White Scratch Removal": ("White Scratch Removal", "Detect and repair white dots and bright scratches in your photographs."),
}


def apply_style():
    st.markdown(f"<style>{Path(__file__).with_name('ui.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor": "#101b2d", "axes.facecolor": "#101b2d",
        "axes.edgecolor": "#30435c", "axes.labelcolor": "#a5b6cb",
        "text.color": "#e6edf7", "xtick.color": "#a5b6cb", "ytick.color": "#a5b6cb",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#26364d", "grid.alpha": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "medium", "lines.linewidth": 1.8,
    })


def render_brand():
    st.markdown('<div class="brand"><span class="brand-mark">b.</span><div>Bindu<span class="brand-caption">IMAGE WORKSPACE</span></div></div>', unsafe_allow_html=True)


def render_header(tool):
    title, description = TOOL_DETAILS[tool]
    st.markdown('<div class="eyebrow">BINDU STUDIO &nbsp; / &nbsp; IMAGE TOOLS</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-description">{description}</p>', unsafe_allow_html=True)


def choose_sample(name):
    st.session_state["source_mode"] = "Sample gallery"
    st.session_state["sample_name"] = name


def render_welcome(samples):
    st.markdown('''<div class="welcome-panel">
        <div class="spectrum"><i></i><i></i><i></i></div>
        <div class="eyebrow">A CLOSER LOOK AT EVERY PIXEL</div>
        <h2>Every image has a story.<br>Explore whatâ€™s underneath.</h2>
        <p>Separate colors, discover patterns, and bring out the details.<br>
        Upload an image in the sidebar or start with a sample below.</p>
        <span class="format-badge">PNG Â· JPG Â· JPEG Â· BMP</span>
        </div>''', unsafe_allow_html=True)
    st.subheader("Start with a little inspiration")
    st.caption("Choose a sample to explore the workspace. You can switch tools at any time.")
    descriptions = ["Smooth color transitions & bold shapes", "Repeating patterns & frequency detail", "Soft gradients & subtle color shifts"]
    for column, (name, factory), description in zip(st.columns(3), list(samples.items())[:3], descriptions):
        with column, st.container(border=True):
            # Wide thumbnails preserve the sample's original pixels when opened.
            preview = factory().astype("uint8")
            st.image(preview[64:192, :], width="stretch")
            st.markdown(f"**{name}**")
            st.caption(description)
            st.button("Explore sample â†’", key=f"sample_{name}", on_click=choose_sample, args=(name,), width="stretch")
    st.caption("01  Load an image     Â·     02  Choose a tool     Â·     03  Explore and compare")
