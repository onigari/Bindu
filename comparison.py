"""Browser-local before/after comparisons; dragging never reruns processing."""
import base64
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


def image_uri(image):
    pixels = np.clip(image, 0, 255).astype(np.uint8)
    if pixels.ndim == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise ValueError("Could not encode comparison preview.")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def render_comparison(before, after, *, before_label="Original", after_label="Result"):
    """Compare aligned images, including grayscale channel outputs."""
    if before.shape[:2] != after.shape[:2]:
        raise ValueError("Comparison images must have matching dimensions.")
    template = Path(__file__).with_name("comparison.html").read_text(encoding="utf-8")
    for marker, value in {
        "__BEFORE_IMAGE__": image_uri(before), "__AFTER_IMAGE__": image_uri(after),
        "__BEFORE_LABEL__": escape(before_label, quote=True),
        "__AFTER_LABEL__": escape(after_label, quote=True),
    }.items():
        template = template.replace(marker, value)
    st.iframe(template, height=390)

