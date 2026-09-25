"""A pixel-by-pixel demonstration using original pixels and the actual kernel."""
import json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st


def animate_filter(image, kernel):
    h, w = image.shape[:2]
    height, width = h, w
    y, x = 0, 0
    radius = kernel.shape[0]//2
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    patch = padded[y:y+height+2*radius, x:x+width+2*radius]
    config = dict(width=width, height=height, originX=x, originY=y,
                  patchWidth=patch.shape[1], size=kernel.shape[0],
                  pixels=patch.ravel().tolist(), kernel=kernel.ravel().tolist())
    st.subheader("Live kernel calculation")
    st.caption(f"Calculating the entire {width} x {height} image at original resolution.")
    template = Path(__file__).with_name("filter_player.html").read_text(encoding="utf-8")
    engine = Path(__file__).with_name("filter_runtime.js").read_text(encoding="utf-8")
    st.iframe(template.replace("/*ENGINE*/", engine).replace("/*CONFIG*/", json.dumps(config)), height=820)
