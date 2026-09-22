"""Full-range YCbCr conversion for RGB images with values in [0, 255]."""

import numpy as np


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """Return float64 Y, Cb, Cr channels using full-range BT.601 coefficients.

    Neutral chroma is 128. Values are clipped to the 8-bit display range;
    no chroma subsampling or compression is performed.
    """
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or 0 in rgb.shape:
        raise ValueError("Expected a nonempty RGB image with shape (H, W, 3).")
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("RGB values must be finite and in the range [0, 255].")
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ])
    return np.clip(rgb @ matrix.T + [0, 128, 128], 0, 255)
