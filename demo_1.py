"""
Run: python demo_1.py path/to/image.jpg
If no path is given, generates a synthetic test image.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from imgutil.core import *
from imgutil.histograms import *


def make_synthetic_test_image(size=256) -> np.ndarray:
    """A colorful gradient/shape image, useful when no sample photo is on hand."""
    x = np.linspace(0, 255, size)
    y = np.linspace(0, 255, size)
    xx, yy = np.meshgrid(x, y)
    r = xx
    g = yy
    b = 255 - (xx + yy) / 2
    img = np.stack([r, g, b], axis=-1)

    # add a few shapes so edges/frequency content is interesting later
    yy_idx, xx_idx = np.ogrid[:size, :size]
    circle_mask = (xx_idx - size * 0.7) ** 2 + (yy_idx - size * 0.3) ** 2 < (size * 0.15) ** 2
    img[circle_mask] = [255, 255, 255]

    return img.astype(np.float64)


def main():
    if len(sys.argv) > 1:
        img = load_image_rgb(sys.argv[1])
        print(f"Loaded image from {sys.argv[1]}, shape={img.shape}")
    else:
        img = make_synthetic_test_image()
        print("No path given -- using synthetic test image, shape=", img.shape)

    channels = split_channels(img)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(img.astype(np.uint8))
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    names = ["R", "G", "B"]
    for i, name in enumerate(names):
        axes[0, i + 1].imshow(channel_as_grayscale_image(channels[name]), cmap="gray")
        axes[0, i + 1].set_title(f"{name} channel")
        axes[0, i + 1].axis("off")

    plot_channel_histograms(channels, title="Combined Histogram", ax=axes[1, 0])
    axes[1, 0].axis("on")

    colors = {"R": "red", "G": "green", "B": "blue"}
    for i, name in enumerate(names):
        hist, edges = np.histogram(channels[name].ravel(), bins=256, range=(0, 256))
        centers = (edges[:-1] + edges[1:]) / 2
        axes[1, i + 1].plot(centers, hist, color=colors[name])
        axes[1, i + 1].fill_between(centers, hist, 0, color=colors[name], alpha=0.4)
        axes[1, i + 1].set_title(f"{name} histogram")
        axes[1, i + 1].set_xlabel("Intensity")

    fig.tight_layout()
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_1.png")
    fig.savefig(out_path, dpi=120)

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
