"""
Run: py demo_3.py path/to/image.jpg
If no path is given, generates a synthetic test image.

Stage D: spatial filtering per channel.

Applies Gaussian blur, box blur, and unsharp mask, and for each filter
verifies that filtering each channel independently gives the same result
as filtering the merged RGB image at once (as expected, since these are
all per-pixel/per-channel linear filters with no cross-channel mixing).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from imgutil.core import *
from imgutil.histograms import *
from imgutil.filtering import (
    gaussian_blur_channel, gaussian_blur,
    box_blur_channel, box_blur,
    unsharp_mask_channel, unsharp_mask,
    filter_channels, compare_perchannel_vs_merged,
)


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

    # Filters under test: (label, per-channel fn, merged fn, kwargs)
    filters = [
        ("Gaussian blur", gaussian_blur_channel, gaussian_blur, {"sigma": 3.0}),
        ("Box blur", box_blur_channel, box_blur, {"ksize": 7}),
        ("Unsharp mask", unsharp_mask_channel, unsharp_mask, {"sigma": 2.0, "amount": 1.5}),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 0: original + each filter's merged-image result
    axes[0, 0].imshow(np.clip(img, 0, 255).astype(np.uint8))
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    results = {}
    for i, (label, chan_fn, merged_fn, kwargs) in enumerate(filters):
        # merged-image result, for display
        merged_result = merged_fn(img, **kwargs)
        axes[0, i + 1].imshow(np.clip(merged_result, 0, 255).astype(np.uint8))
        axes[0, i + 1].set_title(label)
        axes[0, i + 1].axis("off")

        # per-channel vs merged comparison (the actual Stage D question)
        stats = compare_perchannel_vs_merged(img, channels, chan_fn, **kwargs)
        results[label] = stats

        per_channel_result = merge_channels(filter_channels(channels, chan_fn, **kwargs))
        diff_map = np.abs(per_channel_result - merged_result).sum(axis=-1)  # sum over R,G,B

        im = axes[1, i + 1].imshow(diff_map, cmap="magma")
        axes[1, i + 1].set_title(
            f"|per-channel - merged|\nmax={stats['max_abs_diff']:.2e}"
        )
        axes[1, i + 1].axis("off")
        fig.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5,
        "Row 2 shows |per-channel result\n- merged result| per filter.\n\n"
        "Expected: ~0 everywhere, since\n"
        "these filters don't mix channels.",
        ha="center", va="center", wrap=True, fontsize=10,
    )

    fig.suptitle("Stage D: Spatial Filtering, Per-Channel vs. Merged")
    fig.tight_layout()

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_3.png")
    fig.savefig(out_path, dpi=120)

    print(f"Saved figure to {out_path}")
    print()
    print("Per-channel vs. merged filtering comparison:")
    for label, stats in results.items():
        print(
            f"  {label:15s} max_abs_diff={stats['max_abs_diff']:.3e}  "
            f"mean_abs_diff={stats['mean_abs_diff']:.3e}  matches={stats['matches']}"
        )


if __name__ == "__main__":
    main()
