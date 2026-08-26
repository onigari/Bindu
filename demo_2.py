"""
Run: python demo_2.py path/to/image.jpg
If no path is given, generates the same synthetic test image as
demo_1.py for consistency.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from imgutil.core import *
from imgutil.frequency import *
from demo_1 import make_synthetic_test_image


def sanity_check_round_trip(channels: dict) -> None:
    """Verify FFT -> inverse FFT reproduces the original channel (within float error).

    This matters because Stage E (partial reconstruction) and Stage G
    (compression) both build on this exact round trip -- if it's not
    solid here, bugs downstream will be very confusing to trace.
    """
    print("\nFFT round-trip sanity check:")
    for name, channel in channels.items():
        fft_result = compute_fft(channel)
        reconstructed = compute_inverse_fft(fft_result)
        max_error = np.max(np.abs(channel - reconstructed))
        status = "OK" if max_error < 1e-6 else "FAILED"
        print(f"  {name}: max abs error = {max_error:.2e}  [{status}]")


def main():
    if len(sys.argv) > 1:
        img = load_image_rgb(sys.argv[1])
        print(f"Loaded image from {sys.argv[1]}, shape={img.shape}")
    else:
        img = make_synthetic_test_image()
        print("No path given -- using synthetic test image, shape=", img.shape)

    channels = split_channels(img)

    sanity_check_round_trip(channels)

    energy_report = compare_channel_energy(channels)
    print("\nFrequency-domain energy per channel:")
    for name, stats in energy_report.items():
        print(f"  {name}: energy={stats['energy']:.3e}  fraction={stats['fraction']:.1%}")

    names = ["R", "G", "B"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Plot Individual Channels
    for i, name in enumerate(names):
        # Row 0: Spatial
        axes[0, i].imshow(channel_as_grayscale_image(channels[name]), cmap="gray")
        axes[0, i].set_title(f"{name} Channel (Spatial)")
        axes[0, i].axis("off")

        # Row 1: Frequency
        fft_result = compute_fft(channels[name])
        spectrum = compute_magnitude_spectrum(fft_result)
        im = axes[1, i].imshow(spectrum, cmap="viridis")
        axes[1, i].set_title(f"{name} Spectrum")
        axes[1, i].axis("off")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    # Plot Combined Luminance
    # Calculate spatial luminance
    luminance = 0.299 * channels["R"] + 0.587 * channels["G"] + 0.114 * channels["B"]
    
    # Row 0: Spatial Luminance
    axes[0, 3].imshow(luminance, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title("Combined Luminance (Spatial)")
    axes[0, 3].axis("off")

    # Row 1: Combined Frequency Spectrum
    lum_fft = compute_fft(luminance)
    lum_spectrum = compute_magnitude_spectrum(lum_fft)
    im_lum = axes[1, 3].imshow(lum_spectrum, cmap="viridis")
    axes[1, 3].set_title("Combined Spectrum")
    axes[1, 3].axis("off")
    fig.colorbar(im_lum, ax=axes[1, 3], fraction=0.046, pad=0.04)

    fig.suptitle("Spatial vs. Frequency Content")
    fig.tight_layout()

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo2.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
