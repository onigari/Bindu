import numpy as np
import matplotlib.pyplot as plt


def compute_fft(channel: np.ndarray) -> np.ndarray:
    """2D FFT of a single channel, shifted so DC sits at the center.

    Returns a complex-valued HxW array.
    """
    f = np.fft.fft2(channel)
    f_shifted = np.fft.fftshift(f)
    return f_shifted


def compute_inverse_fft(fft_shifted: np.ndarray) -> np.ndarray:
    """Inverse of compute_fft(): shifted complex spectrum -> real spatial channel.

    Returns a real-valued HxW array (small imaginary residue from
    floating-point error is discarded).
    """
    unshifted = np.fft.ifftshift(fft_shifted)
    spatial = np.fft.ifft2(unshifted)
    return np.real(spatial)


def compute_magnitude_spectrum(fft_shifted: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """Magnitude of a shifted FFT result, optionally log-scaled for display.

    Use log_scale=False for direct numerical work where the true magnitude is required.
    """
    magnitude = np.abs(fft_shifted)
    if log_scale:
        return np.log1p(magnitude) # +1 avoids log(0) for exact-zero bins

    return magnitude


def compute_channel_ffts(channels: dict) -> dict:
    """FFT every channel in a {'R','G','B'} dict.

    Returns {'R': fft_array, 'G': fft_array, 'B': fft_array}, each a
    shifted complex HxW array from compute_fft().
    """
    return {name: compute_fft(channel) for name, channel in channels.items()}


def plot_magnitude_spectra(channels: dict, title: str = "Frequency Content (log-magnitude)"):
    """Show the R/G/B magnitude spectra side by side for visual comparison.

    channels: dict like {'R': arr, 'G': arr, 'B': arr} from split_channels().
    """
    names = list(channels.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        fft_result = compute_fft(channels[name])
        spectrum = compute_magnitude_spectrum(fft_result)
        im = ax.imshow(spectrum, cmap="viridis")
        ax.set_title(f"{name} channel spectrum")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def channel_energy(signal: np.ndarray, is_fft: bool = False) -> float:
    """Total energy of a channel, either in the spatial or frequency domain.
    By Parseval's theorem these are proportional.

    If the `signal` array is a channel, just sum the squares of the pixel values.
    If the `signal` array is an FFT result, sum the squares of the magnitudes.
    """
    if is_fft:
        return float(np.sum(np.abs(signal) ** 2))
    return float(np.sum(signal.astype(np.float64) ** 2))    # channel values are uint8, so convert to float64


def compare_channel_energy(channels: dict) -> dict:
    """Frequency-domain energy per channel, and each channel's share of the total.

    Returns {'R': {'energy': ..., 'fraction': ...}, 'G': {...}, 'B': {...}}.
    """
    ffts = compute_channel_ffts(channels)
    energies = {name: channel_energy(fft, is_fft=True) for name, fft in ffts.items()}
    total = sum(energies.values())
    return {
        name: {"energy": e, "fraction": (e / total if total > 0 else 0.0)} for name, e in energies.items()
    }
