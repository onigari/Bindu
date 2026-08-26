import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(channel: np.ndarray, bins: int = 256, value_range=(0, 256)):
    """Compute a histogram for a single 2D channel array.
    
    Returns (hist, bin_edges).
    """
    hist, bin_edges = np.histogram(channel.ravel(), bins=bins, range=value_range)
    return hist, bin_edges

def compute_combined_histogram(channels: dict, bins: int = 256, value_range=(0, 256)):
    """Add the R, G, and B histograms bin-by-bin into one total curve.

    Returns (hist, bin_edges), same shape as compute_histogram().
    """
    if not channels:
        raise ValueError("compute_combined_histogram() got an empty channels dict")

    channel_iter = iter(channels.values())
    total, edges = compute_histogram(next(channel_iter), bins=bins, value_range=value_range)
    for channel in channel_iter:
        hist, edges = compute_histogram(channel, bins=bins, value_range=value_range)
        total = total + hist
    return total, edges

def compute_luminance_histogram(channels: dict, bins: int = 256, value_range=(0, 256)):
    """Histogram of perceptual luminance per pixel.
    Calculated using the standard ITU-R BT.601 formula (0.299R + 0.587G + 0.114B).

    Different from compute_combined_histogram(): this blends the three
    channels *per pixel first* into a single brightness value
    """
    luminance = 0.299 * channels["R"] + 0.587 * channels["G"] + 0.114 * channels["B"]
    hist, edges = np.histogram(luminance.ravel(), bins=bins, range=value_range)
    return hist, edges


def plot_channel_histograms(channels: dict, title: str = "Channel Histograms", ax=None,
                            fill_axis: bool = True, show_combined: bool = True, show_luminance: bool = False):
    """Overlay R/G/B histograms on one plot along with their combined and/or luminance histogram.

    channels: dict like {'R': arr, 'G': arr, 'B': arr} from split_channels().
    """
    colors = {"R": "red", "G": "green", "B": "blue"}
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    for name, channel in channels.items():
        hist, edges = compute_histogram(channel)
        centers = (edges[:-1] + edges[1:]) / 2
        color = colors.get(name, "black")
        ax.plot(centers, hist, color=color, label=name, alpha=0.8)
        if fill_axis:
            ax.fill_between(centers, hist, 0, color=color, alpha=0.35)  # if want to fill color up to x-axis

    if show_combined:
        total_hist, edges = compute_combined_histogram(channels)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, total_hist, color="black", linestyle="--",
                    label="Total", alpha=0.9, linewidth=1.5)
                    
    if show_luminance:
        lum_hist, edges = compute_luminance_histogram(channels)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, lum_hist, color="gray", linestyle="-.",
                    label="Luminance", alpha=0.9, linewidth=1.5)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if fig is not None:
        fig.tight_layout()
        return fig
    
    return ax


def plot_histogram_comparison(channels_a: dict, channels_b: dict, label_a: str = "Image A", label_b: str = "Image B"):
    """Side-by-side histogram comparison between the channel sets of two images."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_channel_histograms(channels_a, title=label_a, ax=axes[0])
    plot_channel_histograms(channels_b, title=label_b, ax=axes[1])
    fig.tight_layout()
    return fig
