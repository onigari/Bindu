import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(channel: np.ndarray, bins: int = 256, value_range=(0, 256)):
    """Compute a histogram for a single 2D channel array."""
    hist, bin_edges = np.histogram(channel.ravel(), bins=bins, range=value_range)
    return hist, bin_edges


def plot_channel_histograms(channels: dict, title: str = "Channel Histograms", ax=None):
    """Overlay R/G/B histograms on one plot for easy comparison.

    channels: dict like {'R': arr, 'G': arr, 'B': arr} from split_channels().
    """
    colors = {"R": "red", "G": "green", "B": "blue"}
    fig = None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    for name, channel in channels.items():
        hist, edges = compute_histogram(channel)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, color=colors.get(name, "black"), label=name, alpha=0.8)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if fig is not None:
        fig.tight_layout()
        return fig
    
    return ax


def plot_histogram_comparison(channels_a: dict, channels_b: dict,
                               label_a: str = "Image A", label_b: str = "Image B"):
    """Side-by-side histogram comparison between two images' channel sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_channel_histograms(channels_a, title=label_a, ax=axes[0])
    plot_channel_histograms(channels_b, title=label_b, ax=axes[1])
    fig.tight_layout()
    return fig
