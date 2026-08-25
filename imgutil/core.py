import cv2
import numpy as np


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from disk as float64 RGB, shape (H, W, 3), range [0,255]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR) # OpenCV reads images in BGR format, not RGB
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    return rgb.astype(np.float64)


def save_image_rgb(path: str, img: np.ndarray) -> None:
    """Save a float64 RGB image (range [0,255]) to disk."""
    clipped = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(clipped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def split_channels(img: np.ndarray) -> dict:
    """Split an RGB image into its three channels."""
    return {
        "R": img[:, :, 0].copy(),
        "G": img[:, :, 1].copy(),
        "B": img[:, :, 2].copy(),
    }


def merge_channels(channels: dict) -> np.ndarray:
    """Inverse of split_channels: {'R','G','B'} -> H×W×3 RGB image."""
    r, g, b = channels["R"], channels["G"], channels["B"]
    return np.stack([r, g, b], axis=-1)


def channel_as_grayscale_image(channel: np.ndarray) -> np.ndarray:
    """View a single channel as a displayable uint8 grayscale image."""
    return np.clip(channel, 0, 255).astype(np.uint8)


def channel_as_color_image(channel: np.ndarray, which: str) -> np.ndarray:
    """View a single channel 'as if' it were the only color present."""
    h, w = channel.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    idx = {"R": 0, "G": 1, "B": 2}[which]
    out[:, :, idx] = np.clip(channel, 0, 255).astype(np.uint8)
    return out
