import cv2
import numpy as np


def gaussian_blur_channel(channel: np.ndarray, ksize: int = 0, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur on a single HxW channel.

    ksize=0 lets OpenCV derive an odd kernel size from sigma automatically.
    Input/output stay float64 (no clipping) so results compose cleanly with
    the rest of the float64 pipeline.
    """
    return cv2.GaussianBlur(channel.astype(np.float64), (ksize, ksize), sigmaX=sigma)


def box_blur_channel(channel: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Simple averaging (box) blur on a single HxW channel."""
    return cv2.blur(channel.astype(np.float64), (ksize, ksize))


def laplacian_sharpen_channel(channel: np.ndarray, ksize: int = 3, scale: float = 1.0) -> np.ndarray:
    """Sharpen via Laplacian kernel: channel - scale * Laplacian(channel).

    The Laplacian highlights local intensity changes (edges); subtracting it
    pulls edges apart from their neighbors, increasing local contrast.
    """
    channel = channel.astype(np.float64)
    laplacian = cv2.Laplacian(channel, ddepth=cv2.CV_64F, ksize=ksize)
    sharpened = channel - (scale * laplacian)
    return sharpened


def unsharp_mask_channel(channel: np.ndarray, ksize : int = 0, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray:
    """Classic unsharp mask: original + amount * (original - blurred).

    Blur first to isolate low-frequency content, subtract it out to get a
    "detail" layer, then add that detail back in, amplified by `amount`.
    """
    channel = channel.astype(np.float64)
    blurred = cv2.GaussianBlur(channel, (ksize, ksize) ,sigmaX=sigma)
    detail = channel - blurred
    return channel + (amount * detail)


def filter_channels(channels: dict, filter_fn, **kwargs) -> dict:
    """Apply filter_fn independently to each channel in a {'R','G','B'} dict.

    filter_fn should be one of the single-channel filters above (or any
    function taking a HxW float64 array and returning one). kwargs are
    forwarded to filter_fn for each channel.

    Returns a new dict, same keys as `channels`.
    """
    return {name: filter_fn(channel, **kwargs) for name, channel in channels.items()}


def gaussian_blur(img: np.ndarray, ksize: int = 0, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur applied to the full HxWx3 image at once."""
    return cv2.GaussianBlur(img.astype(np.float64), (ksize, ksize), sigmaX=sigma)


def box_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Box blur applied to the full HxWx3 image at once."""
    return cv2.blur(img.astype(np.float64), (ksize, ksize))


def laplacian_sharpen(img: np.ndarray, ksize: int = 3, scale: float = 1.0) -> np.ndarray:
    """Laplacian sharpen applied to the full HxWx3 image at once."""
    img = img.astype(np.float64)
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=ksize)
    sharpened = img - (scale * laplacian)
    return sharpened


def unsharp_mask(img: np.ndarray, ksize : int = 0, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask applied to the full HxWx3 image at once."""
    img = img.astype(np.float64)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    detail = img - blurred
    return img + (amount * detail)


_PERCHANNEL_TO_MERGED = {
    gaussian_blur_channel: gaussian_blur,
    box_blur_channel: box_blur,
    laplacian_sharpen_channel: laplacian_sharpen,
    unsharp_mask_channel: unsharp_mask,
}


def compare_perchannel_vs_merged(img: np.ndarray, channels: dict, filter_fn, **kwargs) -> dict:
    """Filter per-channel and filter the merged image with the matching
    whole-image filter, then compare the two results.

    filter_fn: one of the single-channel filters above (its merged
    counterpart is looked up automatically).

    Returns {'max_abs_diff': ..., 'mean_abs_diff': ..., 'matches': bool}.
    `matches` uses a small tolerance to account for floating-point rounding,
    not because the two paths are expected to differ mathematically.
    """
    from imgutil.core import merge_channels

    merged_fn = _PERCHANNEL_TO_MERGED.get(filter_fn)
    if merged_fn is None:
        raise ValueError(
            f"No merged counterpart registered for {filter_fn.__name__}; "
            f"add it to _PERCHANNEL_TO_MERGED."
        )

    per_channel_result = merge_channels(filter_channels(channels, filter_fn, **kwargs))
    merged_result = merged_fn(img, **kwargs)

    diff = np.abs(per_channel_result - merged_result)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "matches": max_abs_diff < 1e-8,
    }


def filter_kernel(filter_name: str, **params) -> np.ndarray:
    """Return the effective 2D weights of the app's float64 linear filters.

    Probe GaussianBlur with an isolated impulse so automatic support and
    coefficients come from the installed OpenCV implementation itself.
    """
    if filter_name in ("Gaussian blur", "Unsharp mask"):
        sigma = float(params.get("sigma", 2.0))
        ksize = int(params.get("ksize", 0))
        radius = max(ksize, int(np.ceil(8 * sigma)) + 8)
        impulse = np.zeros((1, 2 * radius + 1), dtype=np.float64)
        impulse[0, radius] = 1
        response = cv2.GaussianBlur(impulse, (ksize, 1), sigmaX=sigma)[0]
        support = np.flatnonzero(response)
        weights = response[support[0]:support[-1] + 1]
        kernel = np.outer(weights, weights)
        if filter_name == "Unsharp mask":
            amount = float(params.get("amount", 1.0))
            kernel *= -amount
            kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] += 1 + amount
        return kernel
    if filter_name == "Box blur":
        size = int(params.get("ksize", 5))
        return np.full((size, size), 1.0 / (size * size))
    if filter_name == "Laplacian sharpen":
        size = int(params.get("ksize", 3))
        support = max(3, size)
        impulse = np.zeros((support * 2 + 1, support * 2 + 1), dtype=np.float64)
        impulse[support, support] = 1
        response = laplacian_sharpen_channel(impulse, ksize=size, scale=params.get("scale", 1.0))
        radius = support // 2
        return response[support-radius:support+radius+1, support-radius:support+radius+1]
    raise ValueError(f"Unknown filter: {filter_name}")
