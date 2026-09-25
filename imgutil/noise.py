"""Reproducible synthetic noise and RGB denoising."""
import cv2
import numpy as np


def rgb_pixels(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected a nonempty RGB image.")
    if image.shape[0] * image.shape[1] > 4_000_000:
        raise ValueError("Please use an image with at most 4 million pixels.")
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite pixels.")
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def add_noise(image, kind="Gaussian", amount=20.0, seed=42):
    pixels = rgb_pixels(image)
    rng = np.random.default_rng(seed)
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Noise amount must be finite and nonnegative.")
    if kind == "Gaussian":
        noisy = pixels.astype(float) + rng.normal(0, amount, pixels.shape)
        return np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)
    if kind == "Salt and pepper":
        if amount > 100:
            raise ValueError("Noise percentage must be between 0 and 100.")
        mask = rng.random(pixels.shape[:2])
        result = pixels.copy()
        result[mask < amount / 200] = 0
        result[(mask >= amount / 200) & (mask < amount / 100)] = 255
        return result
    raise ValueError("Unknown noise type.")


def remove_noise(image, method="Median", size=3, strength=10.0):
    pixels = rgb_pixels(image)
    if size not in (3, 5, 7, 9, 11):
        raise ValueError("Invalid filter width.")
    if not np.isfinite(strength) or not 0 <= strength <= 30:
        raise ValueError("Denoising strength must be between 0 and 30.")
    if method == "Median":
        return cv2.medianBlur(pixels, size)
    if method == "Gaussian blur":
        return cv2.GaussianBlur(pixels, (size, size), 0)
    if method == "Non-local means":
        if strength == 0:
            return pixels.copy()
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(bgr, None, strength, strength, 7, 21)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    raise ValueError("Unknown denoising method.")
