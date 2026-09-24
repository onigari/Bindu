"""Conservative, independently selectable restoration of 8-bit RGB photos."""
import cv2
import numpy as np


def _white_stripe_mask(image, threshold=8):
    """Find thin bright horizontal marks supported across a long local run."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # A vertical opening estimates the scene behind a thin horizontal mark.
    residual = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 1), np.uint8))
    candidates = (residual >= threshold).astype(np.uint8) * 255
    length = max(15, min(81, image.shape[1] // 20))
    supported = np.zeros_like(candidates)
    # Short angled kernels follow small scan skew without selecting isolated dots.
    for rise in (-2, -1, 0, 1, 2):
        kernel = np.zeros((5, length), np.uint8)
        cv2.line(kernel, (0, 2 - rise), (length - 1, 2 + rise), 1, 1)
        supported |= cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(supported, np.ones((3, 3), np.uint8))


def restore_photo(image, *, lines=False, period=12.0, line_strength=0.8,
                  tint=False, tint_strength=0.6, contrast=False,
                  contrast_strength=0.6, denoise=False, noise_strength=7.0,
                  scratches=False, scratch_threshold=30, scratch_size=11,
                  line_mode="Automatic white stripes", line_threshold=5,
                  repair_dark_scratches=False):
    """Return RGB pixels and the candidate scratch mask.

    Automatic line repair detects bright horizontal marks and inpaints them.
    The optional periodic mode attenuates a band around a chosen row frequency and
    its harmonics in the row-mean FFT (the horizontal-frequency-zero slice
    of the 2-D FFT). DC is preserved. This targets full-width additive bands,
    not arbitrary scratches. Inpainting candidates are heuristic, not truth.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or not arr.size:
        raise ValueError("Expected a nonempty RGB image.")
    if arr.shape[0] * arr.shape[1] > 4_000_000:
        raise ValueError("Please use a photo with at most 4 million pixels.")
    if not np.isfinite(arr).all():
        raise ValueError("Image contains non-finite pixels.")
    for value in (line_strength, tint_strength, contrast_strength):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Restoration strengths must be between 0 and 1.")
    if not np.isfinite(period) or period < 2:
        raise ValueError("Scan-line spacing must be at least 2 pixels.")
    if not np.isfinite(noise_strength) or not 0 <= noise_strength <= 20:
        raise ValueError("Noise strength must be between 0 and 20.")
    if not 1 <= scratch_threshold <= 255 or scratch_size not in (3, 5, 7, 11, 15, 21):
        raise ValueError("Invalid scratch sensitivity or width.")
    if line_mode not in ("Automatic white stripes", "Periodic banding (FFT)"):
        raise ValueError("Invalid line removal mode.")
    if not np.isfinite(line_threshold) or not 1 <= line_threshold <= 100:
        raise ValueError("Invalid stripe threshold.")
    work = np.rint(np.clip(arr, 0, 255)).astype(np.float64)
    mask = np.zeros(work.shape[:2], dtype=np.uint8)
    if lines and line_strength and line_mode == "Automatic white stripes":
        original = work.astype(np.uint8)
        mask = _white_stripe_mask(original, line_threshold)
        if np.mean(mask > 0) > 0.35:
            raise ValueError("Too many stripe candidates. Increase the stripe threshold.")
        if mask.any():
            repaired = cv2.inpaint(original, mask, 3, cv2.INPAINT_TELEA)
            work = (1 - line_strength) * work + line_strength * repaired
    elif lines and line_strength:
        height = work.shape[0]
        profile = work.mean(axis=1)
        spectrum = np.fft.rfft(profile, axis=0)
        frequencies = np.fft.rfftfreq(height)
        attenuation = np.zeros_like(frequencies)
        # Narrow, smooth notches avoid hard frequency-cut ringing.
        for harmonic in range(1, min(8, int(period / 2)) + 1):
            center = harmonic / period
            attenuation = np.maximum(attenuation, np.exp(
                -0.5 * ((frequencies - center) * height / 0.75) ** 2))
        attenuation[0] = 0
        removed = np.fft.irfft(spectrum * attenuation[:, None], n=height, axis=0)
        work -= line_strength * removed[:, None, :]
    work = np.rint(np.clip(work, 0, 255)).astype(np.uint8)
    if scratches:
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((scratch_size, scratch_size), np.uint8)
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        residual = np.maximum(bright, dark) if repair_dark_scratches else bright
        scratch_mask = (residual >= scratch_threshold).astype(np.uint8) * 255
        # Include softened edges so inpainting does not sample the white fringe.
        scratch_mask = cv2.dilate(scratch_mask, np.ones((3, 3), np.uint8))
        # Never erase most of an image when texture is mistaken for damage.
        if np.mean(scratch_mask > 0) > 0.1:
            raise ValueError("Too many scratch candidates. Increase the scratch threshold or disable scratch repair.")
        if scratch_mask.any():
            work = cv2.inpaint(work, scratch_mask, 3, cv2.INPAINT_TELEA)
        mask |= scratch_mask
    if denoise and noise_strength:
        bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, noise_strength,
                                            noise_strength, 7, 21)
        work = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if tint and tint_strength:
        # Gray-world balance is deliberately bounded; users control the blend.
        means = work.mean(axis=(0, 1))
        gains = np.clip(means.mean() / np.maximum(means, 1), 0.5, 2)
        work = np.rint(np.clip(work * (1 + tint_strength * (gains - 1)),
                               0, 255)).astype(np.uint8)
    if contrast and contrast_strength:
        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
        lightness = lab[:, :, 0]
        low, high = np.percentile(lightness, [1, 99])
        if high > low:
            stretched = np.clip((lightness.astype(float) - low) * 255 / (high - low), 0, 255)
            lab[:, :, 0] = np.rint((1 - contrast_strength) * lightness
                                   + contrast_strength * stretched).astype(np.uint8)
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return work, mask
