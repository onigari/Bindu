"""Sparse RGB Fourier archives with conjugate-pair energy pruning."""

import io
import zipfile

import numpy as np


MAX_PIXELS = 4_000_000
MAX_ARCHIVE_BYTES = 160_000_000


def _partners(height, width):
    indices = np.arange(height * width)
    return (-(indices // width) % height) * width + (-indices % width)


def compress_image(img: np.ndarray, energy_percent: float = 99.0):
    """Return (NPZ bytes, channel statistics), keeping target energy per channel.

    Conjugate pairs are ranked by their combined squared magnitude. Only one
    coefficient per pair is stored, as complex64, with an unshifted FFT index.
    DC is always retained. Input is rounded to 8-bit RGB before transformation.
    """
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or 0 in rgb.shape:
        raise ValueError("Expected a nonempty RGB image of shape (H, W, 3).")
    h, w, _ = rgb.shape
    if h * w > MAX_PIXELS:
        raise ValueError("Please use an image with at most 4 million pixels.")
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("RGB values must be finite and between 0 and 255.")
    if not np.isfinite(energy_percent) or not 0 < energy_percent <= 100:
        raise ValueError("Energy retention must be greater than 0 and at most 100.")
    rgb = np.rint(rgb)
    partners = _partners(h, w)
    representatives = np.flatnonzero(np.arange(h * w) <= partners)
    weights = np.where(representatives == partners[representatives], 1, 2)
    payload = {"version": np.array([1], dtype=np.uint8),
               "shape": np.array([h, w], dtype=np.int32)}
    stats = []
    for c, name in enumerate("RGB"):
        spectrum = np.fft.fft2(rgb[:, :, c], norm="ortho").ravel()
        energies = np.abs(spectrum[representatives]) ** 2 * weights
        total = float(energies.sum())
        # DC is representative 0; rank the remaining pairs by energy.
        order = np.argsort(-energies[1:], kind="stable") + 1
        if energy_percent == 100:
            chosen = np.arange(len(representatives))
        else:
            target = max(0.0, total * energy_percent / 100 - energies[0])
            count = 0 if target == 0 else min(len(order), int(np.searchsorted(np.cumsum(energies[order]), target)) + 1)
            chosen = np.concatenate(([0], order[:count]))
        indices = np.sort(representatives[chosen])
        payload[f"{name}_indices"] = indices.astype(np.int32)
        payload[f"{name}_values"] = spectrum[indices].astype(np.complex64)
        stats.append({"Channel": name, "Retained energy (%)": 100 * float(energies[chosen].sum()) / total if total else 100.0,
                      "Kept FFT coefficients": int(weights[chosen].sum()), "Total FFT coefficients": h * w})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue(), stats


def decompress_image(data: bytes) -> np.ndarray:
    """Validate an archive and reconstruct uint8 RGB without the original image."""
    expected = {"version", "shape"} | {f"{name}_{field}" for name in "RGB" for field in ("indices", "values")}
    try:
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive is too large.")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != {f"{key}.npy" for key in expected}:
                raise ValueError("Not a supported Fourier archive.")
            if sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError("Expanded archive is too large.")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            version, shape = archive["version"], archive["shape"]
            if version.shape != (1,) or version.dtype.kind not in "iu" or version[0] != 1:
                raise ValueError("Unsupported archive version.")
            if shape.shape != (2,) or shape.dtype.kind not in "iu":
                raise ValueError("Invalid image dimensions.")
            h, w = map(int, shape)
            if h <= 0 or w <= 0 or h * w > MAX_PIXELS:
                raise ValueError("Invalid image dimensions or image exceeds 4 million pixels.")
            partners = _partners(h, w)
            result = np.empty((h, w, 3), dtype=np.uint8)
            for c, name in enumerate("RGB"):
                indices, values = archive[f"{name}_indices"], archive[f"{name}_values"]
                if indices.ndim != 1 or indices.dtype.kind not in "iu" or values.shape != indices.shape or values.dtype.kind != "c":
                    raise ValueError("Invalid coefficient arrays.")
                if not len(indices) or indices[0] != 0 or np.any(indices >= h * w) or np.any(indices < 0):
                    raise ValueError("Invalid coefficient indices.")
                if np.any(indices[1:] <= indices[:-1]) or np.any(indices > partners[indices]) or not np.isfinite(values).all():
                    raise ValueError("Invalid or duplicate coefficients.")
                # Normalized FFT coefficients for 8-bit data cannot exceed this bound.
                if np.any(np.abs(values) > 256 * np.sqrt(h * w)):
                    raise ValueError("Coefficient values are out of range.")
                self_paired = indices == partners[indices]
                if np.any(np.abs(values[self_paired].imag) > 1e-4):
                    raise ValueError("Invalid real-frequency coefficients.")
                spectrum = np.zeros(h * w, dtype=np.complex128)
                spectrum[indices] = values
                spectrum[partners[indices]] = values.conj()
                spatial = np.fft.ifft2(spectrum.reshape(h, w), norm="ortho").real
                result[:, :, c] = np.rint(np.clip(spatial, 0, 255)).astype(np.uint8)
            return result
    except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"Cannot decompress this file: {exc}") from exc
