"""Lossless integer Haar lifting codec (custom .iwv format, version 1).

RGB -> reversible color differences -> multilevel 2D Haar lifting ->
zigzag signed integers -> byte planes -> the project's DEFLATE codec.
No quantization or floating-point transform is used.
"""
import struct

import numpy as np

from imgutil.deflate import zlib_encode, zlib_decode
from imgutil.lossless import _validate_rgb, crc32

MAGIC = b'IWV1'
HEADER = struct.Struct('>4sIIBI')  # magic, width, height, levels, pixel CRC
MAX_BYTES = 64_000_000


def _lift(values, axis, inverse=False):
    """Pair a,b as detail=b-a, average=a+floor(detail/2); carry odd tails."""
    source = np.moveaxis(values, axis, 0)
    result = np.empty_like(source)
    pairs = source.shape[0] // 2
    low_count = (source.shape[0] + 1) // 2
    if inverse:
        detail = source[low_count:]
        first = source[:pairs] - detail // 2
        result[:2 * pairs:2] = first
        result[1:2 * pairs:2] = first + detail
        if source.shape[0] % 2:
            result[-1] = source[pairs]
    else:
        first, second = source[:2 * pairs:2], source[1:2 * pairs:2]
        detail = second - first
        result[:pairs] = first + detail // 2
        result[low_count:] = detail
        if source.shape[0] % 2:
            result[pairs] = source[-1]
    return np.moveaxis(result, 0, axis)


def haar_transform(values, levels=4, inverse=False):
    """Transform rows then columns of successive low-frequency regions.

    Inverse visits regions in reverse order, undoing columns before rows.
    Works on odd dimensions and one-pixel axes without padding.
    """
    result = np.asarray(values, dtype=np.int64).copy()
    if result.ndim != 3 or not result.size:
        raise ValueError('Expected a nonempty three-dimensional array.')
    if not isinstance(levels, (int, np.integer)) or not 1 <= levels <= 8:
        raise ValueError('Wavelet levels must be an integer from 1 to 8.')
    h, w = result.shape[:2]
    regions = []
    for _ in range(levels):
        if h == 1 and w == 1:
            break
        regions.append((h, w))
        h, w = (h + 1) // 2, (w + 1) // 2
    for h, w in reversed(regions) if inverse else regions:
        region = result[:h, :w]
        axes = (0, 1) if inverse else (1, 0)
        for axis in axes:
            region = _lift(region, axis, inverse)
        result[:h, :w] = region
    return result


def compress_wavelet(rgb, level=9, levels=4):
    """Encode exact uint8 RGB pixels to a standalone IWV archive."""
    rgb = _validate_rgb(rgb)
    pixels = rgb.astype(np.int64)
    # Reversible color transform: green, red-green, blue-green.
    pixels[:, :, 0] -= pixels[:, :, 1]
    pixels[:, :, 2] -= pixels[:, :, 1]
    coefficients = haar_transform(pixels, levels)
    zigzag = np.where(coefficients >= 0, coefficients * 2,
                      -coefficients * 2 - 1).astype('<u4')
    # Group equal-significance bytes to expose runs of zeros to DEFLATE.
    raw = zigzag.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    h, w = rgb.shape[:2]
    return HEADER.pack(MAGIC, w, h, levels, crc32(rgb.tobytes())) + zlib_encode(raw, level)


def decompress_wavelet(data):
    """Validate, decode and verify an IWV archive without the source image."""
    if len(data) < HEADER.size + 6 or len(data) > MAX_BYTES:
        raise ValueError('Invalid wavelet archive size (maximum 64 MB).')
    magic, w, h, levels, checksum = HEADER.unpack_from(data)
    if magic != MAGIC or not w or not h or w * h > 4_000_000 or not 1 <= levels <= 8:
        raise ValueError('Unsupported wavelet archive or invalid dimensions/levels.')
    expected = h * w * 3 * 4
    raw = zlib_decode(data[HEADER.size:], expected)
    if len(raw) != expected:
        raise ValueError('Incorrect wavelet coefficient count.')
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(4, -1).T.copy()
    zigzag = packed.reshape(-1).view('<u4').reshape(h, w, 3).astype(np.int64)
    coefficients = (zigzag >> 1) ^ -(zigzag & 1)
    pixels = haar_transform(coefficients, levels, inverse=True)
    pixels[:, :, 0] += pixels[:, :, 1]
    pixels[:, :, 2] += pixels[:, :, 1]
    if np.any((pixels < 0) | (pixels > 255)):
        raise ValueError('Decoded pixels are outside the RGB range.')
    rgb = pixels.astype(np.uint8)
    if crc32(rgb.tobytes()) != checksum:
        raise ValueError('Wavelet pixel checksum mismatch.')
    return rgb
