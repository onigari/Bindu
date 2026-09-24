"""Custom non-interlaced RGB PNG codec: filters, chunks, CRC-32 and DEFLATE.

No Pillow or compression-library calls. Supports single-frame 8-bit RGB.
"""

import struct
import numpy as np
from imgutil.deflate import zlib_encode, zlib_decode

MAX_PIXELS = 4_000_000
SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _crc_table():
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0xEDB88320 if value & 1 else 0)
        table.append(value)
    return table


CRC_TABLE = _crc_table()


def crc32(data):
    value = 0xFFFFFFFF
    for byte in data:
        value = CRC_TABLE[(value ^ byte) & 255] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', crc32(kind + data))


def _validate_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or 0 in rgb.shape:
        raise ValueError('Expected a nonempty uint8 RGB image with shape (H, W, 3).')
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise ValueError('Please use an image with at most 4 million pixels.')
    return rgb


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _filter_rows(rgb):
    previous = np.zeros(rgb.shape[1] * 3, dtype=np.int16)
    output = bytearray()
    for row in rgb:
        row = row.ravel().astype(np.int16)
        left, upper_left = np.zeros_like(row), np.zeros_like(row)
        left[3:], upper_left[3:] = row[:-3], previous[:-3]
        predictors = [0, left, previous, (left + previous) // 2, _paeth(left, previous, upper_left)]
        candidates = [((row - predictor) & 255).astype(np.uint8) for predictor in predictors]
        # Minimize the sum of absolute signed byte residuals.
        scores = [np.abs(candidate.view(np.int8).astype(np.int16)).sum() for candidate in candidates]
        kind = int(np.argmin(scores))
        output.append(kind)
        output.extend(candidates[kind].tobytes())
        previous = row
    return bytes(output)


def compress_png(rgb: np.ndarray, level: int = 9) -> bytes:
    """Encode RGB exactly. Level 0 stores bytes; 1-9 search for LZ77 matches."""
    rgb = _validate_rgb(rgb)
    if not isinstance(level, (int, np.integer)) or not 0 <= level <= 9:
        raise ValueError('PNG compression level must be an integer from 0 to 9.')
    height, width, _ = rgb.shape
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    compressed = zlib_encode(_filter_rows(rgb), int(level))
    return SIGNATURE + _chunk(b'IHDR', header) + _chunk(b'IDAT', compressed) + _chunk(b'IEND', b'')


def _unfilter(raw, height, width):
    stride = width * 3
    result = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.int64)
    for y in range(height):
        start = y * (stride + 1)
        kind = raw[start]
        row = np.frombuffer(raw[start + 1:start + 1 + stride], dtype=np.uint8).astype(np.int64)
        if kind == 1:
            row = np.cumsum(row.reshape(width, 3), axis=0).ravel() & 255
        elif kind == 2:
            row = (row + previous) & 255
        elif kind in (3, 4):
            for x in range(stride):
                a = int(row[x - 3]) if x >= 3 else 0
                b = int(previous[x])
                c = int(previous[x - 3]) if x >= 3 else 0
                if kind == 3:
                    predictor = (a + b) // 2
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
                row[x] = (row[x] + predictor) & 255
        elif kind != 0:
            raise ValueError('Unknown PNG row filter.')
        result[y] = row
        previous = row
    return result.reshape(height, width, 3)


def decompress_png(data: bytes) -> np.ndarray:
    """Validate PNG chunks, inflate, and reverse filters without image libraries."""
    if len(data) > 64_000_000 or not data.startswith(SIGNATURE):
        raise ValueError('Invalid PNG signature or file exceeds 64 MB.')
    position, header, ended, idat_closed = 8, None, False, False
    compressed = bytearray()
    seen_idat = seen_palette = False
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError('Truncated PNG chunk.')
        size = int.from_bytes(data[position:position + 4], 'big')
        kind = data[position + 4:position + 8]
        end = position + 12 + size
        if end > len(data) or size > 0x7FFFFFFF:
            raise ValueError('Invalid PNG chunk length.')
        payload = data[position + 8:end - 4]
        if crc32(kind + payload) != int.from_bytes(data[end - 4:end], 'big'):
            raise ValueError('PNG chunk checksum mismatch.')
        if not all(65 <= b <= 90 or 97 <= b <= 122 for b in kind) or kind[2] & 32:
            raise ValueError('Invalid PNG chunk type.')
        if header is None and kind != b'IHDR':
            raise ValueError('PNG must start with IHDR.')
        if seen_idat and kind != b'IDAT':
            idat_closed = True
        if kind == b'IHDR':
            if header is not None or size != 13:
                raise ValueError('Invalid PNG header.')
            width, height, depth, color, method, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            if not width or not height or width * height > MAX_PIXELS:
                raise ValueError('Image dimensions must be nonzero and at most 4 million pixels.')
            if (depth, color, method, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError('Only non-interlaced 8-bit RGB PNG is supported.')
            header = (height, width)
        elif kind == b'IDAT':
            if idat_closed:
                raise ValueError('PNG IDAT chunks must be consecutive.')
            seen_idat = True
            compressed.extend(payload)
        elif kind == b'IEND':
            if size or not seen_idat or end != len(data):
                raise ValueError('Invalid PNG ending.')
            ended = True
            break
        elif kind in (b'acTL', b'fcTL', b'fdAT', b'tRNS'):
            raise ValueError('Animation and transparency are not supported.')
        elif kind == b'PLTE':
            if seen_palette or seen_idat or not size or size % 3 or size > 768:
                raise ValueError('Invalid optional palette.')
            seen_palette = True
        elif not kind[0] & 32:
            raise ValueError('Unsupported critical PNG chunk.')
        position = end
    if not ended:
        raise ValueError('Missing PNG ending.')
    height, width = header
    expected = height * (width * 3 + 1)
    raw = zlib_decode(bytes(compressed), expected)
    if len(raw) != expected:
        raise ValueError('Incorrect decoded PNG size.')
    return _unfilter(raw, height, width)


def encode_bmp(rgb: np.ndarray) -> bytes:
    """Write bottom-up uncompressed 24-bit BMP pixels with row padding."""
    rgb = _validate_rgb(rgb)
    height, width, _ = rgb.shape
    padding = (-width * 3) % 4
    pixels = b''.join(row[:, ::-1].tobytes() + bytes(padding) for row in rgb[::-1])
    header = struct.pack('<2sIHHI', b'BM', 54 + len(pixels), 0, 0, 54)
    dib = struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return header + dib + pixels
