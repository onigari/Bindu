"""Codec tests; stdlib zlib is an independent test oracle, never a runtime codec."""
import struct
import unittest
import zlib

import numpy as np

from imgutil.deflate import adler32, zlib_encode, zlib_decode
from imgutil.lossless import (SIGNATURE, compress_png, decompress_png, encode_bmp,
                             crc32, _chunk, _filter_rows, _unfilter)


class LosslessTests(unittest.TestCase):
    def test_every_level_preserves_pixels(self):
        rng = np.random.default_rng(7)
        for shape in [(1, 1, 3), (1, 11, 3), (7, 1, 3), (7, 11, 3), (32, 32, 3)]:
            original = rng.integers(0, 256, shape, dtype=np.uint8)
            before = original.copy()
            for level in range(10):
                with self.subTest(shape=shape, level=level):
                    data = compress_png(original, level)
                    np.testing.assert_array_equal(decompress_png(data), original)
                    # Independently decompress the actual IDAT payload.
                    size = int.from_bytes(data[33:37], 'big')
                    self.assertEqual(zlib.decompress(data[41:41 + size]), _filter_rows(original))
            np.testing.assert_array_equal(original, before)

    def test_smooth_compression_and_bmp_layout(self):
        original = np.full((128, 127, 3), [0, 128, 255], dtype=np.uint8)
        self.assertLess(len(compress_png(original)), original.nbytes)
        original = np.random.default_rng(9).integers(0, 256, (128, 127, 3), dtype=np.uint8)
        bmp = encode_bmp(original)
        self.assertEqual(bmp[:2], b'BM')
        self.assertEqual(int.from_bytes(bmp[2:6], 'little'), len(bmp))
        self.assertEqual(struct.unpack('<ii', bmp[18:26]), (127, 128))
        self.assertEqual(int.from_bytes(bmp[28:30], 'little'), 24)
        stride = (127 * 3 + 3) // 4 * 4
        rows = np.frombuffer(bmp[54:], dtype=np.uint8).reshape(128, stride)
        restored = rows[:, :127 * 3].reshape(128, 127, 3)[::-1, :, ::-1]
        np.testing.assert_array_equal(restored, original)

    def test_all_filters_with_external_deflate(self):
        original = np.random.default_rng(8).integers(0, 256, (9, 13, 3), dtype=np.uint8)
        for kind in range(5):
            filtered = bytearray()
            for y, row in enumerate(original.reshape(9, -1)):
                filtered.append(kind)
                for x, value in enumerate(row):
                    a = int(row[x - 3]) if x >= 3 else 0
                    b = int(original.reshape(9, -1)[y - 1, x]) if y else 0
                    c = int(original.reshape(9, -1)[y - 1, x - 3]) if y and x >= 3 else 0
                    p = a + b - c
                    candidates = [a, b, c]
                    paeth = candidates[min(range(3), key=lambda i: abs(p - candidates[i]))]
                    predictor = [0, a, b, (a + b) // 2, paeth][kind]
                    filtered.append((int(value) - predictor) % 256)
            header = struct.pack('>IIBBBBB', 13, 9, 8, 2, 0, 0, 0)
            compressed = zlib.compress(bytes(filtered))
            # Split IDAT demonstrates that DEFLATE is continuous across chunks.
            png = SIGNATURE + _chunk(b'IHDR', header) + _chunk(b'IDAT', compressed[:5]) + _chunk(b'IDAT', compressed[5:]) + _chunk(b'IEND', b'')
            np.testing.assert_array_equal(decompress_png(png), original)

    def test_invalid_encoding_inputs(self):
        for invalid in [np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                compress_png(invalid)
        for level in [-1, 10, 2.5]:
            with self.assertRaises(ValueError):
                compress_png(np.zeros((1, 1, 3), dtype=np.uint8), level)

    def test_rejects_corrupt_pngs_and_unsupported_modes(self):
        good = compress_png(np.zeros((2, 2, 3), dtype=np.uint8))
        damaged = bytearray(good)
        damaged[45] ^= 1
        for data in [b'not a png', good[:30], bytes(damaged), good[:-12], good + b'trailing']:
            with self.assertRaises(ValueError):
                decompress_png(data)
        for depth, color, interlace in [(8, 6, 0), (8, 0, 0), (16, 2, 0), (8, 2, 1)]:
            header = struct.pack('>IIBBBBB', 2, 2, depth, color, 0, 0, interlace)
            with self.assertRaises(ValueError):
                decompress_png(SIGNATURE + _chunk(b'IHDR', header) + good[33:])
        for chunk in [b'acTL', b'tRNS']:
            with self.assertRaises(ValueError):
                decompress_png(good[:33] + _chunk(chunk, b'') + good[33:])
        with self.assertRaises(ValueError):
            _unfilter(bytes([5, 0, 0, 0]), 1, 1)


class DeflateTests(unittest.TestCase):
    def test_checksum_known_vectors(self):
        self.assertEqual(crc32(b'123456789'), 0xCBF43926)
        self.assertEqual(adler32(b'Wikipedia'), 0x11E60398)

    def test_encoder_against_standard_decoder(self):
        rng = np.random.default_rng(15)
        seed = rng.integers(0, 256, 32768, dtype=np.uint8).tobytes()
        for data in [b'', b'a', b'a' * 100000, b'abc' * 2000, bytes(range(256)) * 8, seed + seed, seed * 3]:
            for level in (0, 1, 9):
                with self.subTest(size=len(data), level=level):
                    encoded = zlib_encode(data, level)
                    self.assertEqual(zlib.decompress(encoded), data)
                    self.assertEqual(zlib_decode(encoded, len(data)), data)

    def test_decoder_stored_fixed_dynamic_and_multiple_blocks(self):
        data = bytes(range(64)) * 2000
        for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FIXED, zlib.Z_HUFFMAN_ONLY):
            encoder = zlib.compressobj(6, zlib.DEFLATED, 15, 8, strategy)
            encoded = encoder.compress(data[:60000]) + encoder.flush(zlib.Z_SYNC_FLUSH)
            encoded += encoder.compress(data[60000:]) + encoder.flush()
            self.assertEqual(zlib_decode(encoded, len(data)), data)
        dynamic = zlib.compress(data)
        self.assertEqual((dynamic[2] >> 1) & 3, 2)
        self.assertEqual(zlib_decode(dynamic, len(data)), data)

    def test_rejects_corruption_and_output_overflow(self):
        encoded = zlib_encode(b'abc' * 200)
        for bad in [b'', encoded[:-1], encoded[:-4] + bytes(4), b'\x78\x01\x07' + bytes(4)]:
            with self.assertRaises(ValueError):
                zlib_decode(bad, 600)
        with self.assertRaises(ValueError):
            zlib_decode(encoded, 599)


if __name__ == '__main__':
    unittest.main()
