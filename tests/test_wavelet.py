import unittest

import numpy as np

from imgutil.wavelet import HEADER, compress_wavelet, decompress_wavelet, haar_transform


class WaveletTests(unittest.TestCase):
    def test_known_haar_coefficients(self):
        source = np.array([10, 14, 20, 26, 31]).reshape(1, 5, 1)
        np.testing.assert_array_equal(haar_transform(source, 1).ravel(), [12, 23, 31, 4, 6])

    def test_exact_pixels(self):
        rng = np.random.default_rng(42)
        for shape in ((1, 1, 3), (1, 19, 3), (17, 1, 3), (15, 17, 3), (32, 32, 3)):
            noise = rng.integers(0, 256, shape, dtype=np.uint8)
            checker = np.repeat(((np.indices(shape[:2]).sum(axis=0) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)
            for pixels in (noise, checker, np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8)):
                for levels in (1, 4, 8):
                    for effort in (0, 1, 9):
                        with self.subTest(shape=shape, levels=levels, effort=effort):
                            encoded = compress_wavelet(pixels, level=effort, levels=levels)
                            np.testing.assert_array_equal(decompress_wavelet(encoded), pixels)

    def test_signed_transform_inverse(self):
        values = np.random.default_rng(7).integers(-255, 256, (31, 29, 3))
        transformed = haar_transform(values, 8)
        np.testing.assert_array_equal(haar_transform(transformed, 8, inverse=True), values)

    def test_damage_rejected(self):
        encoded = compress_wavelet(np.zeros((7, 9, 3), np.uint8))
        bad_crc = bytearray(encoded)
        bad_crc[HEADER.size - 1] ^= 1
        for damaged in (encoded[:-1], encoded + b'junk', bytes(bad_crc), b'bad', b'BAD!' + encoded[4:]):
            with self.subTest(damaged=damaged[:20]), self.assertRaises(ValueError):
                decompress_wavelet(damaged)

    def test_invalid_input(self):
        for pixels in (np.zeros((2, 2, 3)), np.zeros((0, 2, 3), np.uint8)):
            with self.assertRaises(ValueError):
                compress_wavelet(pixels)
        for levels in (0, 9, 1.5):
            with self.assertRaises(ValueError):
                compress_wavelet(np.zeros((2, 2, 3), np.uint8), levels=levels)


if __name__ == '__main__':
    unittest.main()
