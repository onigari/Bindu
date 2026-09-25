import unittest
import numpy as np
from imgutil.noise import add_noise, remove_noise


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 120, dtype=np.uint8)

    def test_reproducible_without_modifying_source(self):
        for kind, amount in [("Gaussian", 20), ("Salt and pepper", 5)]:
            first = add_noise(self.image, kind, amount, 42)
            np.testing.assert_array_equal(first, add_noise(self.image, kind, amount, 42))
            self.assertFalse(np.array_equal(first, add_noise(self.image, kind, amount, 43)))
            self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.all(self.image == 120))

    def test_zero_noise_preserves_pixels(self):
        for kind in ("Gaussian", "Salt and pepper"):
            np.testing.assert_array_equal(add_noise(self.image, kind, 0), self.image)

    def test_impulses_change_whole_pixels(self):
        noisy = add_noise(self.image, "Salt and pepper", 10)
        self.assertTrue(np.isin(noisy, [0, 120, 255]).all())
        np.testing.assert_array_equal(noisy[:, :, 0], noisy[:, :, 2])
        cleaned = remove_noise(noisy, "Median")
        self.assertLess(np.mean((cleaned.astype(float)-120)**2), np.mean((noisy.astype(float)-120)**2))

    def test_gaussian_removal_on_flat_image(self):
        noisy = add_noise(self.image)
        for method in ("Gaussian blur", "Non-local means"):
            cleaned = remove_noise(noisy, method)
            self.assertEqual(cleaned.shape, self.image.shape)
            self.assertLess(np.mean((cleaned.astype(float)-120)**2), np.mean((noisy.astype(float)-120)**2))

    def test_invalid_inputs(self):
        for image in (np.zeros((2, 2)), np.full((2, 2, 3), np.nan)):
            with self.assertRaises(ValueError):
                add_noise(image)
        with self.assertRaises(ValueError):
            add_noise(self.image, "Salt and pepper", 101)
        with self.assertRaises(ValueError):
            remove_noise(self.image, size=4)


if __name__ == "__main__":
    unittest.main()
