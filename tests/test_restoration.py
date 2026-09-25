import unittest

import cv2
import numpy as np

from imgutil.restoration import restore_photo


class RestorationTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:180, :240]
        gray = (70 + x * 0.2 + y * 0.1).astype(np.uint8)
        self.clean = np.stack([gray, gray, gray], axis=-1)

    def test_irregular_and_skewed_white_stripes(self):
        damaged = self.clean.copy()
        for row in (21, 49, 84, 130):
            cv2.line(damaged, (0, row), (239, row + 3), (230, 230, 230), 1)
        result, mask = restore_photo(damaged, lines=True, line_strength=1)
        defects = np.any(damaged != self.clean, axis=2)
        before = np.abs(damaged.astype(float) - self.clean)[defects].mean()
        after = np.abs(result.astype(float) - self.clean)[defects].mean()
        self.assertLess(after, before * 0.15)
        self.assertGreater(np.mean(mask[defects] > 0), 0.95)
        np.testing.assert_array_equal(result[mask == 0], damaged[mask == 0])

    def test_dots_larger_than_old_window(self):
        damaged = self.clean.copy()
        for point in ((40, 40), (120, 90), (190, 140)):
            cv2.circle(damaged, point, 3, (250, 250, 250), -1)
        result, mask = restore_photo(damaged, scratches=True)
        defects = np.any(damaged != self.clean, axis=2)
        self.assertTrue(np.all(mask[defects] > 0))
        self.assertLess(np.abs(result.astype(float) - self.clean)[defects].mean(), 5)

    def test_disabled_and_zero_strength_preserve_pixels(self):
        for options in ({}, {"lines": True, "line_strength": 0}):
            result, mask = restore_photo(self.clean, **options)
            np.testing.assert_array_equal(result, self.clean)
            self.assertFalse(mask.any())

    def test_clean_gradient_and_broad_edge_are_not_scratches(self):
        clean = self.clean.copy()
        clean[90:] += 50
        result, mask = restore_photo(clean, lines=True, scratches=True)
        np.testing.assert_array_equal(result, clean)
        self.assertFalse(mask.any())

    def test_periodic_mode_remains_available(self):
        rows = np.arange(180)
        clean = np.full((180, 240, 3), 120.0)
        damaged = clean + (20 * np.cos(2 * np.pi * rows / 12))[:, None, None]
        result, _ = restore_photo(damaged, lines=True, period=12,
                                  line_strength=1, line_mode="Periodic banding (FFT)")
        self.assertLess(np.abs(result.astype(float) - clean).mean(), 1)


if __name__ == "__main__":
    unittest.main()
