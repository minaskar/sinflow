import unittest
import numpy as np
from sinflow.kde import KernelDensityEstimator

class TestKernelDensityEstimator(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1, 2, 3, 4, 5])
        self.kde = KernelDensityEstimator(self.data)

    def test_bandwidth_default(self):
        expected_bandwidth = 1.06 * np.std(self.data, ddof=1) * len(self.data) ** (-1 / 5)
        self.assertAlmostEqual(self.kde.bandwidth, expected_bandwidth)

    def test_bandwidth_custom(self):
        kde = KernelDensityEstimator(self.data, bandwidth=0.5)
        self.assertEqual(kde.bandwidth, 0.5)

    def test_pdf_single_point(self):
        result = self.kde.pdf(3)
        self.assertIsInstance(result, float)

    def test_pdf_multiple_points(self):
        result = self.kde.pdf([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_cdf_single_point(self):
        result = self.kde.cdf(3)
        self.assertIsInstance(result, float)

    def test_cdf_multiple_points(self):
        result = self.kde.cdf([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_derivative_pdf_single_point(self):
        result = self.kde.derivative_pdf(3)
        self.assertIsInstance(result, float)

    def test_derivative_pdf_multiple_points(self):
        result = self.kde.derivative_pdf([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            KernelDensityEstimator([])

    def test_negative_bandwidth(self):
        with self.assertRaises(ValueError):
            KernelDensityEstimator(self.data, bandwidth=-1)

if __name__ == '__main__':
    unittest.main()
