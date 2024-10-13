import unittest
import numpy as np
from sinflow.transforms import AffineTransform, SplineTransform, ProjectedSplineTransform

class TestAffineTransform(unittest.TestCase):

    def setUp(self):
        self.transform = AffineTransform()
        self.data = np.random.randn(100, 3)
        self.transform.fit(self.data)

    def test_forward(self):
        transformed_data, log_det = self.transform.forward(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)
        self.assertEqual(log_det.shape[0], self.data.shape[0])

    def test_inverse(self):
        transformed_data, log_det_forward = self.transform.forward(self.data)
        inverse_data, log_det_inverse = self.transform.inverse(transformed_data)
        self.assertEqual(inverse_data.shape, self.data.shape)
        np.testing.assert_almost_equal(inverse_data, self.data, decimal=5)
        self.assertEqual(log_det_forward.shape[0], self.data.shape[0])
        self.assertEqual(log_det_inverse.shape[0], self.data.shape[0])
        np.testing.assert_almost_equal(log_det_forward, -log_det_inverse, decimal=5)

class TestSplineTransform(unittest.TestCase):

    def setUp(self):
        self.transform = SplineTransform()
        self.data = np.random.randn(100, 3)
        self.transform.fit(self.data)

    def test_forward(self):
        transformed_data, log_det = self.transform.forward(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)
        self.assertEqual(log_det.shape[0], self.data.shape[0])

    def test_inverse(self):
        transformed_data, log_det_forward = self.transform.forward(self.data)
        inverse_data, log_det_inverse = self.transform.inverse(transformed_data)
        self.assertEqual(inverse_data.shape, self.data.shape)
        np.testing.assert_almost_equal(inverse_data, self.data, decimal=5)
        self.assertEqual(log_det_forward.shape[0], self.data.shape[0])
        self.assertEqual(log_det_inverse.shape[0], self.data.shape[0])
        np.testing.assert_almost_equal(log_det_forward, -log_det_inverse, decimal=5)

class TestProjectedSplineTransform(unittest.TestCase):

    def setUp(self):
        direction = np.random.randn(3)
        self.transform = ProjectedSplineTransform(direction)
        self.data = np.random.randn(100, 3)
        self.transform.fit(self.data)

    def test_forward(self):
        transformed_data, log_det = self.transform.forward(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)
        self.assertEqual(log_det.shape[0], self.data.shape[0])

    def test_inverse(self):
        transformed_data, log_det_forward = self.transform.forward(self.data)
        inverse_data, log_det_inverse = self.transform.inverse(transformed_data)
        self.assertEqual(inverse_data.shape, self.data.shape)
        np.testing.assert_almost_equal(inverse_data, self.data, decimal=5)
        self.assertEqual(log_det_forward.shape[0], self.data.shape[0])
        self.assertEqual(log_det_inverse.shape[0], self.data.shape[0])
        np.testing.assert_almost_equal(log_det_forward, -log_det_inverse, decimal=5)

if __name__ == '__main__':
    unittest.main()
