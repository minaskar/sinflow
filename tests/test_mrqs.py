import unittest
import numpy as np
from sinflow.mrqs import MonotonicRationalQuadraticSpline

class TestMonotonicRationalQuadraticSpline(unittest.TestCase):
    def setUp(self):
        self.x_knots = np.array([0, 1, 2, 3, 4])
        self.y_knots = np.array([0, 1, 4, 9, 16])
        self.spline = MonotonicRationalQuadraticSpline(self.x_knots, self.y_knots)

    def test_forward_within_bounds(self):
        x = np.array([0.5, 1.5, 2.5])
        y, dy_dx = self.spline.forward(x)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(dy_dx > 0))

    def test_forward_extrapolate_below(self):
        x = np.array([-1, -0.5])
        y, dy_dx = self.spline.forward(x)
        self.assertTrue(np.all(y <= 0))
        self.assertTrue(np.all(dy_dx > 0))

    def test_forward_extrapolate_above(self):
        x = np.array([5, 6])
        y, dy_dx = self.spline.forward(x)
        self.assertTrue(np.all(y >= 16))
        self.assertTrue(np.all(dy_dx > 0))

    def test_inverse_within_bounds(self):
        y = np.array([0.5, 2, 5])
        x, dx_dy = self.spline.inverse(y)
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(dx_dy > 0))

    def test_inverse_extrapolate_below(self):
        y = np.array([-1, -0.5])
        x, dx_dy = self.spline.inverse(y)
        self.assertTrue(np.all(x <= 0))
        self.assertTrue(np.all(dx_dy >= 0))

    def test_inverse_extrapolate_above(self):
        y = np.array([17, 18])
        x, dx_dy = self.spline.inverse(y)
        self.assertTrue(np.all(x >= 4))
        self.assertTrue(np.all(dx_dy >= 0))

    def test_derivative_within_bounds(self):
        x = np.array([0.5, 1.5, 2.5])
        dy_dx = self.spline.derivative(x)
        self.assertTrue(np.all(dy_dx > 0))

    def test_derivative_extrapolate_below(self):
        x = np.array([-1, -0.5])
        dy_dx = self.spline.derivative(x)
        self.assertTrue(np.all(dy_dx >= 0))

    def test_derivative_extrapolate_above(self):
        x = np.array([5, 6])
        dy_dx = self.spline.derivative(x)
        self.assertTrue(np.all(dy_dx >= 0))

    def test_derivative_of_forward(self):
        x = np.array([0.5, 1.5, 2.5])
        _, dy_dx = self.spline.forward(x)
        dy_dx_reconstructed = self.spline.derivative(x)
        np.testing.assert_almost_equal(dy_dx, dy_dx_reconstructed, decimal=5)

    def test_forward_and_inverse_derivative(self):
        x = np.array([0.5, 1.5, 2.5])
        y, dy_dx = self.spline.forward(x)
        x_reconstructed, dx_dy = self.spline.inverse(y)
        np.testing.assert_almost_equal(x, x_reconstructed, decimal=5)
        np.testing.assert_almost_equal(dy_dx, 1 / dx_dy, decimal=5)


if __name__ == '__main__':
    unittest.main()
