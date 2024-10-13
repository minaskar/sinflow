import unittest
import numpy as np
from sinflow.tools import make_strictly_monotonic, sliced_wasserstein_distance, gradient_sliced_wasserstein_distance, gradient_ascent_unit_vector

class TestTools(unittest.TestCase):

    def test_make_strictly_monotonic_non_decreasing(self):
        x = np.array([1.0, 2.0, 2.0, 3.0])
        y = make_strictly_monotonic(x)
        self.assertTrue(np.all(np.diff(y) > 0))

    def test_make_strictly_monotonic_non_increasing(self):
        x = np.array([3.0, 2.0, 2.0, 1.0])
        y = make_strictly_monotonic(x)
        self.assertTrue(np.all(np.diff(y) < 0))

    def test_make_strictly_monotonic_empty(self):
        x = np.array([])
        y = make_strictly_monotonic(x)
        self.assertTrue(np.array_equal(y, x))

    def test_make_strictly_monotonic_not_monotonic(self):
        x = np.array([1.0, 3.0, 2.0])
        with self.assertRaises(ValueError):
            make_strictly_monotonic(x)

    def test_make_strictly_monotonic_not_numpy_array(self):
        x = [1.0, 2.0, 3.0]
        with self.assertRaises(TypeError):
            make_strictly_monotonic(x)

    def test_sliced_wasserstein_distance(self):
        x = np.random.randn(100, 2)
        direction = np.array([1, 0])
        distance = sliced_wasserstein_distance(x, direction)
        self.assertTrue(distance >= 0)

    def test_gradient_sliced_wasserstein_distance(self):
        x = np.random.randn(100, 2)
        direction = np.array([1, 0])
        gradient = gradient_sliced_wasserstein_distance(x, direction)
        self.assertEqual(gradient.shape, (2,))

    def test_gradient_ascent_unit_vector(self):
        x = np.random.randn(100, 2)
        direction, loss = gradient_ascent_unit_vector(x, max_iter=10, verbose=True)
        self.assertEqual(direction.shape, (2,))
        self.assertTrue(np.isclose(np.linalg.norm(direction), 1))

if __name__ == '__main__':
    unittest.main()
