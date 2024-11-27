import unittest
import numpy as np
from sinflow.tools import (make_strictly_monotonic, 
                           #sliced_wasserstein_distance,
                           max_sliced_wasserstein_distance,) 
                           #gradient_max_sliced_wasserstein_distance, 
                           #optimize_direction)

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

    #def test_sliced_wasserstein_distance(self):
    #    x = np.random.randn(100, 2)
    #    distance = sliced_wasserstein_distance(x)
    #    self.assertTrue(distance >= 0)

    def test_max_sliced_wasserstein_distance(self):
        x = np.random.randn(100, 2)
        direction = np.array([1, 0])
        distance = max_sliced_wasserstein_distance(x, direction)
        self.assertTrue(distance >= 0)

    #def test_gradient_max_sliced_wasserstein_distance(self):
    #    x = np.random.randn(100, 2)
    #    direction = np.array([1, 0])
    #    gradient = gradient_max_sliced_wasserstein_distance(x, direction)
    #    self.assertEqual(gradient.shape, (2,))

    #def test_optimize_direction(self):
    #    x = np.random.randn(100, 2)
    #    x[:, 0] *= 10
    #    direction, loss = optimize_direction(x, max_iter=100, verbose=False)
    #    self.assertEqual(direction.shape, (2,))
    #    self.assertTrue(np.isclose(np.linalg.norm(direction), 1))

if __name__ == '__main__':
    unittest.main()
