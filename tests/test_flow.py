import unittest
from sinflow.flow import Flow
import numpy as np
    
class TestFlow(unittest.TestCase):

    def setUp(self):
        self.flow = Flow()

    def test_initialization(self):
        self.assertEqual(self.flow.n_transforms, 100)
        self.assertEqual(self.flow.n_knots, 1000)
        self.assertEqual(self.flow.validation_fraction, 0.2)
        self.assertTrue(self.flow.early_stopping)
        self.assertEqual(self.flow.n_iter_no_change, 20)
        self.assertEqual(self.flow.reg_cov, 1e-6)
        self.assertTrue(self.flow.whiten)
        self.assertFalse(self.flow.warm_start)
        self.assertEqual(self.flow.p, 2)
        self.assertEqual(self.flow.max_iter, 1000)
        self.assertEqual(self.flow.alpha_init, 1.0)
        self.assertEqual(self.flow.beta, 0.2)
        self.assertEqual(self.flow.tol, 1e-6)
        self.assertEqual(self.flow.c, 1e-4)
        self.assertFalse(self.flow.verbose)
        self.assertFalse(self.flow.initialized)
        self.assertIsNone(self.flow.transforms)
        self.assertIsNone(self.flow.train_history)
        self.assertIsNone(self.flow.val_history)

    def test_fit(self):
        x = np.random.randn(100, 10)
        self.flow.fit(x)
        self.assertTrue(self.flow.initialized)
        self.assertIsNotNone(self.flow.transforms)
        self.assertIsNotNone(self.flow.train_history)
        self.assertIsNotNone(self.flow.val_history)

    def test_forward(self):
        x = np.random.randn(100, 10)
        self.flow.fit(x)
        y, log_det = self.flow.forward(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(log_det.shape, (100,))

    def test_inverse(self):
        x = np.random.randn(100, 10)
        self.flow.fit(x)
        y, log_det = self.flow.forward(x)
        x_reconstructed, log_det_reconstructed = self.flow.inverse(y)
        np.testing.assert_almost_equal(x, x_reconstructed, decimal=2)
        np.testing.assert_almost_equal(log_det, -log_det_reconstructed, decimal=2)

    def test_log_prob(self):
        x = np.random.randn(100, 10)
        self.flow.fit(x)
        log_prob = self.flow.log_prob(x)
        self.assertEqual(log_prob.shape, (100,))

    def test_sample(self):
        self.flow.fit(np.random.randn(100, 10))
        samples = self.flow.sample(50)
        self.assertEqual(samples.shape, (50, 10))

if __name__ == '__main__':
    unittest.main()
