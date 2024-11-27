import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular

from .kde import KernelDensityEstimator
from .mrqs import MonotonicRationalQuadraticSpline
from .tools import make_strictly_monotonic

class AffineTransform:
    r"""
    
    Affine transformation.

    Parameters:
    -----------
    eps: float, default=1e-9
        Small constant to add to the covariance matrix for numerical stability.
    """

    def __init__(self, eps=1e-9):
        self.eps = eps
        self.dim = None
        self.mean = None
        self.cov = None
        self.chol = None
        self.log_det = None

    def fit(self, x):
        r"""
        
        Fit the affine transformation to the given data.

        Parameters:
        -----------
        x : numpy.ndarray
            Data to fit the transformation to.
        """
        self.dim = x.shape[1]
        self.mean = np.mean(x, axis=0)
        self.cov = np.cov(x, rowvar=False) + self.eps * np.eye(x.shape[1])
        self.chol = cholesky(self.cov, lower=True)
        self.log_det = np.sum(np.log(np.diag(self.chol)))

    def forward(self, x):
        r"""
        
        Apply the forward transformation to the given data.

        Parameters:
        -----------
        x : numpy.ndarray
            Data to transform.
        
        Returns:
        --------
        y : numpy.ndarray
            Transformed data.
        log_det : numpy.ndarray
            Logarithm of the determinant of the Jacobian of the transformation.
        """
        y = x - self.mean
        y = solve_triangular(self.chol, y.T, lower=True).T
        log_det = - self.log_det * np.ones(y.shape[0])

        return y, log_det

    def inverse(self, y):
        r"""

        Apply the inverse transformation to the given data.

        Parameters:
        -----------
        y : numpy.ndarray
            Data to transform.

        Returns:
        --------
        x : numpy.ndarray
            Transformed data.
        log_det : numpy.ndarray
            Logarithm of the determinant of the Jacobian of the transformation.
        """
        x = y @ self.chol.T + self.mean
        #x = np.einsum('ij,jk->ik', y, self.chol.T) + self.mean
        log_det = self.log_det * np.ones(y.shape[0])

        return x, log_det


class SplineTransform:
    r"""

    Monotonic spline transformation.

    Parameters:
    -----------
    n_knots : int, default=1000
        Number of knots to use in the spline.
    bandwidth : float, default=None
        Bandwidth for the KDEs. If not provided, Silverman's rule of thumb is used.
    eps : float, default=1e-6
        Small constant to add to the spline values for numerical stability.
    """
    def __init__(self, n_knots=1000, bandwidth=None):
        self.n_knots = n_knots
        self.bandwidth = bandwidth
        self.dim = None
        self.splines = None

    def fit(self, x):
        r"""

        Fit the monotonic spline transformation to the given data.

        Parameters:
        -----------
        x : numpy.ndarray
            Data to fit the transformation to.
        """
        self.dim = x.shape[1]

        self.splines = []
        
        for d in range(self.dim):

            kde = KernelDensityEstimator(x[:, d], 
                        bandwidth=self.bandwidth)
            
            # Define knots
            x_knots = np.linspace(np.min(x[:, d]), 
                                  np.max(x[:, d]), 
                                  self.n_knots, 
                                  dtype=np.float64)
            
            # Ensure monotonicity
            x_knots = make_strictly_monotonic(x_knots)
            
            # Compute y_knots using the inverse CDF
            y_knots = norm.ppf(kde.cdf(x_knots))

            # Ensure monotonicity
            y_knots = make_strictly_monotonic(y_knots)

            try:
                spline = MonotonicRationalQuadraticSpline(
                                x_knots=x_knots,
                                y_knots=y_knots,
                                fixed_end_slopes=True,
                                extrapolate="linear")
            except:
                raise ValueError("Could not fit monotonic spline to data.")
            self.splines.append(spline)


    def forward(self, x):
        r"""
        
        Apply the forward transformation to the given data.

        Parameters:
        -----------
        x : numpy.ndarray
            Data to transform.
        
        Returns:
        --------
        y : numpy.ndarray
            Transformed data.
        log_det : numpy.ndarray
            Logarithm of the determinant of the Jacobian of the transformation.
        """
        
        y = np.zeros_like(x)
        log_det = np.zeros(x.shape[0])

        for d in range(self.dim):
            y[:, d], dy_dx = self.splines[d].forward(x[:, d])
            log_det += np.log(dy_dx)

        return y, log_det

    def inverse(self, y):
        r"""

        Apply the inverse transformation to the given data.

        Parameters:
        -----------
        y : numpy.ndarray
            Data to transform.
        
        Returns:
        --------
        x : numpy.ndarray
            Transformed data.
        log_det : numpy.ndarray
            Logarithm of the determinant of the Jacobian of the transformation.
        """
        
        x = np.zeros_like(y)
        log_det = np.zeros(y.shape[0])

        for d in range(self.dim):
            x[:, d], dx_dy = self.splines[d].inverse(y[:, d])
            log_det += np.log(dx_dy)

        return x, log_det
    

class ProjectedSplineTransform:
    r"""
    Monotonic spline transformation along an arbitrary direction.

    Parameters:
    -----------
    direction : numpy.ndarray of shape (D,)
        Unit direction vector along which the spline transform is applied.
    n_knots : int, default=1000
        Number of knots to use in the spline.
    bandwidth : float, default=None
        Bandwidth for the KDE. If not provided, Silverman's rule of thumb is used.
    """
    def __init__(self, direction, n_knots=1000, bandwidth=None):
        self.n_knots = n_knots
        self.bandwidth = bandwidth
        self.direction = self._normalize(direction)
        self.spline = None
        self.dim = None

    def _normalize(self, v):
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            raise ValueError("Direction vector must be non-zero.")
        return v / norm_v

    def fit(self, x):
        r"""
        Fit the monotonic spline transformation to the given data along the specified direction.

        Parameters:
        -----------
        x : numpy.ndarray of shape (N, D)
            Data to fit the transformation to.
        """
        if x.ndim != 2:
            raise ValueError("Input data x must be a 2D array with shape (N, D).")

        self.dim = x.shape[1]
        # Project data onto the direction
        projections = x @ self.direction  # Shape: (N,)
        #projections = np.einsum('ij,j->i', x, self.direction)

        # Fit KDE on projections
        kde = KernelDensityEstimator(projections, bandwidth=self.bandwidth)

        # Define knots
        x_min, x_max = projections.min(), projections.max()
        if x_min == x_max:
            raise ValueError("All projections are identical; cannot create knots.")
        x_knots = np.linspace(x_min, x_max, self.n_knots, dtype=np.float64)

        # Ensure knots are strictly increasing
        x_knots = make_strictly_monotonic(x_knots)

        # Compute y_knots using the inverse CDF (probit function)
        y_knots = norm.ppf(kde.cdf(x_knots))

        # Ensure y_knots are strictly increasing
        y_knots = make_strictly_monotonic(y_knots)

        # Fit the monotonic spline
        try:
            self.spline = MonotonicRationalQuadraticSpline(
                x_knots=x_knots,
                y_knots=y_knots,
                fixed_end_slopes=True,
                extrapolate="linear"
            )
        except Exception as e:
            raise ValueError("Could not fit monotonic spline to data.") from e

    def forward(self, x):
        r"""
        Apply the forward transformation to the given data.

        Parameters:
        -----------
        x : numpy.ndarray of shape (N, D)
            Data to transform.

        Returns:
        --------
        y : numpy.ndarray of shape (N, D)
            Transformed data.
        log_det : numpy.ndarray of shape (N,)
            Logarithm of the determinant of the Jacobian of the transformation.
        """
        if self.spline is None:
            raise ValueError("The transform must be fitted before calling forward.")

        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Input data must have shape (N, {self.dim}).")

        # Project data onto the direction
        projections = x @ self.direction  # Shape: (N,)
        #projections = np.einsum('ij,j->i', x, self.direction)

        # Apply the spline forward transformation
        transformed_projections, dy_dx = self.spline.forward(projections)

        # Reconstruct the transformed data
        transformed_X = np.outer(transformed_projections, self.direction) + (x - np.outer(projections, self.direction))

        # Compute log determinant of the Jacobian
        # Since the transformation is along one direction, log_det is log dy/dx for each sample
        log_det = np.log(dy_dx + 1e-12)  # Adding epsilon for numerical stability

        return transformed_X, log_det

    def inverse(self, y):
        r"""
        Apply the inverse transformation to the given data.

        Parameters:
        -----------
        y : numpy.ndarray of shape (N, D)
            Data to inverse transform.

        Returns:
        --------
        x : numpy.ndarray of shape (N, D)
            Inverse transformed data.
        log_det : numpy.ndarray of shape (N,)
            Logarithm of the determinant of the Jacobian of the inverse transformation.
        """
        if self.spline is None:
            raise ValueError("The transform must be fitted before calling inverse.")

        if y.ndim != 2 or y.shape[1] != self.dim:
            raise ValueError(f"Input data must have shape (N, {self.dim}).")

        # Project data onto the direction
        projections = y @ self.direction  # Shape: (N,)
        #projections = np.einsum('ij,j->i', y, self.direction)

        # Apply the spline inverse transformation
        inverse_projections, dx_dy = self.spline.inverse(projections)

        # Reconstruct the inverse transformed data
        inverse_X = np.outer(inverse_projections, self.direction) + (y - np.outer(projections, self.direction))

        # Compute log determinant of the Jacobian
        # Since the transformation is along one direction, log_det is log dx/dy for each sample
        log_det = np.log(dx_dy + 1e-12)  # Adding epsilon for numerical stability
        #log_det = -np.log(1.0/(dx_dy + 1e-3))  # Adding epsilon for numerical stability

        return inverse_X, log_det

