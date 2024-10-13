import numpy as np
from scipy.stats import norm

class KernelDensityEstimator:
    r"""
    Kernel Density Estimator for 1D data using Gaussian kernels.
    
    Attributes:
    -----------
    data : (np.ndarray)
        1D array of data points.
    bandwidth : float
        Bandwidth of the Gaussian kernels.
    """
    
    def __init__(self, data, bandwidth=None):
        r"""
        Initializes the KDE1D estimator with the provided data and bandwidth.
        
        Parameters:
        -----------
        data : (array-like)
            1D array of data points.
        bandwidth : float (optional)
            Bandwidth for the kernels. If not provided, Silverman's rule of 
            thumb is used.
        """
        self.data = np.asarray(data)
        self.n = len(self.data)
        
        if self.n == 0:
            raise ValueError("Data must contain at least one data point.")
        
        self.std = np.std(self.data, ddof=1)
        
        if bandwidth is None:
            # Silverman's rule of thumb for bandwidth selection
            self.bandwidth = 1.06 * self.std * self.n ** (-1 / 5)
        else:
            self.bandwidth = bandwidth
        
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        
        # Precompute constants for efficiency
        self.inv_bandwidth = 1.0 / self.bandwidth
        self.const = 1.0 / (self.n * self.bandwidth)
    
    def pdf(self, x):
        r"""
        Evaluates the KDE PDF at the given points.
        
        Parameters:
        -----------
        x : float or array-like
            Points at which to evaluate the PDF.
        
        Returns:
        --------
        float or np.ndarray
            PDF values at the specified points.
        """
        x = np.atleast_1d(x)
        # Compute standardized distances
        u = (x[:, np.newaxis] - self.data[np.newaxis, :]) * self.inv_bandwidth
        # Gaussian kernel values
        K = norm.pdf(u)
        # Sum over all kernels
        result = self.const * np.sum(K, axis=1)
        # Return scalar if input was scalar
        if result.size == 1:
            return result[0]
        return result
    
    def cdf(self, x):
        r"""
        Evaluates the KDE CDF at the given points.
        
        Parameters:
        -----------
        x : float or array-like
            Points at which to evaluate the CDF.
        
        Returns:
        --------
        float or np.ndarray
            CDF values at the specified points.
        """
        x = np.atleast_1d(x)
        # Compute standardized distances
        u = (x[:, np.newaxis] - self.data[np.newaxis, :]) * self.inv_bandwidth
        # Gaussian CDF values
        K_cdf = norm.cdf(u)
        # Average over all kernels
        result = (1.0 / self.n) * np.sum(K_cdf, axis=1)
        # Return scalar if input was scalar
        if result.size == 1:
            return result[0]
        return result
    
    def derivative_pdf(self, x):
        r"""
        Evaluates the derivative of the KDE PDF at the given points.
        
        Parameters:
        -----------
        x : float or array-like
            Points at which to evaluate the derivative of the PDF.
        
        Returns:
        --------
        float or np.ndarray
            Derivative of the PDF at the specified points.
        """
        x = np.atleast_1d(x)
        # Compute standardized distances
        u = (x[:, np.newaxis] - self.data[np.newaxis, :]) * self.inv_bandwidth
        # Gaussian kernel values
        K = norm.pdf(u)
        # Derivative computation
        derivative = -self.const * self.inv_bandwidth * np.sum(u * K, axis=1)
        # Return scalar if input was scalar
        if derivative.size == 1:
            return derivative[0]
        return derivative

