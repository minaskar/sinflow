import numpy as np

class MonotonicRationalQuadraticSpline:
    def __init__(self, x_knots, y_knots, fixed_end_slopes=False, extrapolate="linear"):
        r"""
        Initialize the Monotonic Rational Quadratic Spline.

        Parameters:
        -----------
        x_knots : array-like, shape (M,)
            Strictly increasing x-coordinates of the knots.
        y_knots : array-like, shape (M,)
            Strictly increasing y-coordinates of the knots.
        fixed_end_slopes : bool, default=True
            If True, set the first and last derivatives to 1.
            If False, compute the first and last derivatives based on the neighboring slopes.
        extrapolate : str, default="linear"
            Extrapolation method for out-of-bounds x values.
            - "linear": Linear extrapolation.
            - "constant": Constant extrapolation using the first or last y value.
        """
        x_knots = np.asarray(x_knots, dtype=np.float64)
        y_knots = np.asarray(y_knots, dtype=np.float64)
        
        if x_knots.ndim != 1 or y_knots.ndim != 1:
            raise ValueError("x_knots and y_knots must be one-dimensional arrays.")
        if len(x_knots) != len(y_knots):
            raise ValueError("x_knots and y_knots must have the same length.")
        if not np.all(np.diff(x_knots) > 0):
            raise ValueError("x_knots must be strictly increasing.")
        if not np.all(np.diff(y_knots) > 0):
            raise ValueError("y_knots must be strictly increasing.")
        
        self.x_knots = x_knots
        self.y_knots = y_knots
        self.M = len(x_knots)
        self.fixed_end_slopes = fixed_end_slopes
        self.extrapolate = extrapolate
        
        # Compute s_m = (y_{m+1} - y_m)/(x_{m+1} - x_m), shape (M-1,)
        self.s_m = (self.y_knots[1:] - self.y_knots[:-1]) / (self.x_knots[1:] - self.x_knots[:-1])
        
        # Initialize y_m'
        self.y_m_prime = np.zeros(self.M, dtype=np.float64)
        
        if self.M > 2:
            # Compute derivatives for internal knots
            self.y_m_prime[1:-1] = (
                self.s_m[:-1] * (self.x_knots[2:] - self.x_knots[1:-1]) +
                self.s_m[1:] * (self.x_knots[1:-1] - self.x_knots[:-2])
            ) / (self.x_knots[2:] - self.x_knots[:-2])
        
        if self.fixed_end_slopes:
            self.y_m_prime[0] = 1.0
            self.y_m_prime[-1] = 1.0
        else:
            # Compute end derivatives using the same formula as internal points
            if self.M >= 2:
                self.y_m_prime[0] = self.s_m[0]
                self.y_m_prime[-1] = self.s_m[-1]
        
        # Ensure all derivatives are positive
        if not np.all(self.y_m_prime > 0):
            raise ValueError("All derivatives y_m' must be positive.")
        
        # Compute sigma_m = y_{m+1}' + y_m' - 2 * s_m, shape (M-1,)
        self.sigma_m = self.y_m_prime[:-1] + self.y_m_prime[1:] - 2 * self.s_m
        
        # Precompute delta_x and delta_y for efficiency
        self.delta_x = self.x_knots[1:] - self.x_knots[:-1]  # shape (M-1,)
        self.delta_y = self.y_knots[1:] - self.y_knots[:-1]  # shape (M-1,)
        
    def forward(self, x):
        r"""
        Compute the forward transformation y = f(x) and its derivative dy/dx.

        Parameters:
        -----------
        x : scalar or array-like
            Input value(s).

        Returns:
        --------
        y : scalar or ndarray
            Transformed value(s).
        dy_dx : scalar or ndarray
            Derivative(s) at the input value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        dy_dx = np.empty_like(x)
        
        # Find the bin indices for each x
        bin_indices = np.searchsorted(self.x_knots, x, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, self.M - 2)  # Ensure within [0, M-2]
        
        # Handle extrapolation
        below = x < self.x_knots[0]
        above = x > self.x_knots[-1]
        within = ~(below | above)
        
        # Compute y and dy/dx for within knots
        m = bin_indices[within]
        xi = (x[within] - self.x_knots[m]) / self.delta_x[m]
        s_m = self.s_m[m]
        sigma_m = self.sigma_m[m]
        y_m = self.y_knots[m]
        y_m_prime = self.y_m_prime[m]
        y_m1_prime = self.y_m_prime[m + 1]
        
        numerator = s_m * xi**2 + y_m_prime * xi * (1 - xi)
        denominator = s_m + sigma_m * xi * (1 - xi)
        y_within = y_m + self.delta_y[m] * numerator / denominator
        y[within] = y_within
        
        # Compute derivative for within knots
        dy_dx_within_numerator = s_m**2 * (y_m1_prime * xi**2 + 2 * s_m * xi * (1 - xi) + y_m_prime * (1 - xi)**2)
        dy_dx_within_denominator = (s_m + sigma_m * xi * (1 - xi))**2
        dy_dx_within = dy_dx_within_numerator / dy_dx_within_denominator
        dy_dx[within] = dy_dx_within
        
        # Extrapolate below
        if np.any(below):
            if self.extrapolate == "linear":
                y[below] = self.y_knots[0] + self.y_m_prime[0] * (x[below] - self.x_knots[0])
                dy_dx[below] = self.y_m_prime[0]
            elif self.extrapolate == "constant":
                y[below] = self.y_knots[0]
                dy_dx[below] = 0.0
        
        # Extrapolate above
        if np.any(above):
            if self.extrapolate == "linear":
                y[above] = self.y_knots[-1] + self.y_m_prime[-1] * (x[above] - self.x_knots[-1])
                dy_dx[above] = self.y_m_prime[-1]
            elif self.extrapolate == "constant":
                y[above] = self.y_knots[-1]
                dy_dx[above] = 0.0
        
        return y, dy_dx
    
    def inverse(self, y):
        r"""
        Compute the inverse transformation x = f^{-1}(y) and its derivative dx/dy.

        Parameters:
        -----------
        y : scalar or array-like
            Output value(s).

        Returns:
        --------
        x : scalar or ndarray
            Inverted value(s).
        dx_dy : scalar or ndarray
            Derivative(s) at the output value(s).
        """
        y = np.asarray(y, dtype=np.float64)
        x = np.empty_like(y)
        dx_dy = np.empty_like(y)
        
        # Find the bin indices for each y
        bin_indices = np.searchsorted(self.y_knots, y, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, self.M - 2)  # Ensure within [0, M-2]
        
        # Handle extrapolation
        below = y < self.y_knots[0]
        above = y > self.y_knots[-1]
        within = ~(below | above)
        
        # Compute x for within knots
        m = bin_indices[within]
        zeta = (y[within] - self.y_knots[m]) / self.delta_y[m]
        s_m = self.s_m[m]
        sigma_m = self.sigma_m[m]
        y_m_prime = self.y_m_prime[m]
        y_m1_prime = self.y_m_prime[m + 1]
        
        a = (s_m - y_m_prime) + sigma_m * zeta
        b = y_m_prime - sigma_m * zeta
        c = -s_m * zeta
        
        discriminant = b**2 - 4 * a * c
        # Introduce a small tolerance to handle numerical precision
        epsilon = 1e-12
        discriminant = np.maximum(discriminant, 0.0)
    
        sqrt_discriminant = np.sqrt(discriminant)
        # Compute both roots
        xi1 = (-b + sqrt_discriminant) / (2 * a)
        xi2 = (-b - sqrt_discriminant) / (2 * a)
    
        # Initialize xi with NaNs
        xi = np.full_like(xi1, np.nan)
    
        # Assign xi1 or xi2 based on which lies within [0, 1], considering tolerance
        valid_xi1 = (xi1 >= -epsilon) & (xi1 <= 1.0 + epsilon)
        valid_xi2 = (xi2 >= -epsilon) & (xi2 <= 1.0 + epsilon)
    
        xi[valid_xi1] = xi1[valid_xi1]
        xi[valid_xi2] = xi2[valid_xi2]
    
        # Handle cases where both roots are valid or none are valid
        # Since the spline is monotonic, only one root should be valid
        # If both are valid, choose the one closest to the expected direction
        both_valid = valid_xi1 & valid_xi2
        if np.any(both_valid):
            # For monotonic increasing, xi should be closer to 0 for lower y and closer to 1 for higher y
            # Here, we can select the root that results in xi closer to the expected position
            # Alternatively, choose xi1 as the primary root
            xi[both_valid] = xi1[both_valid]
    
        # Assign nearest valid xi if any are still NaN
        # This handles floating-point overflows beyond [0,1]
        xi = np.where(np.isnan(xi), np.clip(xi, 0.0, 1.0), xi)
        # Check for cases where neither root is valid
        #if np.any(np.isnan(xi)):
        #    raise ValueError("No valid root found within [0, 1] for some y values.")
        
        x_m = self.x_knots[m]
        delta_x = self.delta_x[m]
        x_within = x_m + delta_x * xi
        x[within] = x_within

        #if np.any(np.isnan(x)):
        #    raise ValueError("NaN encountered in inverse transformation.")
        
        # Compute dy/dx at the computed x to obtain dx/dy
        dy_dx_at_x = self.derivative(x_within)
        dx_dy_within = 1.0 / dy_dx_at_x
        dx_dy[within] = dx_dy_within
        #dx_dy[np.isnan(dx_dy)] = np.inf # Check this
        
        # Extrapolate below
        if np.any(below):
            if self.extrapolate == "linear":
                x[below] = self.x_knots[0] + (y[below] - self.y_knots[0]) / self.y_m_prime[0]
                dx_dy[below] = 1.0 / self.y_m_prime[0]
            elif self.extrapolate == "constant":
                x[below] = self.x_knots[0]
                dx_dy[below] = 0.0
        
        # Extrapolate above
        if np.any(above):
            if self.extrapolate == "linear":
                x[above] = self.x_knots[-1] + (y[above] - self.y_knots[-1]) / self.y_m_prime[-1]
                dx_dy[above] = 1.0 / self.y_m_prime[-1]
            elif self.extrapolate == "constant":
                x[above] = self.x_knots[-1]
                dx_dy[above] = 0.0
        
        return x, dx_dy
    
    def derivative(self, x):
        r"""
        Compute the derivative dy/dx at given x.

        Parameters:
        -----------
        x : scalar or array-like
            Input value(s).

        Returns:
        --------
        dy_dx : scalar or ndarray
            Derivative(s) at the input value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        dy_dx = np.empty_like(x)
        
        # Find the bin indices for each x
        bin_indices = np.searchsorted(self.x_knots, x, side='right') -1
        bin_indices = np.clip(bin_indices, 0, self.M -2)  # Ensure within [0, M-2]
        
        # Handle extrapolation
        below = x < self.x_knots[0]
        above = x > self.x_knots[-1]
        within = ~(below | above)
        
        # Compute dy/dx for within knots
        m = bin_indices[within]
        xi = (x[within] - self.x_knots[m]) / self.delta_x[m]
        s_m = self.s_m[m]
        sigma_m = self.sigma_m[m]
        y_m_prime = self.y_m_prime[m]
        y_m1_prime = self.y_m_prime[m + 1]
        
        numerator = s_m**2 * (y_m1_prime * xi**2 + 2 * s_m * xi * (1 - xi) + y_m_prime * (1 - xi)**2)
        denominator = (s_m + sigma_m * xi * (1 - xi))**2
        dy_dx_within = numerator / denominator
        dy_dx[within] = dy_dx_within
        
        # Extrapolate below
        if np.any(below):
            if self.extrapolate == "linear":
                dy_dx[below] = self.y_m_prime[0]
            elif self.extrapolate == "constant":
                dy_dx[below] = 0.0
        
        # Extrapolate above
        if np.any(above):
            if self.extrapolate == "linear":
                dy_dx[above] = self.y_m_prime[-1]
            elif self.extrapolate == "constant":
                dy_dx[above] = 0.0
        
        return dy_dx