import numpy as np
from scipy.stats import norm

from .transforms import ProjectedSplineTransform, AffineTransform, SplineTransform
from .tools import sliced_wasserstein_distance
from .tools import gradient_ascent_unit_vector

class Flow:
    r"""

    A class representing a normalizing flow model.

    Attributes:
    -----------
    n_transforms : int
        Number of transformations to apply.
    n_knots : int
        Number of knots to use in the spline transformations.
    validation_fraction : float
        Fraction of the data to use for validation.
    early_stopping : bool
        If True, stop training when the validation loss does not improve for n_iter_no_change iterations.
    n_iter_no_change : int
        Number of iterations with no improvement to wait before stopping training.
    reg_cov : float
        Regularization parameter for the covariance matrix in the affine transformation.
    whiten : bool
        If True, apply an affine transformation to whiten the data.
    warm_start : bool
        If True, continue training from the current state.
    p : int
        Power for the Wasserstein distance calculation.
    max_iter : int
        Maximum number of iterations for the gradient ascent algorithm.
    alpha_init : float
        Initial step size for the backtracking line search.
    beta : float
        Step size reduction factor for the backtracking line search.
    tol : float
        Tolerance for the gradient norm to declare convergence.
    c : float
        Armijo condition constant.
    verbose : bool
        If True, print progress information.
    initialized : bool
        If True, the model has been initialized.
    transforms : list
        List of transformations applied to the data.
    train_history : list
        List of maximum sliced Wasserstein distances on the training set.
    val_history : list
        List of maximum sliced Wasserstein distances on the validation set.
    
    Methods:
    --------
    fit(x)
        Fit the model to the data.
    forward(x)
        Apply the forward transformation to the given data.
    inverse(y)
        Apply the inverse transformation to the given data.
    log_prob(x)
        Compute the log PDF of the Flow model at the given points.
    sample(n)
        Sample from the Flow model.

    """

    def __init__(self, 
                n_transforms=100, 
                n_knots=1000, 
                validation_fraction=0.2, 
                early_stopping=True,
                n_iter_no_change=20,
                reg_cov=1e-6,
                whiten=True,
                warm_start=False,
                p=2,
                max_iter=1000,
                alpha_init=1.0,
                beta=0.2,
                tol=1e-6,
                c=1e-4,
                verbose=False,
                ):
        r"""

        Initialize a Flow model.

        Parameters:
        -----------
        n_transforms : int
            Number of transformations to apply. Default is 100.
        n_knots : int
            Number of knots to use in the spline transformations. Default is 1000.
        validation_fraction: float
            Fraction of the data to use for validation. Default is 0.2.
        early_stopping : bool
            If True, stop training when the validation loss does not improve for n_iter_no_change iterations. Default is True.
        n_iter_no_change : int
            Number of iterations with no improvement to wait before stopping training. Default is 20.
        reg_cov : float
            Regularization parameter for the covariance matrix in the affine transformation. Default is 1e-6.
        whiten : bool
            If True, apply an affine transformation to whiten the data. Default is True.
        warm_start : bool
            If True, continue training from the current state. Default is False.
        p : int
            Power for the Wasserstein distance calculation. Default is 2.
        max_iter : int
            Maximum number of iterations for the gradient ascent algorithm. Default is 1000.
        alpha_init : float
            Initial step size for the backtracking line search. Default is 1.0.
        beta : float
            Step size reduction factor for the backtracking line search. Default is 0.2.
        tol : float
            Tolerance for the gradient norm to declare convergence. Default is 1e-6.
        c : float
            Armijo condition constant. Default is 1e-4.
        verbose : bool
            If True, print progress information. Default is False.
        """
        
        self.n_transforms = n_transforms
        self.n_knots = n_knots
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.reg_cov = reg_cov
        self.whiten = whiten
        self.warm_start = warm_start
        self.p = p
        self.max_iter = max_iter
        self.alpha_init = alpha_init
        self.beta = beta
        self.tol = tol
        self.c = c
        self.verbose = verbose

        self.initialized = False
        self.transforms = None
        self.train_history = None
        self.val_history = None

    def fit(self, x):
        r"""

        Fit the model to the data.

        Parameters:
        -----------
        x : numpy.ndarray
            Data to fit the model to.
        
        """
        # Data dimensions
        N, D = x.shape

        # Split data into training and validation sets
        if self.validation_fraction > 0:
            n_val = int(self.validation_fraction * N)
            np.random.shuffle(x)
            x_val = x[:n_val]
            x_train = x[n_val:]
        else:
            x_train = x
            x_val = x

        # Check if warm start is enabled and model is initialized
        if self.warm_start and self.initialized:
            # Continue training from current state
            y_train = self.forward(x_train)[0]
            y_val = self.forward(x_val)[0]
        else:
            # Initialize lists to store transforms and history
            self.transforms = []
            self.train_history = []
            self.val_history = []

            # Perform initial affine transformation
            if self.whiten:
                at = AffineTransform(eps=self.reg_cov)
                at.fit(x_train)
                self.transforms.append(at)
                y_train = at.forward(x_train)[0]
                y_val = at.forward(x_val)[0]
            else:
                y_train = x_train
                y_val = x_val

            # Perform initial component-wise spline transformation
            st = SplineTransform(n_knots=self.n_knots)
            st.fit(y_train)
            self.transforms.append(st)

            # Set initialized flag to True
            self.initialized = True

        for i in range(self.n_transforms):

            # Apply the current transformation to the data
            y_train = self.transforms[-1].forward(y_train)[0]
            y_val = self.transforms[-1].forward(y_val)[0]

            # Compute the maximum sliced Wasserstein direction
            direction, loss = gradient_ascent_unit_vector(y_train,
                                                          p=self.p,
                                                          initial_direction=None,
                                                          max_iter=self.max_iter,
                                                          tol=self.tol,
                                                          alpha_init=self.alpha_init,
                                                          beta=self.beta,
                                                          c=self.c,
                                                          verbose=self.verbose,)

            # Compute the maximum sliced Wasserstein distance
            max_swd_train = sliced_wasserstein_distance(y_train, direction)
            max_swd_val = sliced_wasserstein_distance(y_val, direction)

            if self.verbose and (i + 1) % 10 == 0: 
                print(f"Iteration {i+1} | Training MAX-SWD {max_swd_train} | Validation MAX-SWD {max_swd_val}")

            # Append the maximum sliced Wasserstein distances to the history lists
            self.train_history.append(max_swd_train)
            self.val_history.append(max_swd_val)
            
            # Check early stopping criterion
            if self.early_stopping:
                if len(self.val_history) > self.n_iter_no_change:
                    # Check if validation loss has not improved for n_iter_no_change iterations
                    if self.val_history[-self.n_iter_no_change] - min(self.val_history[-self.n_iter_no_change:]) < self.tol:
                        if self.verbose:
                            print(f"Validation loss has not improved for {self.n_iter_no_change} iterations. Stopping.")
                        # Remove last self.n_iter_no_change elements from history lists and transforms
                        self.val_history = self.val_history[:-self.n_iter_no_change]
                        self.train_history = self.train_history[:-self.n_iter_no_change]
                        self.transforms = self.transforms[:-self.n_iter_no_change]
                        break
            
            # Create a new projected spline transform and fit it to the data
            pst = ProjectedSplineTransform(direction, 
                                           n_knots=self.n_knots)
            pst.fit(y_train)
            self.transforms.append(pst)
            

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
            Log determinant of the Jacobian matrix.
        """
        log_det = np.zeros(x.shape[0])
        for transform in self.transforms:
            x, ld = transform.forward(x)
            log_det += ld
        return x, log_det
    
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
        """
        log_det = np.zeros(y.shape[0])
        for transform in reversed(self.transforms):
            y, ld = transform.inverse(y)
            log_det += ld
        return y, log_det
    
    def log_prob(self, x):
        r"""

        Compute the log PDF of the Flow model at the given points.

        Parameters:
        -----------
        x : numpy.ndarray
            Points at which to evaluate the log PDF.

        Returns:
        --------
        numpy.ndarray
            Log PDF values at the specified points.

        """
        z, log_det = self.forward(x)
        return norm.logpdf(z).sum(axis=1) + log_det
    
    def sample(self, n):
        r"""

        Sample from the Flow model.

        Parameters:
        -----------
        n : int
            Number of samples to generate.

        Returns:
        --------
        numpy.ndarray
            Generated samples.

        """
        z = np.random.randn(n, self.transforms[-1].dim)
        x, _ = self.inverse(z)
        return x