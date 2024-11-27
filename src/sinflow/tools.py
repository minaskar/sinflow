import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import qr
from typing import Optional

def make_strictly_monotonic(x):
    r"""
    Transforms a monotonic numpy array into a strictly monotonic array
    by changing each value as little as possible.

    Parameters:
    -----------
    x : np.ndarray
        Input 1D numpy array which is monotonic (non-decreasing or non-increasing).

    Returns:
    --------
    np.ndarray
        Strictly monotonic array.
    
    Raises:
    -------
    ValueError
        If the input array is not monotonic.
    TypeError
        If the input is not a numpy array.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    if x.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    
    y = x.copy()
    N = len(y)
    
    if N == 0:
        return y  # Empty array is already strictly monotonic
    
    # Determine monotonicity
    diffs = np.diff(y)
    is_non_decreasing = np.all(diffs >= 0)
    is_non_increasing = np.all(diffs <= 0)
    
    if not (is_non_decreasing or is_non_increasing):
        raise ValueError("Input array is not monotonic (neither non-decreasing nor non-increasing).")
    
    if is_non_decreasing:
        # Make strictly increasing
        for i in range(1, N):
            if y[i] <= y[i-1]:
                # Adjust to the next representable float greater than y[i-1]
                y[i] = np.nextafter(y[i-1], np.inf)
    else:
        # Make strictly decreasing
        for i in range(1, N):
            if y[i] >= y[i-1]:
                # Adjust to the next representable float less than y[i-1]
                y[i] = np.nextafter(y[i-1], -np.inf)
    
    return y


def max_sliced_wasserstein_distance(x, 
                                    direction, 
                                    p=2):
    r"""
    Compute the max sliced Wasserstein distance between the data x and a standard normal distribution
    along the given direction.

    Parameters:
    -----------
    x : np.ndarray of shape [N, D]
        The data points.
    direction : np.ndarray of shape [D,] 
        The projection vector (unit norm).
    p : int 
        The power for the Wasserstein distance calculation (default is 2).

    Returns:
    --------
    distance : float 
        The sliced Wasserstein distance.
    """
    # Project the data onto the direction
    projections = x @ direction  # Shape: [N,]

    # Compute the sorted quantiles of the standard normal distribution
    ranks = (np.arange(1, len(projections) + 1) - 0.5) / len(projections)
    sorted_y = norm.ppf(ranks)  # Shape: [N,]

    # Sort the projections
    sorted_indices = np.argsort(projections)
    projections_sorted = projections[sorted_indices]  # Shape: [N,]

    # Compute the Wasserstein distance
    if p == 1:
        diff = np.abs(projections_sorted - sorted_y)
    elif p == 2:
        diff = (projections_sorted - sorted_y) ** 2
    else:
        diff = np.abs(projections_sorted - sorted_y) ** p

    return np.mean(diff)

def max_sliced_wasserstein_distance_gradient(x, 
                                             direction, 
                                             p=2):
    r"""
    Compute the gradient of the max sliced Wasserstein distance with respect to the direction vector.

    Parameters:
    -----------
    x : np.ndarray of shape [N, D]
        The data points.
    direction : np.ndarray of shape [D,] 
        The projection vector (unit norm).
    p : int 
        The power for the Wasserstein distance calculation (default is 2).

    Returns:
    --------
    gradient : np.ndarray of shape [D,]
        The gradient of the sliced Wasserstein distance with respect to the direction vector.
    """
    # Ensure direction is a unit vector
    direction = direction / np.linalg.norm(direction)
    
    # Project the data onto the direction
    projections = x @ direction  # Shape: [N,]

    # Compute the sorted quantiles of the standard normal distribution
    N = len(projections)
    ranks = (np.arange(1, N + 1) - 0.5) / N
    sorted_y = norm.ppf(ranks)  # Shape: [N,]

    # Sort the projections
    sorted_indices = np.argsort(projections)
    projections_sorted = projections[sorted_indices]  # Shape: [N,]
    x_sorted = x[sorted_indices]  # Shape: [N, D]

    # Compute the difference
    diff = projections_sorted - sorted_y  # Shape: [N,]

    if p == 1:
        # Subgradient: sign(diff) where diff != 0, undefined at diff=0
        # To handle diff=0, we can set subgradient to 0
        sign_diff = np.sign(diff)
        sign_diff[diff == 0] = 0
        coeff = 1.0
    elif p == 2:
        coeff = 2.0 * diff
    else:
        coeff = p * np.abs(diff) ** (p - 1) * np.sign(diff)
    
    # Compute the gradient
    gradient = (coeff @ x_sorted) / N  # Shape: [D,]

    # Project the gradient to the tangent space of the unit sphere
    # to ensure the direction remains a unit vector
    gradient -= np.dot(gradient, direction) * direction

    return gradient


def max_sliced_wasserstein_distance_vectorized(
    x: np.ndarray,
    directions: np.ndarray,
    p: int = 2,
    sum_over_directions: bool = True,
) -> float:
    """
    Compute the sum of max sliced Wasserstein distances between the data x and a standard normal distribution
    along the given multiple directions.

    Parameters:
    -----------
    x : np.ndarray of shape [N, D]
        The data points.
    directions : np.ndarray of shape [D, K] 
        The projection matrix where each column is a projection vector (unit norm).
    p : int 
        The power for the Wasserstein distance calculation (default is 2).

    Returns:
    --------
    total_distance : float 
        The total sliced Wasserstein distance across all directions.
    """
    N, D = x.shape
    D_dir, K = directions.shape
    assert D == D_dir, "Dimension mismatch between data and directions."

    # Project the data onto all directions: [N, K]
    projections = x @ directions  # Shape: [N, K]

    # Compute the sorted quantiles of the standard normal distribution: [N,]
    ranks = (np.arange(1, N + 1) - 0.5) / N
    sorted_y = norm.ppf(ranks)  # Shape: [N,]

    # Sort the projections along each direction: [N, K]
    sorted_projections = np.sort(projections, axis=0)  # Shape: [N, K]

    # Compute the difference based on p
    if p == 1:
        diff = np.abs(sorted_projections - sorted_y[:, np.newaxis])  # Shape: [N, K]
    elif p == 2:
        diff = (sorted_projections - sorted_y[:, np.newaxis]) ** 2
    else:
        diff = np.abs(sorted_projections - sorted_y[:, np.newaxis]) ** p

    # Compute Wasserstein distance for each direction and sum them
    distance_per_direction = np.mean(diff, axis=0)  # Shape: [K,]
    if sum_over_directions:
        return np.mean(distance_per_direction)
    else:
        return distance_per_direction


def gradient_max_sliced_wasserstein_distance_vectorized(
    x: np.ndarray,
    directions: np.ndarray,
    p: int = 2,
) -> np.ndarray:
    """
    Compute the gradient of the sum of max sliced Wasserstein distances between the data x 
    and a standard normal distribution with respect to the given multiple directions.

    Parameters:
    -----------
    x : np.ndarray of shape [N, D]
        The data points.
    directions : np.ndarray of shape [D, K] 
        The projection matrix where each column is a projection vector (unit norm).
    p : int 
        The power for the Wasserstein distance calculation (default is 2).
    sum_over_directions : bool
        If True, return the average gradient over all directions. 
        If False, return the gradient per direction.

    Returns:
    --------
    gradient : np.ndarray of shape [D, K] or [D,]
        The gradient with respect to the directions. Shape depends on `sum_over_directions`.
    """
    N, D = x.shape
    D_dir, K = directions.shape
    assert D == D_dir, "Dimension mismatch between data and directions."

    # Project the data onto all directions: [N, K]
    projections = x @ directions  # Shape: [N, K]

    # Sort the projections and get the sorted indices: [N, K]
    sorted_indices = np.argsort(projections, axis=0)  # Indices to sort each column
    sorted_projections = np.take_along_axis(projections, sorted_indices, axis=0)  # [N, K]

    # Compute the sorted quantiles of the standard normal distribution: [N,]
    ranks = (np.arange(1, N + 1) - 0.5) / N
    sorted_y = norm.ppf(ranks)  # Shape: [N,]

    # Compute the difference: [N, K]
    diff = sorted_projections - sorted_y[:, np.newaxis]  # Broadcasting Y to [N, K]

    # Compute the coefficient for the gradient: [N, K]
    if p == 1:
        grad_coeff = np.sign(diff)  # [N, K]
    else:
        grad_coeff = p * (np.abs(diff) ** (p - 1)) * np.sign(diff)  # [N, K]

    # Initialize grad_coeff_original to map gradients back to original data points: [N, K]
    grad_coeff_original = np.zeros_like(projections)  # [N, K]

    # Map grad_coeff back to original data points
    for k in range(K):
        grad_coeff_original[sorted_indices[:, k], k] = grad_coeff[:, k]

    # Expand dimensions of x to [N, 1, D] for broadcasting
    x_expanded = x[:, np.newaxis, :]  # [N, 1, D]

    # Expand grad_coeff_original to [N, K, 1] for broadcasting
    grad_coeff_expanded = grad_coeff_original[:, :, np.newaxis]  # [N, K, 1]

    # Multiply element-wise: [N, K, D]
    multiplied = x_expanded * grad_coeff_expanded  # [N, K, D]

    # Compute the mean over N: [K, D]
    gradient_per_direction = multiplied.mean(axis=0)  # [K, D]

    # Transpose to match directions' shape: [D, K]
    gradient = gradient_per_direction.T  # [D, K]

    # If sum_over_directions=True in loss, average the gradient over K
    gradient = gradient / K

    return gradient


def maximize_max_sliced_wasserstein_distance_SGD(
        x: np.ndarray,
        K: int,
        max_iter: int = 1000,
        learning_rate: float = 1.0,
        beta: float = 0.2,
        tol: float = 1e-6,
        verbose: bool = False,
        initial_A: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Perform maximization of the max_sliced_wasserstein_distance on the Stiefel manifold using SGD.

    Parameters:
    - x: np.ndarray, input data of shape (n_samples, d)
    - K: int, number of orthonormal directions
    - max_iter: int, maximum number of iterations
    - learning_rate: float, learning rate for SGD
    - beta: float, decay rate for the learning rate
    - tol: float, tolerance for convergence
    - verbose: bool, if True, print progress
    - initial_A: Optional[np.ndarray], initial orthonormal matrix of shape (d, K)

    Returns:
    - A: np.ndarray, optimized orthonormal matrix of shape (d, K)
    """

    d = x.shape[1]

    tau = learning_rate

    if initial_A is not None:
        A = initial_A
        # Ensure A is orthonormal
        A, _ = qr(A, mode='economic')
    else:
        # Initialize A as a random orthonormal matrix
        random_matrix = np.random.randn(d, K)
        A, _ = qr(random_matrix, mode='economic')

    # Compute current objective
    F = max_sliced_wasserstein_distance_vectorized(x, A)

    for iteration in range(max_iter):
        
        # Compute gradient
        grad = gradient_max_sliced_wasserstein_distance_vectorized(x, A)  # Shape: (d, K)

        # Compute G = -grad
        G = -grad  # Shape: (d, K)

        # Construct U and V matrices
        U = np.hstack((G, A))  # Shape: (d, 2K)
        V = np.hstack((A, -G))  # Shape: (d, 2K)

        # Compute M = I + (tau/2) V^T U
        M = np.eye(2 * K) + (tau / 2) * (V.T @ U)  # Shape: (2K, 2K)
        M_inv = np.linalg.inv(M)

        # Compute the update term: U @ M_inv @ (V.T @ A)
        update_term = U @ (M_inv @ (V.T @ A))  # Shape: (d, K)

        # Update A
        A = A - tau * update_term

        # Check for convergence
        F_new = max_sliced_wasserstein_distance_vectorized(x, A)
        improvement = F_new - F
        update_term_norm = np.linalg.norm(update_term)
        gradient_norm = np.linalg.norm(grad)
        if verbose:
            print(f"Iteration {iteration}: F = {F_new:.6f}, Improvement = {improvement:.6f}")
            print(f"Update term norm = {update_term_norm:.6f}")
            print(f"Gradient norm = {gradient_norm:.6f}")
            print(f"Learning rate = {tau:.6f}")

        if improvement < 0:
            tau *= beta

        if np.abs(improvement) < tol:
            if verbose:
                print(f"Convergence achieved at iteration {iteration}.")
            break
        else:
            F = F_new

    # Re-orthonormalize A to prevent numerical drift
    A, _ = qr(A, mode='economic')

    max_swd = max_sliced_wasserstein_distance_vectorized(x, A, sum_over_directions=False)
    order = np.argsort(max_swd)[::-1]

    return A[:,order]

def maximize_max_sliced_wasserstein_distance_SGD_backtracking(
        x: np.ndarray,
        K: int,
        max_iter: int = 1000,
        learning_rate: float = 1.0,
        tol: float = 1e-6,
        alpha=0.5,
        beta=0.5,
        verbose: bool = False,
        initial_A: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Perform maximization of the max_sliced_wasserstein_distance on the Stiefel manifold using SGD.

    Parameters:
    - x: np.ndarray, input data of shape (n_samples, d)
    - K: int, number of orthonormal directions
    - max_iter: int, maximum number of iterations
    - learning_rate: float, learning rate for SGD
    - tol: float, tolerance for convergence
    - verbose: bool, if True, print progress
    - initial_A: Optional[np.ndarray], initial orthonormal matrix of shape (d, K)

    Returns:
    - A: np.ndarray, optimized orthonormal matrix of shape (d, K)
    """

    d = x.shape[1]

    tau = learning_rate

    if initial_A is not None:
        A = initial_A
        # Ensure A is orthonormal
        A, _ = qr(A, mode='economic')
    else:
        # Initialize A as a random orthonormal matrix
        random_matrix = np.random.randn(d, K)
        A, _ = qr(random_matrix, mode='economic')

    # Compute current objective
    F = max_sliced_wasserstein_distance_vectorized(x, A)
    loss = -F

    for iteration in range(max_iter):
        
        # Compute gradient
        grad = gradient_max_sliced_wasserstein_distance_vectorized(x, A)  # Shape: (d, K)

        # Compute G = -grad
        G = -grad  # Shape: (d, K)

        # Construct U and V matrices
        U = np.hstack((G, A))  # Shape: (d, 2K)
        V = np.hstack((A, -G))  # Shape: (d, 2K)

        while True:
            # Compute M = I + (tau/2) V^T U
            M = np.eye(2 * K) + (tau / 2) * (V.T @ U)  # Shape: (2K, 2K)
            M_inv = np.linalg.inv(M)

            # Compute the update term: U @ M_inv @ (V.T @ A)
            dA = - U @ (M_inv @ (V.T @ A))  # Shape: (d, K)

            # Update A
            A_new = A + tau * dA

            # Compute the new objective
            F_new = max_sliced_wasserstein_distance_vectorized(x, A_new)
            loss_new = -F_new

            # Compute the expected improvement
            expected_improvement = tau * np.sum(-grad * dA)

            # Compute the actual improvement
            improvement = loss_new - loss

            # Check the Armijo condition
            if improvement >= alpha * expected_improvement:
                break
            else:
                tau *= beta

        # Update A
        A = A_new
        F = F_new

        # Check for convergence
        F_new = max_sliced_wasserstein_distance_vectorized(x, A)
        improvement = F_new - F
        if verbose:
            print(f"Iteration {iteration}: F = {F_new:.6f}, Improvement = {improvement:.6f}")
            print(f"Learning rate = {tau:.6f}")

        if np.abs(improvement) < tol:
            if verbose:
                print(f"Convergence achieved at iteration {iteration}.")
            break

    # Re-orthonormalize A to prevent numerical drift
    A, _ = qr(A, mode='economic')

    max_swd = max_sliced_wasserstein_distance_vectorized(x, A, sum_over_directions=False)
    order = np.argsort(max_swd)[::-1]

    return A#[:,order]