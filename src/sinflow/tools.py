import numpy as np
from scipy.stats import norm

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


def sliced_wasserstein_distance(x, direction, p=2):
    r"""
    Compute the sliced Wasserstein distance between the data x and a standard normal distribution
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
    #projections = x @ direction  # Shape: [N,]
    projections = np.einsum('ij,j->i', x, direction)

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

def gradient_sliced_wasserstein_distance(x, direction, p=2):
    r"""
    Compute the gradient of the sliced Wasserstein distance with respect to the direction unit vector.

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
        The gradient with respect to the direction.
    """
    N, D = x.shape

    # Ensure the direction is a unit vector
    direction = direction / np.linalg.norm(direction)

    # Project the data onto the direction
    #projections = x @ direction  # Shape: [N,]
    projections = np.einsum('ij,j->i', x, direction)

    # Compute the sorted quantiles of the standard normal distribution
    ranks = (np.arange(1, N + 1) - 0.5) / N
    sorted_y = norm.ppf(ranks)  # Shape: [N,]

    # Sort the projections
    sorted_indices = np.argsort(projections)
    projections_sorted = projections[sorted_indices]  # Shape: [N,]
    sorted_x = x[sorted_indices]  # Shape: [N, D]

    # Compute the difference based on p
    if p == 1:
        # Subgradient for p=1
        diff = np.sign(projections_sorted - sorted_y)  # Shape: [N,]
    elif p == 2:
        # Gradient for p=2
        diff = 2 * (projections_sorted - sorted_y)  # Shape: [N,]
    else:
        # Gradient for general p
        diff = p * np.abs(projections_sorted - sorted_y) ** (p - 1) * np.sign(projections_sorted - sorted_y)  # Shape: [N,]

    # Compute the raw gradient
    #raw_gradient = (sorted_x.T @ diff) / N  # Shape: [D,]
    raw_gradient = np.einsum('ij,i->j', sorted_x, diff) / N

    # Project the gradient onto the tangent space of the unit sphere
    # This ensures that the updated direction remains a unit vector
    #projection = np.dot(raw_gradient, direction) * direction
    projection = np.einsum('i,i->', raw_gradient, direction) * direction
    gradient = raw_gradient - projection  # Shape: [D,]

    return gradient


def gradient_ascent_unit_vector(
    x,
    p=2,
    initial_direction=None,
    max_iter=1000,
    tol=1e-6,
    alpha_init=1.0,
    beta=0.2,
    c=1e-4,
    verbose=False
):
    r"""
    Gradient ascent algorithm to maximize the sliced Wasserstein distance using backtracking line search.

    Parameters:
    -----------
    x : np.ndarray of shape [N, D]
        The data points.
    p : int
        The power for the Wasserstein distance calculation (default is 2).
    initial_direction : np.ndarray of shape [D,]
        Optional initial unit vector.
    max_iter : int 
        Maximum number of iterations (default: 1000).
    tol : float 
        Tolerance for the gradient norm to declare convergence (default: 1e-6).
    alpha_init : float 
        Initial step size for backtracking (default: 1.0).
    beta : float
        Step size reduction factor for backtracking (default: 0.2).
    c : float
        Armijo condition constant (default: 1e-4).
    verbose : bool
        If True, prints progress information.

    Returns:
    --------
    direction : np.ndarray of shape [D,]
        The optimized unit vector.
    loss : float
        The final loss (max-SWD) value.
    """
    N, D = x.shape

    # Initialize the direction
    if initial_direction is None:
        direction = np.random.randn(D)
        direction /= np.linalg.norm(direction)
    else:
        direction = initial_direction / np.linalg.norm(initial_direction)

    for iteration in range(1, max_iter + 1):
        # Compute current loss
        current_loss = sliced_wasserstein_distance(x, direction, p)

        # Compute gradient
        gradient = gradient_sliced_wasserstein_distance(x, direction, p)
        grad_norm = np.linalg.norm(gradient)

        # Check for convergence
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iteration {iteration} with gradient norm {grad_norm:.6e}")
            break

        # Backtracking line search
        step_size = alpha_init
        while step_size > 1e-10:
            # Propose new direction
            direction_new = direction + step_size * gradient
            direction_new /= np.linalg.norm(direction_new)

            # Compute new loss
            new_loss = sliced_wasserstein_distance(x, direction_new, p)

            # Check Armijo condition for ascent
            if new_loss >= current_loss + c * step_size * (grad_norm ** 2):
                break  # Accept the step
            else:
                step_size *= beta  # Reduce the step size

        else:
            if verbose:
                print(f"Step size became too small at iteration {iteration}. Stopping.")
            break  # Exit if step size is too small

        # Update direction
        direction = direction_new

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {current_loss:.6f}, Gradient Norm = {grad_norm:.6e}, Step Size = {step_size:.6e}")

    # Final loss
    final_loss = sliced_wasserstein_distance(x, direction, p)

    return direction, final_loss