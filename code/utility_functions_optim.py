import numpy as np

def phi_1(x):
    return x * (1 - x)

def phi_2(x):
    return 2 + 2 * x * (1 - 2 * x) + (1 / 50) * np.sin(52 * np.pi * x)

def gaussian_noise(x, seed=None):
    rng = np.random.default_rng(seed)
    sigma = (1 - x)**2  # Variance decreases with x
    return rng.normal(0, sigma)

def phi_1_noisy(x, seed=None):
    return phi_1(x) + gaussian_noise(x)

def phi_2_noisy(x, seed=None):
    return phi_2(x) + gaussian_noise(x)

import numpy as np

def build_proposal_probabilities(mu, sigma, num_samples=1000):
    """
    Build proposal probabilities from the joint Gaussian distribution.

    Args:
        mu (array): Mean vector [mu_1, mu_2, ..., mu_m].
        sigma (array): Variance vector [var_1, var_2, ..., var_m].
        num_samples (int): Number of samples to draw.

    Returns:
        proposal_probabilities (array): Normalized probabilities for each interval.
    """
    m = len(mu)  # Number of intervals
    covariance = np.diag(sigma)  # Diagonal covariance matrix
    samples = np.random.multivariate_normal(mu, covariance, size=num_samples)
    
    # Count maxima
    max_counts = np.zeros(m)
    for sample in samples:
        max_idx = np.argmax(sample)
        max_counts[max_idx] += 1

    # Normalize to get proposal probabilities
    proposal_probabilities = max_counts / num_samples
    return proposal_probabilities

def get_entropy(P_t):
    P_t = P_t[P_t>0]
    return -np.sum(P_t*np.log(P_t))

def refine_interval_with_cdf(intervals, proposal_probabilities, chosen_index, num_subintervals):
    """
    Refines the chosen interval based on the scaled CDF of proposal probabilities.

    Args:
        intervals (list): Current intervals.
        proposal_probabilities (list): Current proposal probabilities.
        chosen_index (int): Index of the interval to refine.
        num_subintervals (int): Number of subintervals to divide into.

    Returns:
        new_intervals (list): Refined intervals.
    """
    # Get bounds of the chosen interval
    a, b = intervals[chosen_index]

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(proposal_probabilities)
    
    # Rescale the CDF for the chosen interval
    cdf_min = cdf[chosen_index]
    cdf_max = cdf[chosen_index + 1] if chosen_index + 1 < len(cdf) else 1  # Handle edge case
    scaled_cdf = (cdf - cdf_min) / (cdf_max - cdf_min)
    
    # Find points where scaled CDF equals k/m
    refined_boundaries = [a + (b - a) * (k / num_subintervals) for k in range(1, num_subintervals)]
    
    # Construct new subintervals
    refined_intervals = [[a, refined_boundaries[0]]] + \
                        [[refined_boundaries[i], refined_boundaries[i + 1]] for i in range(len(refined_boundaries) - 1)] + \
                        [[refined_boundaries[-1], b]]

    return refined_intervals

def refine_interval_and_neighbors(intervals, chosen_index, num_subdivisions):
    """
    Refine the chosen interval by shrinking it and adjusting its neighbors.
    
    Args:
        intervals (list): Current intervals.
        chosen_index (int): Index of the interval to refine.
        num_subdivisions (int): Factor by which to shrink the interval.

    Returns:
        list: Updated intervals with the refined chosen interval.
    """
    # Get bounds of the chosen interval
    a, b = intervals[chosen_index]
    original_length = b - a
    
    # Compute the shrink factor
    delta = original_length / num_subdivisions
    
    # Shrink the chosen interval
    a_new = a + delta
    b_new = b - delta
    refined_interval = [a_new, b_new]
    
    # Adjust neighbors if they exist
    if chosen_index > 0:  # There is a left neighbor
        intervals[chosen_index - 1][1] = a_new  # Expand upper bound of left neighbor
    if chosen_index < len(intervals) - 1:  # There is a right neighbor
        intervals[chosen_index + 1][0] = b_new  # Expand lower bound of right neighbor
    
    # Replace the chosen interval with the refined interval
    intervals[chosen_index] = refined_interval
    
    return intervals