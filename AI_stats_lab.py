import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    data = np.asarray(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    if not (0 < theta < 1):
        raise ValueError("theta must be in (0,1)")
    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must contain only 0 and 1")

    # Log-likelihood
    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    data = np.asarray(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Data must contain only 0 and 1")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    # Counts
    num_successes = int(np.sum(data))
    n = data.size
    num_failures = n - num_successes

    # MLE (mean of Bernoulli)
    mle = num_successes / n

    # Log-likelihoods
    log_likelihoods = {}
    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(data, theta)
        log_likelihoods[theta] = ll

    # Best candidate (first max)
    best_candidate = None
    best_ll = -np.inf
    for theta in candidate_thetas:
        if log_likelihoods[theta] > best_ll:
            best_ll = log_likelihoods[theta]
            best_candidate = theta

    return {
        'mle': mle,
        'num_successes': num_successes,
        'num_failures': num_failures,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    data = np.asarray(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if not np.all((data >= 0) & (np.floor(data) == data)):
        raise ValueError("Data must be nonnegative integers")

    # Log-likelihood using lgamma
    return np.sum(data * np.log(lam) - lam - np.array([math.lgamma(x + 1) for x in data]))


def poisson_mle_analysis(data, candidate_lambdas=None):
    data = np.asarray(data)

    # Validation
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    if not np.all((data >= 0) & (np.floor(data) == data)):
        raise ValueError("Data must be nonnegative integers")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    # Basic stats
    n = data.size
    total_count = int(np.sum(data))
    sample_mean = total_count / n

    # MLE (mean of Poisson)
    mle = sample_mean

    # Log-likelihoods
    log_likelihoods = {}
    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(data, lam)
        log_likelihoods[lam] = ll

    # Best candidate (first max)
    best_candidate = None
    best_ll = -np.inf
    for lam in candidate_lambdas:
        if log_likelihoods[lam] > best_ll:
            best_ll = log_likelihoods[lam]
            best_candidate = lam

    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': total_count,
        'n': n,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }
