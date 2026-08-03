import numpy as np

def summarize_estimates(
    estimates, 
    true_value,
    sample_size,
):
    mean_estimate = np.mean(estimates)
    empirical_bias = mean_estimate - true_value
    empirical_variance = np.mean((estimates - mean_estimate)**2)
    empirical_mse = np.mean((estimates - true_value)**2)
    empirical_std = np.sqrt(empirical_variance)
    scaled_variance = sample_size * empirical_variance

    assert np.isclose(empirical_mse, empirical_bias**2 + empirical_variance)

    return {
        "mean": mean_estimate,
        "bias": empirical_bias,
        "variance": empirical_variance,
        "mse": empirical_mse,
        "std": empirical_std,
        "scaled_variance": scaled_variance,
        "min": estimates.min(),
        "max": estimates.max(),
    }