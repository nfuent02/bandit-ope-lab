
import numpy as np


def fit_reward_model(data):
    
    """ Estimate E[R | X=x, A=a] for every context-action cell """
    
    # Initialize an array to hold the estimated mean rewards
    estimated_mu = np.zeros((2, 2))
    
    # Count occurrences and sum rewards for each context-action pair
    counts = np.zeros((2, 2))
    
    for observation in data:
        xi = observation["xi"]
        ai = observation["ai"]
        ri = observation["ri"]
        estimated_mu[xi, ai] += ri
        counts[xi, ai] += 1
    
    # Avoid division by zero by using np.where
    estimated_mu = np.where(counts > 0, estimated_mu / counts, 0)
    
    return estimated_mu

def fit_logging_policy(data):

    """ Estimate P(A=a | X=x) for every context-action cell """

    # Initialize an array to hold the estimated logging policy probabilities
    estimated_b_hat = np.zeros((2, 2))
    context_counts = np.zeros(2)  # Count occurrences of each context
    
    for observation in data:
        xi = observation["xi"]
        ai = observation["ai"]
        estimated_b_hat[xi, ai] += 1
        context_counts[xi] += 1

    # Normalize to get probabilities
    estimated_b_hat = np.where(context_counts[:, np.newaxis] > 0, estimated_b_hat / context_counts[:, np.newaxis], 0)

    return estimated_b_hat

def check_probabilities(probabilities):

    """ Check if the given probabilities are valid (i.e., they sum to 1 for each context) """

    row_sums = np.sum(probabilities, axis=1)

    assert np.allclose(row_sums, 1), "Each row of the probabilities must sum to 1."
    assert np.all(probabilities >= 0), "Probabilities must be non-negative."
    assert np.all(probabilities <= 1), "Probabilities must be less than or equal to 1."
