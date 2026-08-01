
import numpy as np

p_1 = 0.5
context_probabilities = np.array([1 - p_1, p_1])
true_mu = np.array([[0.2,0.8],[0.7,0.4]])
logging_policy = np.array([[0.7,0.3],[0.3,0.7]])
target_policy = np.array([[0.2,0.8],[0.8,0.2]])
logging_policy_value = 0.435
target_policy_value = 0.66

def reward_mean(x, a):
    return true_mu[x, a]

def logging_prob(x, a):
    return logging_policy[x, a]

def target_prob(x, a):
    return target_policy[x, a]

def generate_data_point(policy, rng):
    xi = rng.binomial(1, p_1)
    ai = rng.binomial(1, policy[xi, 1])
    mui = reward_mean(xi, ai)
    ri = rng.binomial(1, mui)
    return xi, ai, ri, mui, policy[xi,ai], target_prob(xi, ai)

def generate_logged_data(n, logging_policy, seed):
    rng = np.random.default_rng(seed)  # Set a random seed for reproducibility
    data = []
    for _ in range(n):
        xi, ai, ri, mui, logprob, targetprob = generate_data_point(logging_policy, rng)
        data.append((xi, ai, ri, mui, logprob, targetprob))
    return np.array(data)

def make_logging_policy(epsilon):
    assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"
    return np.array([[1-epsilon, epsilon],[epsilon, 1-epsilon]])

def simulate_policy_value(n,policy,seed):
    rng = np.random.default_rng(seed)  # Set a random seed for reproducibility
    rewards = []

    for _ in range(n):
        rewards.append(generate_data_point(policy, rng)[2])  # Append the reward (ri) to the rewards list
    
    return np.mean(rewards)