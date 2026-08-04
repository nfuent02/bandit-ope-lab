
import numpy as np


def dm_policy_value(data, target_policy, reward_model):
    x = [observation["xi"] for observation in data]

    contributions = np.zeros(len(x))

    for i,xi in enumerate(x):
        contribution = sum(reward_model[xi, action] * target_policy[xi, action] for action in [0, 1])
        contributions[i] = contribution

    return np.mean(contributions)

def ips_policy_value(data):
    r = [observation["ri"] for observation in data]
    propensity = [observation["logprob"] for observation in data]
    target_action_prob = [observation["targetprob"] for observation in data]
    importance_weights = np.array(target_action_prob) / np.array(propensity)

    return np.mean(importance_weights * r)

def snips_policy_value(data):
    r = [observation["ri"] for observation in data]
    propensity = [observation["logprob"] for observation in data]
    target_action_prob = [observation["targetprob"] for observation in data]

    importance_weights = np.array(target_action_prob) / np.array(propensity)
    normalized_weights = importance_weights / np.mean(importance_weights)

    return np.mean(normalized_weights * r)

def dr_policy_value(data, reward_model, target_policy, estimated_logging_policy=None):
    x = [observation["xi"] for observation in data]
    a = [observation["ai"] for observation in data]
    r = [observation["ri"] for observation in data]
    propensity = [observation["logprob"] for observation in data]
    importance_weights = target_policy[x, a] / estimated_logging_policy[x, a] if estimated_logging_policy is not None else target_policy[x, a] / propensity

    contributions = np.zeros(len(x))

    for i, xi in enumerate(x):
        baseline = sum(reward_model[xi, action] * target_policy[xi, action] for action in [0, 1])

        residual_correction = importance_weights[i] * (r[i] - reward_model[xi, a[i]])

        contributions[i] = baseline + residual_correction

    return np.mean(contributions)

