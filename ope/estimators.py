
import numpy as np


def dm_policy_value(data, target_policy, reward_model):
    x = data[:, 0].astype(int)

    contributions = np.zeros(len(x))

    for i,xi in enumerate(x):
        contribution = sum(reward_model[xi, action] * target_policy[xi, action] for action in [0, 1])
        contributions[i] = contribution

    return np.mean(contributions)

def ips_policy_value(data):
    r = data[:, 2]
    
    propensity = data[:, 4]
    target_action_prob = data[:, 5]
    importance_weights = target_action_prob / propensity

    return np.mean(importance_weights * r)

def snips_policy_value(data):
    r = data[:, 2]
    propensity = data[:, 4]
    target_action_prob = data[:, 5]

    importance_weights = target_action_prob / propensity
    normalized_weights = importance_weights / np.mean(importance_weights)

    return np.mean(normalized_weights * r)

def dr_policy_value(data, reward_model, target_policy, estimated_logging_policy=None):
    x = data[:, 0].astype(int)
    a = data[:, 1].astype(int)
    r = data[:, 2]
    propensity = data[:, 4]
    importance_weights = target_policy[x, a] / estimated_logging_policy[x, a] if estimated_logging_policy is not None else target_policy[x, a] / propensity

    contributions = np.zeros(len(x))

    for i, xi in enumerate(x):
        baseline = sum(reward_model[xi, action] * target_policy[xi, action] for action in [0, 1])

        residual_correction = importance_weights[i] * (r[i] - reward_model[xi, a[i]])

        contributions[i] = baseline + residual_correction

    return np.mean(contributions)

