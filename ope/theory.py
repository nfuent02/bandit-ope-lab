



def exact_policy_value(context_probabilities, logging_policy, mean_reward):
    value = 0.0
    for x in range(len(context_probabilities)):
        for a in range(logging_policy.shape[1]):
            value += context_probabilities[x] * logging_policy[x, a] * mean_reward[x, a]
    return value

def exact_dm_bias(context_probabilities, estimated_reward_model, true_reward_model, target_policy):
    bias = 0.0
    for x in range(len(context_probabilities)):
        probability_x = context_probabilities[x]
        for a in range(target_policy.shape[1]):
            contribution = probability_x * target_policy[x, a] * (estimated_reward_model[x, a] - true_reward_model[x, a])
            bias += contribution
    return bias

def exact_ips_variance(context_probabilities, logging_policy, target_policy, mean_reward, sample_size):
    e_psi_2 = 0.0

    for x in range(len(context_probabilities)):
        probability_x = context_probabilities[x]
        for a in range(target_policy.shape[1]):
            contribution = probability_x * (target_policy[x, a]**2 * mean_reward[x, a] / logging_policy[x, a])
            e_psi_2 += contribution

    e_psi = exact_policy_value(context_probabilities, target_policy, mean_reward)
    var_psi = e_psi_2 - e_psi**2
    return var_psi / sample_size

def exact_dr_bias(context_probabilities, estimated_reward_model, estimated_logging_policy, true_reward_model, true_logging_policy, target_policy):
    bias = 0.0

    for x in range(len(context_probabilities)):
        probability_x = context_probabilities[x]
        for a in range(target_policy.shape[1]):
            contribution = probability_x * target_policy[x, a] * (estimated_reward_model[x, a] - true_reward_model[x, a]) * (1 - true_logging_policy[x, a] / estimated_logging_policy[x, a])
            bias += contribution

    return bias