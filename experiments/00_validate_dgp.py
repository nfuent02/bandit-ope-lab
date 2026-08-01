from email import policy

import numpy as np
from sympy import re

n = 200000
seed = 42

p_1 = 0.5
true_mu = np.array([[0.2,0.8],[0.7,0.4]])
b = np.array([[0.7,0.3],[0.3,0.7]])
pi = np.array([[0.2,0.8],[0.8,0.2]])
V_b = 0.435
V_pi = 0.66

def reward_mean(x, a):
    return true_mu[x, a]

def logging_prob(x, a):
    return b[x, a]

def target_prob(x, a):
    return pi[x, a]

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
        xi, ai, ri, mui, logprob, targetprob = generate_data_point(policy, rng)
        rewards.append(ri)
    
    return np.mean(rewards)

def exact_policy_value(policy):
    value = 0.0
    for x in (0, 1):
        probability_x = p_1 if x == 1 else (1 - p_1)
        for a in (0, 1):
            value += policy[x, a] * true_mu[x, a] * probability_x
    return value

def ips_policy_value(data):
    r = data[:, 2]
    mu = data[:, 3]
    propensity = data[:, 4]
    target_action_prob = data[:, 5]

    importance_weights = target_action_prob / propensity

    return np.mean(importance_weights * r)


# Las políticas deben ser distribuciones de probabilidad válidas.
assert np.allclose(b.sum(axis=1), 1)
assert np.allclose(pi.sum(axis=1), 1)

# Los parámetros Bernoulli deben estar entre 0 y 1.
assert np.all((true_mu >= 0) & (true_mu <= 1))

# El cálculo programado debe coincidir con el cálculo manual.
assert np.isclose(exact_policy_value(b), V_b)
assert np.isclose(exact_policy_value(pi), V_pi)

# La misma semilla debe reproducir exactamente el dataset.
data_1 = generate_logged_data(1000, b, seed=42)
data_2 = generate_logged_data(1000, b, seed=42)

assert np.array_equal(data_1, data_2)

data = generate_logged_data(n, b, seed)


x = data[:, 0].astype(int)
a = data[:, 1].astype(int)
r = data[:, 2]
mu = data[:, 3]
propensity = data[:, 4]
target_action_prob = data[:, 5]
importance_weights = target_action_prob / propensity

assert np.allclose(propensity, b[x, a])
assert np.allclose(target_action_prob, pi[x, a])


""" print("P(X=1):", np.mean(x == 1))
print("P(A=1 | X=0):", np.mean(a[x == 0] == 1))
print("P(A=1 | X=1):", np.mean(a[x == 1] == 1))

for x_value in (0, 1):
    for a_value in (0, 1):
        mask = (x == x_value) & (a == a_value)

        print(
            x_value,
            a_value,
            mask.sum(),
            r[mask].mean(),
            true_mu[x_value, a_value],
        )

print("Exact V(b):", exact_policy_value(b))
print("Direct simulation V(b):", simulate_policy_value(n, b, seed=43))

print("Exact V(pi):", exact_policy_value(pi))
print("Direct simulation V(pi):", simulate_policy_value(n, pi, seed=44))
print("IPS estimate:", ips_policy_value(data))

print("Minimum weight:", importance_weights.min())
print("Maximum weight:", importance_weights.max())
print("Mean weight:", importance_weights.mean())
 """

def run_ips_monte_carlo(number_of_replications, sample_size, base_seed):
    ips_estimates = np.zeros(number_of_replications)
    for repetition in range(number_of_replications):
        repeated_data = generate_logged_data(
            sample_size,
            b,
            seed=base_seed + repetition,
        )

        ips_estimates[repetition] = ips_policy_value(repeated_data)

    return ips_estimates

def print_monte_carlo_statistics(number_of_replications, sample_size, seed):
    # Run many independent IPS experiments
    ips_estimates = run_ips_monte_carlo(number_of_replications, sample_size, seed)
    true_value = exact_policy_value(pi)

    # Calculate mean estimate
    monte_carlo_mean = np.mean(ips_estimates)

    # Calculate empirical bias
    empirical_bias = monte_carlo_mean - true_value

    # Calculate empirical variance
    empirical_variance = np.mean((ips_estimates - monte_carlo_mean)**2)

    # Calculate empirical MSE
    empirical_mse = np.mean((ips_estimates - true_value)**2)

    print("\n ---- Estadísticas de Monte Carlo ----")
    print("Número de repeticiones: ", number_of_replications)
    print("Tamaño de cada muestra: ", sample_size)
    print("Valor verdadero: ", true_value)

    print("Media IPS: ", monte_carlo_mean)
    print("Sesgo empírico: ", empirical_bias)
    print("Varianza empírica: ", empirical_variance)
    print("MSE empírico: ", empirical_mse)
    print("Mínimo de estimaciones: ", ips_estimates.min())
    print("Máximo de estimaciones: ", ips_estimates.max())

    empirical_std = np.sqrt(empirical_variance)
    scaled_variance = sample_size * empirical_variance

    print("Desviación típica:", empirical_std)
    print("n × varianza:", scaled_variance)

    assert np.isclose(empirical_mse, empirical_bias**2 + empirical_variance)

    return monte_carlo_mean, empirical_bias, empirical_variance, empirical_mse, ips_estimates


""" for sample_size in (50,200,1000,5000):
    print_monte_carlo_statistics(2000, sample_size, seed + sample_size)

 """

def snips_policy_value(data):
    r = data[:, 2]
    propensity = data[:, 4]
    target_action_prob = data[:, 5]

    importance_weights = target_action_prob / propensity
    normalized_weights = importance_weights / np.mean(importance_weights)

    return np.mean(normalized_weights * r)


def print_snips_monte_carlo_statistics(number_of_replications, sample_size, seed):
    # Run many independent SNIPS experiments
    snips_estimates = np.zeros(number_of_replications)
    for repetition in range(number_of_replications):
        repeated_data = generate_logged_data(
            sample_size,
            b,
            seed=seed + repetition,
        )

        snips_estimates[repetition] = snips_policy_value(repeated_data)

    true_value = exact_policy_value(pi)

    # Calculate mean estimate
    monte_carlo_mean = np.mean(snips_estimates)

    # Calculate empirical bias
    empirical_bias = monte_carlo_mean - true_value

    # Calculate empirical variance
    empirical_variance = np.mean((snips_estimates - monte_carlo_mean)**2)

    # Calculate empirical MSE
    empirical_mse = np.mean((snips_estimates - true_value)**2)

    print("\n ---- Estadísticas de Monte Carlo (SNIPS) ----")
    print("Número de repeticiones: ", number_of_replications)
    print("Tamaño de cada muestra: ", sample_size)
    print("Valor verdadero: ", true_value)

    print("Media SNIPS: ", monte_carlo_mean)
    print("Sesgo empírico: ", empirical_bias)
    print("Varianza empírica: ", empirical_variance)
    print("MSE empírico: ", empirical_mse)
    print("Mínimo de estimaciones: ", snips_estimates.min())
    print("Máximo de estimaciones: ", snips_estimates.max())

    empirical_std = np.sqrt(empirical_variance)
    scaled_variance = sample_size * empirical_variance

    print("Desviación típica:", empirical_std)
    print("n × varianza:", scaled_variance)

    assert np.isclose(empirical_mse, empirical_bias**2 + empirical_variance)



""" for sample_size in (50,200,1000,5000):
    print_snips_monte_carlo_statistics(2000, sample_size, seed + sample_size)
 """


def compare_ips_snips_monte_carlo(number_of_replications, sample_size, policy, base_seed):
    ips_values = np.zeros(number_of_replications)
    snips_values = np.zeros(number_of_replications)
    true_value = exact_policy_value(pi)

    for i in range(number_of_replications):
        data = generate_logged_data(sample_size, policy, base_seed + i)
        ips_values[i] = ips_policy_value(data)
        snips_values[i] = snips_policy_value(data)

    print("\n--- IPS stats ---")
    print("Mean: ", np.mean(ips_values))
    print("Bias: ", np.mean(ips_values) - true_value)
    print("Variance: ", np.mean((ips_values - np.mean(ips_values))**2))
    print("MSE: ", np.mean((ips_values - true_value)**2))
    print("Min: ", ips_values.min())
    print("Max: ", ips_values.max())

    print("\n--- SNIPS stats ---")
    print("Mean: ", np.mean(snips_values))
    print("Bias: ", np.mean(snips_values) - true_value)
    print("Variance: ", np.mean((snips_values - np.mean(snips_values))**2))
    print("MSE: ", np.mean((snips_values - true_value)**2))
    print("Min: ", snips_values.min())
    print("Max: ", snips_values.max())

""" 
compare_ips_snips_monte_carlo(2000,200, b,seed)
 """

def compare_overlap_experiment(epsilon_values,number_of_replications, sample_size, base_seed):

    for i, epsilon in enumerate(epsilon_values):
        policy = make_logging_policy(epsilon)
        print("\n\n--- Epsilon: ", epsilon, " ---")
        compare_ips_snips_monte_carlo(number_of_replications, sample_size, policy, base_seed + i)
        print("Max theoretical weight: ", np.max(pi / policy))

""" 
compare_overlap_experiment([0.3, 0.1, 0.05, 0.02, 0.01], 2000, 200, seed)
 """

def exact_ips_variance(logging_policy, sample_size):
    e_psi_2 = 0.0

    for x in (0, 1):
        probability_x = p_1 if x == 1 else 1 - p_1
        for a in (0, 1):
            contribution = probability_x * (pi[x, a]**2 * true_mu[x, a] / logging_policy[x, a])
            e_psi_2 += contribution

    e_psi = exact_policy_value(pi)
    var_psi = e_psi_2 - e_psi**2
    return var_psi / sample_size

""" for epsilon in [0.3, 0.1, 0.05, 0.02, 0.01]:
    logging_policy = make_logging_policy(epsilon)
    print(f"Exact IPS variance for epsilon={epsilon}:", exact_ips_variance(logging_policy, 200))
 """

def dm_policy_value(data, reward_model, target_policy):
    x = data[:, 0].astype(int)

    contributions = np.zeros(len(x))

    for i,xi in enumerate(x):
        contribution = sum(reward_model[xi, action] * target_policy[xi, action] for action in [0, 1])
        contributions[i] = contribution

    return np.mean(contributions)

""" 
oracle_reward_model = true_mu.copy()
constant_reward_model = np.full((2, 2), 0.5)

print(dm_policy_value(data, oracle_reward_model, pi))
print(dm_policy_value(data, constant_reward_model, pi))

 """

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

def compare_dm_ips_snips_dr_monte_carlo(number_of_replications, sample_size, policy, oracle_reward_model, constant_reward_model, base_seed):
    dm_oracle_values = np.zeros(number_of_replications)
    dm_constant_values = np.zeros(number_of_replications)
    ips_values = np.zeros(number_of_replications)
    snips_values = np.zeros(number_of_replications)
    dr_oracle_values = np.zeros(number_of_replications)
    dr_constant_values = np.zeros(number_of_replications)
    true_value = exact_policy_value(pi)

    for i in range(number_of_replications):
        data = generate_logged_data(
            sample_size, 
            policy, 
            base_seed + i)
        
        dm_oracle_values[i] = dm_policy_value(
            data, 
            oracle_reward_model, 
            pi)
        
        dm_constant_values[i] = dm_policy_value(
            data, 
            constant_reward_model, 
            pi)
        
        ips_values[i] = ips_policy_value(data)
        snips_values[i] = snips_policy_value(data)
        dr_oracle_values[i] = dr_policy_value(
            data, 
            oracle_reward_model, 
            pi)
        dr_constant_values[i] = dr_policy_value(
            data, 
            constant_reward_model, 
            pi)

    print("\n--- DM oracle stats ---")
    print("Mean: ", np.mean(dm_oracle_values))
    print("Bias: ", np.mean(dm_oracle_values) - true_value)
    print("Variance: ", np.mean((dm_oracle_values - np.mean(dm_oracle_values))**2))
    print("MSE: ", np.mean((dm_oracle_values - true_value)**2))
    print("Min: ", dm_oracle_values.min())
    print("Max: ", dm_oracle_values.max())
    
    print("\n--- DM constant stats ---")
    print("Mean: ", np.mean(dm_constant_values))
    print("Bias: ", np.mean(dm_constant_values) - true_value)
    print("Variance: ", np.mean((dm_constant_values - np.mean(dm_constant_values))**2))
    print("MSE: ", np.mean((dm_constant_values - true_value)**2))
    print("Min: ", dm_constant_values.min())
    print("Max: ", dm_constant_values.max())

    print("\n--- IPS stats ---")
    print("Mean: ", np.mean(ips_values))
    print("Bias: ", np.mean(ips_values) - true_value)
    print("Variance: ", np.mean((ips_values - np.mean(ips_values))**2))
    print("MSE: ", np.mean((ips_values - true_value)**2))
    print("Min: ", ips_values.min())
    print("Max: ", ips_values.max())

    print("\n--- SNIPS stats ---")
    print("Mean: ", np.mean(snips_values))
    print("Bias: ", np.mean(snips_values) - true_value)
    print("Variance: ", np.mean((snips_values - np.mean(snips_values))**2))
    print("MSE: ", np.mean((snips_values - true_value)**2))
    print("Min: ", snips_values.min())
    print("Max: ", snips_values.max())

    print("\n--- DR oracle stats ---")
    print("Mean: ", np.mean(dr_oracle_values))
    print("Bias: ", np.mean(dr_oracle_values) - true_value)
    print("Variance: ", np.mean((dr_oracle_values - np.mean(dr_oracle_values))**2))
    print("MSE: ", np.mean((dr_oracle_values - true_value)**2))
    print("Min: ", dr_oracle_values.min())
    print("Max: ", dr_oracle_values.max())

    print("\n--- DR constant stats ---")
    print("Mean: ", np.mean(dr_constant_values))
    print("Bias: ", np.mean(dr_constant_values) - true_value)
    print("Variance: ", np.mean((dr_constant_values - np.mean(dr_constant_values))**2))
    print("MSE: ", np.mean((dr_constant_values - true_value)**2))
    print("Min: ", dr_constant_values.min())
    print("Max: ", dr_constant_values.max())
    

""" 
compare_dm_ips_snips_dr_monte_carlo(2000, 200, b, true_mu, np.full((2, 2), 0.5), seed)
 """

def exact_dr_bias(estimated_reward_model, estimated_logging_policy, true_reward_model, true_logging_policy, target_policy):
    bias = 0.0

    for x in (0, 1):
        probability_x = p_1 if x == 1 else 1 - p_1
        for a in (0, 1):
            contribution = probability_x * target_policy[x, a] * (estimated_reward_model[x, a] - true_reward_model[x, a]) * (1 - true_logging_policy[x, a] / estimated_logging_policy[x, a])
            bias += contribution

    return bias

def compare_dr_with_correct_and_wrong_models(number_of_replications, sample_size, target_policy, true_mu, logging_policy, base_seed):

    correct_mu = true_mu.copy()
    wrong_mu = np.full((2, 2), 0.5)

    correct_b_hat = logging_policy.copy()
    wrong_b_hat = np.full((2, 2), 0.5)

    dr_both_correct = np.zeros(number_of_replications)
    dr_both_wrong = np.zeros(number_of_replications)
    dr_mu_wrong = np.zeros(number_of_replications)
    dr_b_wrong = np.zeros(number_of_replications)

    for i in range(number_of_replications):
        data = generate_logged_data(
            sample_size, 
            logging_policy, 
            base_seed + i)
        dr_both_correct[i] = dr_policy_value(
            data, 
            correct_mu, 
            target_policy,
            correct_b_hat)
        dr_both_wrong[i] = dr_policy_value(
            data, 
            wrong_mu, 
            target_policy,
            wrong_b_hat)
        dr_mu_wrong[i] = dr_policy_value(
            data,
            wrong_mu,
            target_policy,
            correct_b_hat)
        dr_b_wrong[i] = dr_policy_value(
            data,
            correct_mu,
            target_policy,
            wrong_b_hat)
        

    print("\n--- model correct, logging policy correct ---")
    print("Mean: ", np.mean(dr_both_correct))
    print("Bias: ", np.mean(dr_both_correct) - exact_policy_value(target_policy))
    print("Variance: ", np.mean((dr_both_correct - np.mean(dr_both_correct))**2))
    print("MSE: ", np.mean((dr_both_correct - exact_policy_value(target_policy))**2))
    print("Exact bias (theoretical): ", exact_dr_bias(correct_mu, correct_b_hat, true_mu, logging_policy, target_policy))

    print("\n--- model wrong, logging policy correct ---")
    print("Mean: ", np.mean(dr_mu_wrong))
    print("Bias: ", np.mean(dr_mu_wrong) - exact_policy_value(target_policy))
    print("Variance: ", np.mean((dr_mu_wrong - np.mean(dr_mu_wrong))**2))
    print("MSE: ", np.mean((dr_mu_wrong - exact_policy_value(target_policy))**2))
    print("Exact bias (theoretical): ", exact_dr_bias(wrong_mu, correct_b_hat, true_mu, logging_policy, target_policy))

    print("\n--- model correct, logging policy wrong ---")
    print("Mean: ", np.mean(dr_b_wrong))
    print("Bias: ", np.mean(dr_b_wrong) - exact_policy_value(target_policy))
    print("Variance: ", np.mean((dr_b_wrong - np.mean(dr_b_wrong))**2))
    print("MSE: ", np.mean((dr_b_wrong - exact_policy_value(target_policy))**2))
    print("Exact bias (theoretical): ", exact_dr_bias(correct_mu, wrong_b_hat, true_mu, logging_policy, target_policy))

    print("\n--- model wrong, logging policy wrong ---")
    print("Mean: ", np.mean(dr_both_wrong))
    print("Bias: ", np.mean(dr_both_wrong) - exact_policy_value(target_policy))
    print("Variance: ", np.mean((dr_both_wrong - np.mean(dr_both_wrong))**2))
    print("MSE: ", np.mean((dr_both_wrong - exact_policy_value(target_policy))**2))
    print("Exact bias (theoretical): ", exact_dr_bias(wrong_mu, wrong_b_hat, true_mu, logging_policy, target_policy))

""" 
compare_dr_with_correct_and_wrong_models(2000, 200, target_policy, true_mu, logging_policy, seed)
 """

def compare_dr_with_misspecification(number_of_replications, sample_size, target_policy, true_mu, logging_policy, levels, base_seed):
    correct_mu = true_mu.copy()
    wrong_mu = np.full((2, 2), 0.5)

    correct_b_hat = logging_policy.copy()
    wrong_b_hat = np.full((2, 2), 0.5)

    true_policy_value = exact_policy_value(target_policy)

    results = []

    datasets = [
        generate_logged_data(
            sample_size, 
            logging_policy, 
            base_seed + k) for k in range(number_of_replications)
    ]

    for i, lambda_value in enumerate(levels):
        for j, gamma_value in enumerate(levels):
            interpolated_mu = (1-lambda_value) * correct_mu + lambda_value * wrong_mu
            interpolated_b_hat = (1-gamma_value) * correct_b_hat + gamma_value * wrong_b_hat
            dr_values = np.zeros(number_of_replications)
            for k, data in enumerate(datasets):
                dr_values[k] = dr_policy_value(
                    data,
                    interpolated_mu,
                    target_policy,
                    interpolated_b_hat
                )
            monte_carlo_mean = np.mean(dr_values)
            monte_carlo_bias = monte_carlo_mean - true_policy_value

            exact_bias = exact_dr_bias(
                interpolated_mu, 
                interpolated_b_hat, 
                true_mu, 
                logging_policy, 
                target_policy)

            results.append({
                "lambda": lambda_value,
                "gamma": gamma_value,
                "exact_bias": exact_dr_bias(interpolated_mu, interpolated_b_hat, true_mu, logging_policy, target_policy),
                "monte_carlo_bias":  monte_carlo_mean - exact_policy_value(target_policy),
                "bias_error": monte_carlo_bias - exact_bias,
                "variance": np.mean((dr_values - monte_carlo_mean)**2),
                "mse": np.mean((dr_values - exact_policy_value(target_policy))**2)
            })

    return results


def print_misspecification_results(results):

    assert np.isclose(
        exact_dr_bias(
            true_mu,
            b,
            true_mu,
            b,
            pi,
        ),
        0.0,
    )

    assert np.isclose(
        exact_dr_bias(
            np.full((2, 2), 0.5),
            np.full((2, 2), 0.5),
            true_mu,
            b,
            pi,
        ),
        -0.096,
    )

    print("\n--- DR misspecification results ---")
    for result in results:
        print(f"\nlambda: {result['lambda']}, \ngamma: {result['gamma']}, \nexact_bias: {result['exact_bias']}, \nmonte_carlo_bias: {result['monte_carlo_bias']}, \nvariance: {result['variance']}, \nmse: {result['mse']}")

print_misspecification_results(compare_dr_with_misspecification(2000, 200, pi, true_mu, b, [0.0, 0.25, 0.5, 0.75, 1.0], seed))