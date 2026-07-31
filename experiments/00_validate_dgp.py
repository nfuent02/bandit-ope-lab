import numpy as np

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
    e_psi_2 = p_1 * (pi[0, 0]**2 * true_mu[0, 0]/logging_policy[0, 0] + pi[0, 1]**2 * true_mu[0, 1]/logging_policy[0, 1]) + (1-p_1) * (pi[1, 0]**2 * true_mu[1, 0]/logging_policy[1, 0] + pi[1, 1]**2 * true_mu[1, 1]/logging_policy[1, 1])
    e_psi = exact_policy_value(pi)
    var_psi = e_psi_2 - e_psi**2
    return var_psi / sample_size

for epsilon in [0.3, 0.1, 0.05, 0.02, 0.01]:
    logging_policy = make_logging_policy(epsilon)
    print(f"Exact IPS variance for epsilon={epsilon}:", exact_ips_variance(logging_policy, 200))
