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
    return xi, ai, ri, mui, logging_prob(xi, ai), target_prob(xi, ai)

def generate_logged_data(n, seed):
    rng = np.random.default_rng(seed)  # Set a random seed for reproducibility
    data = []
    for _ in range(n):
        xi, ai, ri, mui, logprob, targetprob = generate_data_point(b, rng)
        data.append((xi, ai, ri, mui, logprob, targetprob))
    return np.array(data)

def simulate_policy_value(n,policy,seed):
    rng = np.random.default_rng(seed)  # Set a random seed for reproducibility
    rewards = []

    for _ in range(n):
        xi, ai, ri, mui, logprob, targetprob = generate_data_point(policy, rng)
        rewards.append(ri)
    
    return np.mean(rewards)




data = generate_logged_data(n, seed)

x = data[:, 0].astype(int)
a = data[:, 1].astype(int)
r = data[:, 2]
mu = data[:, 3]
propensity = data[:, 4]
target_action_prob = data[:, 5]



assert np.allclose(b.sum(axis=1), 1)
assert np.allclose(pi.sum(axis=1), 1)
assert np.all((true_mu >= 0) & (true_mu <= 1))

data_1 = generate_logged_data(1000, seed=42)
data_2 = generate_logged_data(1000, seed=42)

assert np.array_equal(data_1, data_2)

assert np.allclose(propensity, b[x, a])
assert np.allclose(target_action_prob, pi[x, a])



print("P(X=1):", np.mean(x == 1))
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

print("Exact V(b):", V_b)
print("Simulated V(b):", simulate_policy_value(n, b, seed=43))

print("Exact V(pi):", V_pi)
print("Simulated V(pi):", simulate_policy_value(n, pi, seed=44))









