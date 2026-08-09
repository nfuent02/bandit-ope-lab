import numpy as np
import pandas as pd

from ope import dgp, estimators, theory

BASE_SEED = 42

mu_1 = dgp.true_mu.copy()
mu_2 = np.array([[0.2,0.3],[0.9,0.4]])

context_distribution = dgp.context_probabilities.copy()
target_policy = dgp.target_policy.copy()
logging_policy = dgp.make_logging_policy(0)

w1_target_value = theory.exact_policy_value(context_distribution, target_policy, mu_1)
w2_target_value = theory.exact_policy_value(context_distribution, target_policy, mu_2)

logged_sample = dgp.generate_logged_data(1000, logging_policy, BASE_SEED)

assert np.all((xi := point.get("xi")) in (0, 1) and xi == point.get("ai") for point in logged_sample)

ips_value = estimators.ips_policy_value(logged_sample)
snips_value = estimators.snips_policy_value(logged_sample)

results = pd.DataFrame(
    [
        {
            "world": "World 1",
            "true_target_value": w1_target_value,
            "ips": ips_value,
            "snips": snips_value,
        },
        {
            "world": "World 2",
            "true_target_value": w2_target_value,
            "ips": ips_value,
            "snips": snips_value,
        },
    ]
)

results.to_csv(
    "results/tables/positivity_violation.csv",
    index=False,
)

print("True target policy value in World 1:", w1_target_value)
print("True target policy value in World 2:", w2_target_value)

print("IPS policy value:", ips_value)
print("SNIPS policy value:", snips_value)

