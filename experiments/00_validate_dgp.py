
import numpy as np
from ope import (
    dgp,
    estimators,
    theory,
)

SAMPLE_SIZE = 200_000
SEED = 42



def validate_model_parameters(
    logging_policy, 
    target_policy, 
    mean_rewards
):
    
    assert np.allclose(
        logging_policy.sum(axis=1), 
        1), "Logging policy rows must sum to 1"

    assert np.all(
        (logging_policy >= 0.0) & (logging_policy <= 1)), "Logging policy must be between 0 and 1"
    
    assert np.allclose(
        target_policy.sum(axis=1), 
        1), "Target policy rows must sum to 1"
    
    assert np.all(
        (target_policy >= 0.0) & (target_policy <= 1)), "Target policy must be between 0 and 1"

    assert np.all(
        (mean_rewards >= 0.0) & (mean_rewards <= 1)
        ), "Mean reward must be between 0 and 1"

def validate_exact_policy_values(
    logging_policy, 
    target_policy, 
    logging_policy_value, 
    target_policy_value
):

    exact_logging_policy_value = theory.exact_policy_value(dgp.context_probabilities, logging_policy, dgp.true_mu)
    exact_target_policy_value = theory.exact_policy_value(dgp.context_probabilities, target_policy, dgp.true_mu)
    
    assert np.isclose(
        exact_logging_policy_value, 
        logging_policy_value
        ), "Exact logging policy value does not match provided value"
    
    assert np.isclose(
        exact_target_policy_value, 
        target_policy_value
        ), "Exact target policy value does not match provided value"

def validate_reproducibility():

    data_1 = dgp.generate_logged_data(1000, dgp.logging_policy, seed=SEED)
    data_2 = dgp.generate_logged_data(1000, dgp.logging_policy, seed=SEED)

    assert np.array_equal(
        data_1, 
        data_2
        ), "Data generation is not reproducible with the same seed"

def validate_logged_probabilities(
    x, 
    a, 
    propensity, 
    target_action_prob, 
    logging_policy, 
    target_policy
):

    assert np.allclose(
        propensity, 
        logging_policy[x, a]
        ), "Logged propensities do not match logging policy"
    
    assert np.allclose(
        target_action_prob, 
        target_policy[x, a]
        ), "Logged target action probabilities do not match target policy"


def other_validations(
    x, 
    a, 
    mu
):

    assert np.allclose(
        mu,
        dgp.true_mu[x, a],
    ), "Stored mean rewards are incorrect"

    assert np.all(
        dgp.logging_policy[dgp.target_policy > 0] > 0
    ), "Overlap condition violated"


def main():

    data = dgp.generate_logged_data(SAMPLE_SIZE, dgp.logging_policy, seed=SEED)

    x = data[:, 0].astype(int)
    a = data[:, 1].astype(int)
    r = data[:, 2]
    mu = data[:, 3]
    propensity = data[:, 4]
    target_action_prob = data[:, 5]
    importance_weights = target_action_prob / propensity

    EXPECTED_LOGGING_POLICY_VALUE = 0.435
    EXPECTED_TARGET_POLICY_VALUE = 0.66

    exact_logging_policy_value = theory.exact_policy_value(
        dgp.context_probabilities,
        dgp.logging_policy,
        dgp.true_mu,
    )

    simulated_logging_policy_value = dgp.simulate_policy_value(
        SAMPLE_SIZE,
        dgp.logging_policy,
        seed=SEED + 1,
    )

    exact_target_policy_value = theory.exact_policy_value(
        dgp.context_probabilities,
        dgp.target_policy,
        dgp.true_mu,
    )

    simulated_target_policy_value = dgp.simulate_policy_value(
        SAMPLE_SIZE,
        dgp.target_policy,
        seed=SEED + 2,
    )

    ips_target_policy_value = estimators.ips_policy_value(data)

    validate_model_parameters(
        dgp.logging_policy,
        dgp.target_policy,
        dgp.true_mu
    )

    validate_exact_policy_values(
        dgp.logging_policy,
        dgp.target_policy,
        EXPECTED_LOGGING_POLICY_VALUE,
        EXPECTED_TARGET_POLICY_VALUE,
    )

    validate_reproducibility()

    validate_logged_probabilities(
        x,
        a, 
        propensity,
        target_action_prob,
        dgp.logging_policy,
        dgp.target_policy
    )

    other_validations(
        x,
        a,
        mu
    )


    print("\n" + "=" * 60)
    print("DGP VALIDATION".center(60))
    print("=" * 60)

    print("\n[1] Context distribution\n")
    print(f"Empirical P( X = 1 ): {np.mean(x == 1):.4f}")
    print(f"Expected  P( X = 1 ): {dgp.context_probabilities[1]:.4f}")

    print("\n\n[2] Logging policy\n")
    for x_value in (0, 1):
        empirical = np.mean(a[x == x_value] == 1)
        expected = dgp.logging_policy[x_value, 1]

        print(
            f"X = {x_value} | "
            f"empirical P( A = 1 | X = {x_value} ): {empirical:.4f} | "
            f"expected: {expected:.4f}"
        )

    print("\n\n[3] Conditional reward means\n")
    print(f"{'X':>3} {'A':>3} {'N':>6} {'Empirical':>14} {'Expected':>12}")

    for x_value in (0, 1):
        for a_value in (0, 1):
            mask = (x == x_value) & (a == a_value)

            print(
                f"{x_value:>3} "
                f"{a_value:>3} "
                f"{mask.sum():>8} "
                f"{r[mask].mean():>12.4f} "
                f"{dgp.true_mu[x_value, a_value]:>12.4f}"
            )


    print("\n\n[4] Policy values\n")
    print(f"{'Method':<20} {'Estimate':>22} {'Error':>10}")
    print("-" * 58)
    print(
        f"{'Exact logging value':<30} "
        f"{exact_logging_policy_value:>12.6f} "
        f"{0.0:>12.6f}"
    )

    print(
        f"{'Simulated logging value':<30} "
        f"{simulated_logging_policy_value:>12.6f} "
        f"{simulated_logging_policy_value - exact_logging_policy_value:>12.6f}"
    )

    print(
        f"{'Exact target value':<30} "
        f"{exact_target_policy_value:>12.6f} "
        f"{0.0:>12.6f}"
    )

    print(
        f"{'Simulated target value':<30} "
        f"{simulated_target_policy_value:>12.6f} "
        f"{simulated_target_policy_value - exact_target_policy_value:>12.6f}"
    )

    print(
        f"{'IPS target estimate':<30} "
        f"{ips_target_policy_value:>12.6f} "
        f"{ips_target_policy_value - exact_target_policy_value:>12.6f}"
    )

    print("\n\n[5] Importance weights\n")
    print(f"Minimum: {importance_weights.min():.4f}")
    print(f"Maximum: {importance_weights.max():.4f}")
    print(f"Mean:    {importance_weights.mean():.4f}")

    print(
        "Quantiles:",
        np.quantile(
            importance_weights,
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ),
    )


if __name__ == "__main__":
    main()
    