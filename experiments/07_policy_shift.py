import numpy as np
import pandas as pd

from ope import dgp, estimators, metrics, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZE = 200
BASE_SEED = 42
DELTA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def make_target_policy(delta):

    logging_policy = dgp.logging_policy
    target_policy = dgp.target_policy

    return (1-delta)*logging_policy + delta*target_policy

def sanity_checks():

    for delta in DELTA_LEVELS:
        assert np.isclose(
            theory.exact_policy_value(
                dgp.context_probabilities, 
                make_target_policy(delta),
                dgp.true_mu
            ), 
            0.435 + 0.225*delta)


def compute_monte_carlo_values(
    number_of_replications, 
    sample_size, 
    logging_policy, 
    reward_model, 
    base_seed,
    delta_levels
):

    estimator_functions = {
        "DM": lambda data, delta: estimators.dm_policy_value(
            data,
            target_policy=make_target_policy(delta),
            reward_model=reward_model,
        ),
        "IPS": lambda data, delta: estimators.ips_policy_value(
            data,
            target_policy=make_target_policy(delta),
            estimated_logging_policy=logging_policy,
        ),
        "SNIPS": lambda data, delta: estimators.snips_policy_value(
            data,
            target_policy=make_target_policy(delta),
            estimated_logging_policy=logging_policy,
        ),
        "DR": lambda data, delta: estimators.dr_policy_value(
            data,
            reward_model=reward_model,
            target_policy=make_target_policy(delta),
            estimated_logging_policy=logging_policy,
        ),
    }

    estimates = {delta: {
        estimator_name: np.zeros(number_of_replications)
        for estimator_name in estimator_functions
    } for delta in delta_levels}

    for repetition in range(number_of_replications):
        data = dgp.generate_logged_data(
            sample_size,
            logging_policy,
            seed=base_seed + repetition,
        )
        for delta in delta_levels:
            for estimator_name, estimator in estimator_functions.items():
                estimates[delta][estimator_name][repetition] = estimator(data, delta)


    true_values = {delta: theory.exact_policy_value(dgp.context_probabilities, make_target_policy(delta), dgp.true_mu) for delta in delta_levels}

    return estimates, true_values


def monte_carlo_statistics(
    estimator_values, 
    true_values, 
    sample_size,
    delta_levels
):
    statistics = {delta: {} for delta in delta_levels}

    for delta in delta_levels:
        for estimator_name, values in estimator_values[delta].items():
            statistics[delta][estimator_name] = metrics.summarize_estimates(
                values,
                true_values[delta],
                sample_size,
            )

    return statistics


def main():

    sanity_checks()

    estimates, true_values = compute_monte_carlo_values(
        NUMBER_OF_REPLICATIONS,
        SAMPLE_SIZE,
        dgp.logging_policy,
        dgp.true_mu,
        BASE_SEED,
        DELTA_LEVELS,
    )

    statistics = monte_carlo_statistics(
        estimates,
        true_values,
        SAMPLE_SIZE,
        DELTA_LEVELS,
    )

    rows = []

    for delta in DELTA_LEVELS:
        for estimator_name, stats in statistics[delta].items():
            rows.append({
                "delta": delta,
                "estimator": estimator_name,
                "true_value": true_values[delta],
                **stats,
            })

    df = pd.DataFrame(rows)

    TABLE_WIDTH = 80

    print("\n" + "=" * TABLE_WIDTH)
    print("POLICY SHIFT EXPERIMENT".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)
    
    print(f"\nMonte Carlo replications: {NUMBER_OF_REPLICATIONS}")
    print(f"True policy values: {[round(float(v),4) for v in true_values.values()]}")
    print(f"Base seed: {BASE_SEED}")
    
    print("\n" + "-" * TABLE_WIDTH)

    print(
        f"{'delta':>8} "
        f"{'true value':>12} "
        f"{'estimator':>10} "
        f"{'mean':>8} "
        f"{'bias':>10} "
        f"{'variance':>14} "
        f"{'mse':>8} "
    )

    print("-" * TABLE_WIDTH)

    for delta in DELTA_LEVELS:
        for estimator_name in statistics[delta]:
            print(
                f"{delta:>8.2f} "
                f"{true_values[delta]:>12.6f} "
                f"{estimator_name:>8} "
                f"{statistics[delta][estimator_name]['mean']:>12.6f} "
                f"{statistics[delta][estimator_name]['bias']:>10.6f} "
                f"{statistics[delta][estimator_name]['variance']:>12.6f} "
                f"{statistics[delta][estimator_name]['mse']:>10.6f} "
            )

        print("-" * TABLE_WIDTH)

    df.to_csv(
        "results/tables/policy_shift.csv",
        index=False,
    )

if __name__ == "__main__":
    main()