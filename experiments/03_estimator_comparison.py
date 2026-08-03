
import numpy as np

from ope import dgp, estimators, metrics, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZE = 200
BASE_SEED = 42

oracle_reward_model = dgp.true_mu.copy()
constant_reward_model = np.full_like(dgp.true_mu, 0.5)


def compute_monte_carlo_values(
    number_of_replications, 
    sample_size, 
    logging_policy, 
    target_policy,
    oracle_reward_model, 
    constant_reward_model, 
    base_seed
):

    estimator_functions = {
        "DM oracle": lambda data: estimators.dm_policy_value(
            data,
            target_policy,
            oracle_reward_model,
        ),
        "DM constant": lambda data: estimators.dm_policy_value(
            data,
            target_policy,
            constant_reward_model,
        ),
        "IPS": estimators.ips_policy_value,
        "SNIPS": estimators.snips_policy_value,
        "DR oracle": lambda data: estimators.dr_policy_value(
            data,
            oracle_reward_model,
            target_policy,
            logging_policy,
        ),
        "DR constant": lambda data: estimators.dr_policy_value(
            data,
            constant_reward_model,
            target_policy,
            logging_policy,
        ),
    }

    estimates = {
        estimator_name: np.zeros(number_of_replications)
        for estimator_name in estimator_functions
    }

    for repetition in range(number_of_replications):
        data = dgp.generate_logged_data(
            sample_size,
            logging_policy,
            seed=base_seed + repetition,
        )

        for estimator_name, estimator in estimator_functions.items():
            estimates[estimator_name][repetition] = estimator(data)


    true_value = theory.exact_policy_value(dgp.context_probabilities, target_policy, dgp.true_mu)

    return estimates, true_value


def monte_carlo_statistics(
    estimator_values, 
    true_value, 
    sample_size
):
    statistics = {}

    for estimator_name, values in estimator_values.items():
        statistics[estimator_name] = metrics.summarize_estimates(
            values,
            true_value,
            sample_size,
        )

    return statistics
    

def main():

    TABLE_WIDTH = 96

    values, true_value = compute_monte_carlo_values(
        number_of_replications=NUMBER_OF_REPLICATIONS,
        sample_size=SAMPLE_SIZE,
        logging_policy=dgp.logging_policy,
        target_policy=dgp.target_policy,
        oracle_reward_model=oracle_reward_model,
        constant_reward_model=constant_reward_model,
        base_seed=BASE_SEED,
    )

    statistics = monte_carlo_statistics(
        values,
        true_value,
        SAMPLE_SIZE,
    )


    print("\n" + "=" * TABLE_WIDTH)
    print("ESTIMATOR COMPARISON".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)

    print(f"\nMonte Carlo replications: {NUMBER_OF_REPLICATIONS}")
    print(f"True policy value: {true_value:.6f}")
    print(f"Base seed: {BASE_SEED}")

    print("\n" + "-" * TABLE_WIDTH)

    print(
        f"{'Estimator':>12} "
        f"{'Mean':>10} "
        f"{'Bias':>11} "
        f"{'Variance':>12} "
        f"{'Std':>10} "
        f"{'MSE':>12} "
        f"{'Min':>10} "
        f"{'Max':>10}"
    )
    print("-" * TABLE_WIDTH)

    for estimator_name, stats in statistics.items():
        print(
            f"{estimator_name:<12} "
            f"{stats['mean']:>12.6f} "
            f"{stats['bias']:>+11.6f} "
            f"{stats['variance']:>10.6f} "
            f"{stats['std']:>12.6f} "
            f"{stats['mse']:>12.6f} "
            f"{stats['min']:>10.6f} "
            f"{stats['max']:>10.6f}"
        )

if __name__ == "__main__":
    main()


