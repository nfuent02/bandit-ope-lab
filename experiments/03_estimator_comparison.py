
import numpy as np
import pandas as pd

from ope import dgp, estimators, metrics, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZE = 200
BASE_SEED = 42

oracle_reward_model = dgp.true_mu.copy()
wrong_reward_model = np.full_like(dgp.true_mu, 0.5)
wrong_logging_policy = np.full_like(dgp.logging_policy, 0.5)


def compute_monte_carlo_values(
    number_of_replications, 
    sample_size, 
    logging_policy, 
    target_policy,
    oracle_reward_model, 
    wrong_reward_model,
    wrong_logging_policy,
    base_seed
):

    estimator_functions = {
        "DM oracle": lambda data: estimators.dm_policy_value(
            data,
            target_policy,
            oracle_reward_model,
        ),
        "DM wrong μ": lambda data: estimators.dm_policy_value(
            data,
            target_policy,
            wrong_reward_model,
        ),
        "IPS correct b": lambda data: estimators.ips_policy_value(
            data,
            logging_policy,
        ),
        "IPS wrong b": lambda data: estimators.ips_policy_value(
            data,
            wrong_logging_policy,
        ),
        "SNIPS correct b": lambda data: estimators.snips_policy_value(
            data,
            logging_policy,
        ),
        "SNIPS wrong b": lambda data: estimators.snips_policy_value(
            data,
            wrong_logging_policy,
        ),
        "DR correct μ, correct b": lambda data: estimators.dr_policy_value(
            data,
            oracle_reward_model,
            target_policy,
            logging_policy,
        ),
        "DR correct μ, wrong b": lambda data: estimators.dr_policy_value(
            data,
            oracle_reward_model,
            target_policy,
            wrong_logging_policy,
        ),
        "DR wrong μ, correct b": lambda data: estimators.dr_policy_value(
            data,
            wrong_reward_model,
            target_policy,
            logging_policy,
        ),
        "DR wrong μ, wrong b": lambda data: estimators.dr_policy_value(
            data,
            wrong_reward_model,
            target_policy,
            wrong_logging_policy,
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

    ESTIMATOR_WIDTH = 26
    NUMBER_WIDTH = 12

    TABLE_WIDTH = ESTIMATOR_WIDTH + 7 * (NUMBER_WIDTH + 1)

    values, true_value = compute_monte_carlo_values(
        number_of_replications=NUMBER_OF_REPLICATIONS,
        sample_size=SAMPLE_SIZE,
        logging_policy=dgp.logging_policy,
        target_policy=dgp.target_policy,
        oracle_reward_model=oracle_reward_model,
        wrong_reward_model=wrong_reward_model,
        wrong_logging_policy=wrong_logging_policy,
        base_seed=BASE_SEED,
    )

    statistics = monte_carlo_statistics(
        values,
        true_value,
        SAMPLE_SIZE,
    )

    df = pd.DataFrame(statistics).T
    df.index.name = "estimator"
    df.to_csv("results/tables/estimator_comparison.csv")

    print("\n" + "=" * TABLE_WIDTH)
    print("ESTIMATOR COMPARISON".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)

    print(f"\nMonte Carlo replications: {NUMBER_OF_REPLICATIONS}")
    print(f"True policy value: {true_value:.6f}")
    print(f"Base seed: {BASE_SEED}")

    print("\n" + "-" * TABLE_WIDTH)

    print(
        f"{'Estimator':<{ESTIMATOR_WIDTH}} "
        f"{'Mean':^{NUMBER_WIDTH}} "
        f"{'Bias':^{NUMBER_WIDTH}} "
        f"{'Variance':>{NUMBER_WIDTH}} "
        f"{'Std':^{NUMBER_WIDTH}} "
        f"{'MSE':^{NUMBER_WIDTH}} "
        f"{'Min':^{NUMBER_WIDTH}} "
        f"{'Max':^{NUMBER_WIDTH}}"
    )

    print("-" * TABLE_WIDTH)

    for estimator_name, stats in statistics.items():
        print(
            f"{estimator_name:<{ESTIMATOR_WIDTH}} "
            f"{stats['mean']:>{NUMBER_WIDTH}.6f} "
            f"{stats['bias']:>+{NUMBER_WIDTH}.6f} "
            f"{stats['variance']:>{NUMBER_WIDTH}.6f} "
            f"{stats['std']:>{NUMBER_WIDTH}.6f} "
            f"{stats['mse']:>{NUMBER_WIDTH}.6f} "
            f"{stats['min']:>{NUMBER_WIDTH}.6f} "
            f"{stats['max']:>{NUMBER_WIDTH}.6f}"
        )

    print("=" * TABLE_WIDTH)

if __name__ == "__main__":
    main()


