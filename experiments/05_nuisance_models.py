
import numpy as np
import pandas as pd

from ope import dgp, estimators, metrics, nuisance, theory

NUMBER_OF_REPLICATIONS = 2000
EVALUATION_SAMPLE_SIZE = 200
TRAINING_SAMPLE_SIZES = [100, 200, 500, 1_000, 5_000]
BASE_SEED = 42

def run_estimated_nuisance_experiment(
    number_of_replications,
    training_sample_size,
    evaluation_sample_size,
    base_seed
):
    true_value = theory.exact_policy_value(dgp.context_probabilities, dgp.target_policy, dgp.true_mu)

    estimates = {
        "DM": np.zeros(number_of_replications),
        "IPS": np.zeros(number_of_replications),
        "SNIPS": np.zeros(number_of_replications),
        "DR": np.zeros(number_of_replications),
    }

    for repetition in range(number_of_replications):
        training_data = dgp.generate_logged_data(
            training_sample_size,
            dgp.logging_policy,
            base_seed + 2*repetition
        )
        test_data = dgp.generate_logged_data(
            evaluation_sample_size,
            dgp.logging_policy,
            base_seed + 2*repetition + 1
        )

        estimated_mu = nuisance.fit_reward_model(training_data)
        estimated_b_hat = nuisance.fit_logging_policy(training_data)

        nuisance.check_probabilities(estimated_b_hat)

        estimates["DM"][repetition] = estimators.dm_policy_value(
            test_data, 
            dgp.target_policy, 
            estimated_mu
        )

        estimates["IPS"][repetition] = estimators.ips_policy_value(test_data)
        estimates["SNIPS"][repetition] = estimators.snips_policy_value(test_data)
        estimates["DR"][repetition] = estimators.dr_policy_value(
            test_data, 
            estimated_mu, 
            dgp.target_policy, 
            estimated_b_hat
        )

    statistics = {}

    for estimator_name, estimator_estimates in estimates.items():
        statistics[estimator_name] = metrics.summarize_estimates(
            estimator_estimates,
            true_value,
            evaluation_sample_size,
        )


    df = pd.DataFrame(statistics).T

    return df


    


def validate_results(results):

    assert isinstance(results, pd.DataFrame), "Results must be a pandas DataFrame"
    assert set(results.index) == {"DM", "IPS", "SNIPS", "DR"}, "Results must contain rows for DM, IPS, SNIPS, and DR"


def main():

    TABLE_WIDTH = 96

    print("\n" + "=" * TABLE_WIDTH)
    print("ESTIMATED NUISANCE EXPERIMENT".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)


    print(f"\nMonte Carlo replications: {NUMBER_OF_REPLICATIONS}")
    print(f"Evaluation sample size: {EVALUATION_SAMPLE_SIZE}")
    print(f"Training sample sizes (TSS): {TRAINING_SAMPLE_SIZES}")

    print("\n")
    print(f"{'-' * TABLE_WIDTH}")
    print(
        f"{'TSS':>12} "
        f"{'Estimator':>10} "
        f"{'Mean':>8} "
        f"{'Bias':>11} "
        f"{'Std':>9} "
        f"{'Variance':>15} "
        f"{'MSE':>10} "
    )

    print(f"{'-' * TABLE_WIDTH}")

    all_stats = []

    for training_sample_size in TRAINING_SAMPLE_SIZES:
        stats = run_estimated_nuisance_experiment(
            NUMBER_OF_REPLICATIONS,
            training_sample_size,
            EVALUATION_SAMPLE_SIZE,
            BASE_SEED
        )

        stats.index.name = "estimator"

        stats.insert(0, "training_sample_size", training_sample_size)

        all_stats.append(stats)

        validate_results(stats)

        for estimator_name, estimator_stats in stats.iterrows():
            print(
                f"{training_sample_size:>12,d} "
                f"{estimator_name:>10} "
                f"{estimator_stats['mean']:>10.6f} "
                f"{estimator_stats['bias']:>+11.6f} "
                f"{estimator_stats['std']:>10.6f} "
                f"{estimator_stats['variance']:>12.6f} "
                f"{estimator_stats['mse']:>12.6f} "
            )

        print("-" * TABLE_WIDTH)

    all_stats = pd.concat(all_stats, axis=0)

    validate_results(all_stats)

    all_stats.to_csv("results/tables/nuisance_training_sizes.csv")


if __name__ == "__main__":
    main()