import numpy as np

from ope import dgp, estimators, metrics, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZES = [50, 200, 1_000, 5_000]
BASE_SEED = 42
TABLE_WIDTH = 96
ESTIMATOR_FUNCTIONS = {
    "IPS": estimators.ips_policy_value,
    "SNIPS": estimators.snips_policy_value,
}


def monte_carlo_statistics(
    estimator_functions,
    number_of_replications, 
    sample_size, 
    logging_policy, 
    target_policy,
    base_seed
):
    
    estimates = {
        estimator_name: np.zeros(number_of_replications)
        for estimator_name in estimator_functions
    }

    for repetition in range(number_of_replications):
        logged_data = dgp.generate_logged_data(
            sample_size,
            logging_policy,
            seed=base_seed + repetition,
        )

        for estimator_name, estimator in estimator_functions.items():
            estimates[estimator_name][repetition] = estimator(
                logged_data
            )


    true_value = theory.exact_policy_value(dgp.context_probabilities, target_policy, dgp.true_mu)

    statistics ={}

    for estimator_name, estimator_estimates in estimates.items():
        statistics[estimator_name] = metrics.summarize_estimates(
            estimator_estimates,
            true_value,
            sample_size,
        )

    return statistics


def main():

    true_value = theory.exact_policy_value(dgp.context_probabilities, dgp.target_policy, dgp.true_mu)

    print("\n" + "=" * TABLE_WIDTH)
    print("SAMPLE SIZE SCALING EXPERIMENT".center(TABLE_WIDTH))
    print("=" * TABLE_WIDTH)

    print(f"\nMonte Carlo replications: {NUMBER_OF_REPLICATIONS}")
    print(f"True policy value: {true_value:.6f}")
    print(f"Base seed: {BASE_SEED}")

    print("\n" + "-" * TABLE_WIDTH)

    print(
        f"{'n':>8} "
        f"{'Estimator':>10} "
        f"{'Mean':>8} "
        f"{'Bias':>11} "
        f"{'Std':>9} "
        f"{'Variance':>15} "
        f"{'MSE':>10} "
        f"{'n × Var':>13}"
    )
    print("-" * TABLE_WIDTH)

    for sample_size in SAMPLE_SIZES:

        statistics = monte_carlo_statistics(
            ESTIMATOR_FUNCTIONS,
            NUMBER_OF_REPLICATIONS,
            sample_size,
            dgp.logging_policy,
            dgp.target_policy,
            BASE_SEED
        )

        ips_stats = statistics["IPS"]
        snips_stats = statistics["SNIPS"]
        
        
        for estimator_name, stats in (
            ("IPS", ips_stats),
            ("SNIPS", snips_stats),
        ):
            print(
                f"{sample_size:>8,d} "
                f"{estimator_name:>10} "
                f"{stats['mean']:>10.6f} "
                f"{stats['bias']:>+11.6f} "
                f"{stats['std']:>10.6f} "
                f"{stats['variance']:>12.6f} "
                f"{stats['mse']:>12.6f} "
                f"{stats['scaled_variance']:>12.6f}"
            )

        print( "-" * TABLE_WIDTH)

    print("=" * TABLE_WIDTH)


if __name__ == "__main__":
    main()