
import numpy as np
import pandas as pd

from ope import dgp, estimators, metrics, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZE = 200
EPSILON_VALUES = [0.3, 0.1, 0.05, 0.02, 0.01]
SEED = 42
ESTIMATOR_FUNCTIONS = {"IPS": estimators.ips_policy_value, "SNIPS": estimators.snips_policy_value}


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

    statistics = {}

    for estimator_name, estimator_estimates in estimates.items():
        statistics[estimator_name] = metrics.summarize_estimates(
            estimator_estimates,
            true_value,
            sample_size,
        )

    return statistics


def run_overlap_experiment(
    epsilon_values, 
    number_of_replications, 
    sample_size, 
    base_seed
):

    rows = []

    for epsilon in epsilon_values:
        logging_policy = dgp.make_logging_policy(epsilon)
        stats = monte_carlo_statistics(
            ESTIMATOR_FUNCTIONS,
            number_of_replications,
            sample_size,
            logging_policy,
            dgp.target_policy,
            base_seed + int(epsilon * 1000),
        )

        for estimator_name, estimator_stats in stats.items():
            rows.append({
                "epsilon": epsilon,
                "estimator": estimator_name,
                **estimator_stats
            })

    return pd.DataFrame(rows)


def validate_results(results):
    ips_results = results.loc[results["estimator"] == "IPS"].sort_values(
        "epsilon", 
        ascending=False
    )

    epsilon_values = ips_results["epsilon"].to_numpy()

    exact_variances = np.array([
        theory.exact_ips_variance(
            dgp.context_probabilities, 
            dgp.make_logging_policy(epsilon), 
            dgp.target_policy, 
            dgp.true_mu,
            SAMPLE_SIZE,
        )
        for epsilon in epsilon_values
    ])

    assert np.all(np.diff(exact_variances) >= 0), "IPS variance does not decrease with increasing overlap"




def main():

    results = run_overlap_experiment(
        EPSILON_VALUES,
        NUMBER_OF_REPLICATIONS,
        SAMPLE_SIZE,
        SEED,
    )

    validate_results(results)

    print(results.to_string(index=False))

    results.to_csv(
        "results/tables/overlap.csv", 
        index=False
    )


if __name__ == "__main__":
    main()