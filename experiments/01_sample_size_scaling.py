import numpy as np

from ope import dgp, estimators, theory

NUMBER_OF_REPLICATIONS = 2000
BASE_SEED = 42

def ips_monte_carlo_statistics(
        number_of_replications, 
        sample_size, 
        logging_policy, 
        target_policy,
        base_seed
        ):
    
    ips_estimates = np.zeros(number_of_replications)
    for repetition in range(number_of_replications):
        logged_data = dgp.generate_logged_data(
            sample_size,
            logging_policy,
            seed=base_seed + repetition,
        )

        ips_estimates[repetition] = estimators.ips_policy_value(logged_data)


    true_value = theory.exact_policy_value(dgp.context_probabilities, dgp.target_policy, dgp.true_mu)

    # Calculate mean estimate
    ips_monte_carlo_mean = np.mean(ips_estimates)

    # Calculate empirical bias
    ips_empirical_bias = ips_monte_carlo_mean - true_value

    # Calculate empirical variance
    ips_empirical_variance = np.mean((ips_estimates - ips_monte_carlo_mean)**2)

    # Calculate empirical MSE
    ips_empirical_mse = np.mean((ips_estimates - true_value)**2)

    ips_empirical_std = np.sqrt(ips_empirical_variance)
    ips_scaled_variance = sample_size * ips_empirical_variance

    assert np.isclose(ips_empirical_mse, ips_empirical_bias**2 + ips_empirical_variance)

    return {
        "mean": ips_monte_carlo_mean,
        "bias": ips_empirical_bias,
        "variance": ips_empirical_variance,
        "mse": ips_empirical_mse,
        "std": ips_empirical_std,
        "scaled_variance": ips_scaled_variance,
        "estimates": ips_estimates,
        "min": ips_estimates.min(),
        "max": ips_estimates.max(),
    }



def snips_monte_carlo_statistics(
        number_of_replications, 
        sample_size, 
        logging_policy,
        target_policy,
        base_seed
        ):

    # Run many independent SNIPS experiments
    snips_estimates = np.zeros(number_of_replications)

    for repetition in range(number_of_replications):
        logged_data = dgp.generate_logged_data(
            sample_size,
            logging_policy,
            seed=base_seed + repetition,
        )

        snips_estimates[repetition] = estimators.snips_policy_value(logged_data)

    true_value = theory.exact_policy_value(dgp.context_probabilities, dgp.target_policy, dgp.true_mu)

    # Calculate mean estimate
    snips_monte_carlo_mean = np.mean(snips_estimates)

    # Calculate empirical bias
    snips_empirical_bias = snips_monte_carlo_mean - true_value

    # Calculate empirical variance
    snips_empirical_variance = np.mean((snips_estimates - snips_monte_carlo_mean)**2)

    # Calculate empirical MSE
    snips_empirical_mse = np.mean((snips_estimates - true_value)**2)

    snips_empirical_std = np.sqrt(snips_empirical_variance)
    snips_scaled_variance = sample_size * snips_empirical_variance

    assert np.isclose(snips_empirical_mse, snips_empirical_bias**2 + snips_empirical_variance)

    return {
        "mean": snips_monte_carlo_mean,
        "bias": snips_empirical_bias,
        "variance": snips_empirical_variance,
        "mse": snips_empirical_mse,
        "std": snips_empirical_std,
        "scaled_variance": snips_scaled_variance,
        "estimates": snips_estimates,
        "min": snips_estimates.min(),
        "max": snips_estimates.max(),
    }



def main():

    print("\n" + "=" * 60)
    print("SAMPLE SIZE SCALING EXPERIMENT")
    print("=" * 60)

    print(f"\nMonte Carlo replications: {2000}")
    print(f"True policy value: {theory.exact_policy_value(dgp.context_probabilities, dgp.target_policy, dgp.true_mu):.6f}")
    print(f"Base seed: {BASE_SEED}")

    print("\n" + "-" * 96)

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
    print("-" * 96)

    for sample_size in (50, 200, 1_000, 5_000):
        ips_stats = ips_monte_carlo_statistics(
            NUMBER_OF_REPLICATIONS,
            sample_size,
            dgp.logging_policy,
            dgp.target_policy,
            BASE_SEED + sample_size,
        )

        snips_stats = snips_monte_carlo_statistics(
            NUMBER_OF_REPLICATIONS,
            sample_size,
            dgp.logging_policy,
            dgp.target_policy,
            BASE_SEED + sample_size,
        )

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

        print( "-" * 96)

    print("=" * 96)


if __name__ == "__main__":
    main()