import numpy as np
import pandas as pd

from ope import dgp, estimators, theory

NUMBER_OF_REPLICATIONS = 2000
SAMPLE_SIZE = 200
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_SEED = 42

def interpolate(
    correct_model, 
    wrong_model, 
    interpolation_factor
):
    return (1 - interpolation_factor) * correct_model + interpolation_factor * wrong_model

def run_double_robustness_grid(
    number_of_replications,
    sample_size,
    target_policy,
    true_mu,
    logging_policy,
    levels,
    base_seed,
):
    correct_mu = true_mu.copy()
    wrong_mu = np.full_like(true_mu, 0.5)

    correct_b_hat = logging_policy.copy()
    wrong_b_hat = np.full_like(logging_policy, 0.5)

    true_policy_value = theory.exact_policy_value(dgp.context_probabilities, target_policy, true_mu)

    results = []

    datasets = [
        dgp.generate_logged_data(
            sample_size, 
            logging_policy, 
            base_seed + k) for k in range(number_of_replications)
    ]

    for lambda_value in levels:
        for gamma_value in levels:

            interpolated_mu = interpolate(correct_mu, wrong_mu, lambda_value)
            interpolated_b_hat = interpolate(correct_b_hat, wrong_b_hat, gamma_value)
            dr_values = np.zeros(number_of_replications)

            for k, data in enumerate(datasets):
                dr_values[k] = estimators.dr_policy_value(
                    data,
                    interpolated_mu,
                    target_policy,
                    interpolated_b_hat
                )

            exact_bias = theory.exact_dr_bias(
                dgp.context_probabilities,
                interpolated_mu, 
                interpolated_b_hat, 
                true_mu, 
                logging_policy, 
                target_policy
            )
            
            monte_carlo_mean = np.mean(dr_values)
            monte_carlo_bias = monte_carlo_mean - true_policy_value
            monte_carlo_variance = np.mean((dr_values - monte_carlo_mean)**2)
            monte_carlo_std = np.sqrt(monte_carlo_variance)

            bias_error = monte_carlo_bias - exact_bias
            bias_std_error = monte_carlo_std / np.sqrt(number_of_replications)
            bias_z_score = bias_error / bias_std_error


            results.append({
                "lambda": lambda_value,
                "gamma": gamma_value,
                "exact_bias": exact_bias,
                "monte_carlo_bias":  monte_carlo_bias,
                "bias_error": bias_error,
                "bias_se": bias_std_error,
                "bias_z": bias_z_score,
                "variance": monte_carlo_variance,
                "mse": np.mean((dr_values - true_policy_value)**2),
                "mse_decomposition_error": np.mean((dr_values - true_policy_value)**2) - (monte_carlo_bias**2 + monte_carlo_variance),
            })

    return pd.DataFrame(results)



def validate_results(results):
    assert np.all(
        results.loc[results["lambda"] == 0.0, "exact_bias"] == 0.0
    ), "Exact bias should be zero when reward model is correct"

    assert np.all(
        results.loc[results["gamma"] == 0.0, "exact_bias"] == 0.0
    ), "Exact bias should be zero when logging policy model is correct"

    assert np.allclose(
        results.loc[
            (results["lambda"] == 1.0) 
            & (results["gamma"] == 1.0), 
            "exact_bias"
        ],
        -0.096
    ), "Exact bias should be -0.096 when both models are wrong"

    assert np.allclose(
        results["mse_decomposition_error"],
        0.0,
    ), "Monte Carlo MSE should be equal to the sum of squared bias and variance"


def main():

    results = run_double_robustness_grid(
        number_of_replications=NUMBER_OF_REPLICATIONS,
        sample_size=SAMPLE_SIZE,
        target_policy=dgp.target_policy,
        true_mu=dgp.true_mu,
        logging_policy=dgp.logging_policy,
        levels=LEVELS,
        base_seed=BASE_SEED,
    )

    validate_results(results)

    print(results.to_string(index=False))

    results.to_csv(
        "results/tables/double_robustness_grid.csv", 
        index=False,
    )


if __name__ == "__main__":
    main()