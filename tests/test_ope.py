import numpy as np

from ope import dgp, estimators, theory


def test_policies_are_valid_probability_distributions():
    for policy in [dgp.logging_policy, dgp.target_policy]:
        assert np.all(policy >= 0)
        assert np.all(policy <= 1)
        np.testing.assert_allclose(policy.sum(axis=1), 1.0)


def test_exact_policy_values():
    logging_value = theory.exact_policy_value(
        dgp.context_probabilities,
        dgp.logging_policy,
        dgp.true_mu,
    )

    target_value = theory.exact_policy_value(
        dgp.context_probabilities,
        dgp.target_policy,
        dgp.true_mu,
    )

    np.testing.assert_allclose(logging_value, 0.435)
    np.testing.assert_allclose(target_value, 0.66)


def test_logged_data_are_reproducible_given_seed():
    data_1 = dgp.generate_logged_data(
        n=100,
        logging_policy=dgp.logging_policy,
        seed=42,
    )

    data_2 = dgp.generate_logged_data(
        n=100,
        logging_policy=dgp.logging_policy,
        seed=42,
    )

    assert data_1 == data_2


def test_ips_and_snips_equal_sample_mean_on_policy():
    data = dgp.generate_logged_data(
        n=100,
        logging_policy=dgp.logging_policy,
        seed=42,
    )

    sample_mean = np.mean([observation["ri"] for observation in data])

    ips = estimators.ips_policy_value(
        data,
        target_policy=dgp.logging_policy,
    )

    snips = estimators.snips_policy_value(
        data,
        target_policy=dgp.logging_policy,
    )

    np.testing.assert_allclose(ips, sample_mean)
    np.testing.assert_allclose(snips, sample_mean)


def test_dm_with_true_reward_model_on_balanced_contexts():
    data = [
        {"xi": 0},
        {"xi": 1},
    ]

    estimate = estimators.dm_policy_value(
        data,
        target_policy=dgp.target_policy,
        reward_model=dgp.true_mu,
    )

    np.testing.assert_allclose(estimate, 0.66)


def test_exact_dr_bias_is_doubly_robust():
    wrong_mu = np.full_like(dgp.true_mu, 0.5)
    wrong_logging_policy = np.full_like(dgp.logging_policy, 0.5)

    bias_correct_mu = theory.exact_dr_bias(
        dgp.context_probabilities,
        dgp.true_mu,
        wrong_logging_policy,
        dgp.true_mu,
        dgp.logging_policy,
        dgp.target_policy,
    )

    bias_correct_logging = theory.exact_dr_bias(
        dgp.context_probabilities,
        wrong_mu,
        dgp.logging_policy,
        dgp.true_mu,
        dgp.logging_policy,
        dgp.target_policy,
    )

    bias_both_wrong = theory.exact_dr_bias(
        dgp.context_probabilities,
        wrong_mu,
        wrong_logging_policy,
        dgp.true_mu,
        dgp.logging_policy,
        dgp.target_policy,
    )

    np.testing.assert_allclose(bias_correct_mu, 0.0)
    np.testing.assert_allclose(bias_correct_logging, 0.0)
    np.testing.assert_allclose(bias_both_wrong, -0.096)