# Contextual Bandits OPE Lab

An empirical study of **off-policy evaluation (OPE)** in contextual bandits, focused on understanding when and why four classical estimators succeed or fail:

- Direct Method (DM)
- Inverse Propensity Scoring (IPS)
- Self-Normalized IPS (SNIPS)
- Doubly Robust (DR)

The project uses a controlled synthetic environment in which the true policy value is known exactly, allowing Monte Carlo behavior to be compared against theoretical bias, variance, and mean squared error.

## Questions studied

The experiments isolate four failure mechanisms:

1. **Weak overlap:** what happens as logging propensities become small?
2. **Nuisance-model misspecification:** how do errors in the reward and propensity models affect DM, IPS, SNIPS, and DR?
3. **Positivity violation:** what changes when the target policy assigns probability to actions that are never observed under the logging policy?
4. **Policy shift:** what happens as the target policy moves progressively away from a fixed logging policy?

## Main findings

- **Weak overlap is primarily a variance problem as long as positivity still holds.**  
Importance-weighted estimators become unstable as probability ratios grow.

- **Misspecification affects the estimators differently.**
DM relies on the reward model, IPS and SNIPS rely on logging propensities, while DR retains its double-robustness guarantee when at least one nuisance model is correctly specified.

- **Positivity failure is qualitatively different from weak overlap.**
The target policy value can cease to be identified from the observable logged-data distribution.

- **Policy shift can increase variance even when the logging policy and its
  support remain fixed**, through increased dispersion of importance weights.

- **No estimator dominates universally.**
Their behavior reflects different trade-offs between modeling assumptions, importance weighting, bias, and variance.

## Report

The full analysis, including theoretical derivations, Monte Carlo experiments,
figures, and interpretation, is available here:

- English: HTML · PDF
- Spanish: HTML · PDF

## Repository structure

```text
ope/            Core DGP, estimators, theory, metrics, and nuisance models
experiments/    Reproducible Monte Carlo experiments
results/        Generated tables and figures
docs/           Quarto reports and bibliography
tests/          Sanity and regression tests
```

## Reproducing the experiments

The four main experiments can be run from the repository root:

```bash
python experiments/02_overlap.py
python experiments/04_double_robustness_grid.py
python experiments/06_positivity_violation.py
python experiments/07_policy_shift.py
```

The report can then be rendered with Quarto:

```bash
quarto render docs/report_en.qmd
```


Tested with Python 3.14.2