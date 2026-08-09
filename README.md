# Contextual Bandits OPE Lab

> Work in progress — Summer 2026

An empirical study of **off-policy evaluation (OPE)** in contextual bandits.

The project investigates when and why common OPE estimators fail as a function of:

- sample size,
- policy overlap,
- propensity estimation,
- reward-model misspecification.

The main estimators studied are:

- **Direct Method (DM)**
- **Inverse Propensity Scoring (IPS)**
- **Self-Normalized IPS (SNIPS)**
- **Doubly Robust (DR)**

## What this repository contains

The project is built around a controlled synthetic contextual-bandit environment where the true policy value is known exactly. This makes it possible to compare Monte Carlo behavior against theoretical quantities such as bias, variance, and mean squared error.

Current experiments include:

- validation of the synthetic data-generating process,
- sample-size scaling,
- overlap deterioration,
- empirical verification of double robustness,
- estimation of reward and propensity nuisance models.

The goal is not only to implement the estimators, but to understand their **failure modes and bias–variance trade-offs** under controlled violations of their assumptions.

## Repository structure

```text
ope/            Core data-generating process, estimators, and theory
experiments/    Reproducible simulation experiments
results/        Generated tables and figures
docs/           Quarto report and analysis
```

## Status

This repository is under active development.

A full write-up with experimental results, figures, and interpretation is being prepared in Quarto.

## Motivation

Off-policy evaluation asks:

> How well would a target policy have performed if we had deployed it, given only data collected under a different policy?

This problem appears in contextual bandits, recommender systems, experimentation, and sequential decision-making.

The project serves as a focused empirical study of the statistical foundations of OPE and as preparation for further work on contextual bandits.