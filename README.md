# American Option Pricing via LSMC with ML-Based Continuation Regression

ITCS 5154 — Applied Machine Learning Capstone Project

## Overview

This project prices American put options using the **Longstaff–Schwartz Monte Carlo (LSMC)** method, comparing five different regression approaches for estimating the continuation value:

| Method | Description |
|---|---|
| **OLS** | Standard polynomial regression (Longstaff & Schwartz baseline) |
| **Ridge** | L2-regularized polynomial regression |
| **Lasso** | L1-regularized polynomial regression |
| **Random Forest** | Tree-ensemble continuation estimator |
| **Gradient Boosting** | Sequential boosted trees continuation estimator |

## Repository Structure

```
├── src/                          # Core library code
│   ├── simulation/
│   │   └── gbm.py                # Geometric Brownian Motion path simulator
│   └── pricers/
│       ├── ols.py                # OLS LSMC pricer
│       ├── ridge.py              # Ridge LSMC pricer
│       ├── lasso.py              # Lasso LSMC pricer
│       ├── random_forest.py      # Random Forest LSMC pricer
│       └── gradient_boosting.py  # Gradient Boosting LSMC pricer
├── scripts/                      # Runnable entry points
│   ├── run_comparison.py         # Quick comparison table of all methods
│   ├── run_experiments.py        # Full experiment suite (benchmark, convergence, etc.)
│   └── generate_plots.py         # Generate figures from experiment CSVs
├── output/                       # Experiment results & figures (gitignored)
│   ├── *.csv
│   └── figures/
├── docs/                         # Project proposal and documentation
│   └── Proposal_...pdf
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick comparison of all methods
python scripts/run_comparison.py

# Run full experiment suite
python scripts/run_experiments.py all

# Generate all plots from experiment results
python scripts/generate_plots.py all
```

## Individual Experiments

```bash
python scripts/run_experiments.py benchmark    # Method comparison table
python scripts/run_experiments.py convergence  # Price vs number of paths
python scripts/run_experiments.py hyperparam   # Hyperparameter sensitivity
python scripts/run_experiments.py runtime      # Runtime comparison
python scripts/run_experiments.py optparams    # Option parameter variation
```

## Reference

Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113–147.
