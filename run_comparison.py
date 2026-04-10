# run_comparison.py
# Unified runner that executes all 5 LSMC regression methods on the same
# simulated paths and produces a comparison table.
#
# Usage:
#   python3 run_comparison.py

import time
import numpy as np
import os

# import all pricers
from lsmc import lsmc_american_put
from lsmc_ridge import lsmc_american_put_ridge
from lsmc_lasso import lsmc_american_put_lasso
from lsmc_random_forest import lsmc_american_put_rf
from lsmc_gradient_boosting import lsmc_american_put_gb



DEFAULT_PARAMS = dict(
    S0=36.0,
    K=40.0,
    T=1.0,
    r=0.06,
    sigma=0.2,
    N=50,
    seed=42,
)
BENCHMARK_PRICE = 4.478  # L&S 2001 reference value


def run_single_method(method_name, pricer_fn, paths, **extra_kwargs):
    """Run one pricer, return dict with price, runtime, and error vs benchmark."""
    params = {**DEFAULT_PARAMS, "paths": paths, **extra_kwargs}
    t0 = time.perf_counter()
    price = pricer_fn(**params)
    elapsed = time.perf_counter() - t0
    return {
        "method": method_name,
        "paths": paths,
        "price": price,
        "error": price - BENCHMARK_PRICE,
        "pct_error": (price - BENCHMARK_PRICE) / BENCHMARK_PRICE * 100,
        "runtime_s": elapsed,
    }


def get_all_methods(paths=100_000):
    # Return list of (name, pricer_fn, paths, extra_kwargs) for all methods.
    # tree-based methods use fewer paths by default because they're much slower
    tree_paths = min(paths, 10_000)
    return [
        ("OLS (deg 2)", lsmc_american_put, paths, {"degree": 2}),
        ("Ridge (α=1.0)", lsmc_american_put_ridge, paths, {"degree": 2, "alpha": 1.0}),
        ("Lasso (α=0.001)", lsmc_american_put_lasso, paths, {"degree": 2, "alpha": 0.001}),
        ("Random Forest", lsmc_american_put_rf, tree_paths, {"n_estimators": 50, "max_depth": 4}),
        ("Gradient Boosting", lsmc_american_put_gb, tree_paths, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}),
    ]


def run_all(paths=100_000, save_csv=False):
    #Run all methods and print/save comparison table.
    methods = get_all_methods(paths)
    results = []

    header = f"{'Method':<22} {'Paths':>8} {'Price':>8} {'Error':>8} {'%Err':>7} {'Time(s)':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, fn, p, kwargs in methods:
        res = run_single_method(name, fn, p, **kwargs)
        results.append(res)
        print(f"{res['method']:<22} {res['paths']:>8,} {res['price']:>8.4f} "
              f"{res['error']:>+8.4f} {res['pct_error']:>+7.2f}% {res['runtime_s']:>8.2f}")

    print(sep)
    print(f"Benchmark (L&S 2001): {BENCHMARK_PRICE:.3f}\n")

    if save_csv:
        os.makedirs("results", exist_ok=True)
        import csv
        with open("results/comparison.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Saved → results/comparison.csv")

    return results


if __name__ == "__main__":
    print("LSMC American Put — All Regression Methods Comparison\n")
    print(f"Params: S0={DEFAULT_PARAMS['S0']}, K={DEFAULT_PARAMS['K']}, "
          f"T={DEFAULT_PARAMS['T']}, r={DEFAULT_PARAMS['r']}, σ={DEFAULT_PARAMS['sigma']}, "
          f"N={DEFAULT_PARAMS['N']}\n")
    run_all(paths=100_000)
