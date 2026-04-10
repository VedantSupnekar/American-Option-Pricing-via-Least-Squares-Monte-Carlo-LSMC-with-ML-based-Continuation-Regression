# experiments.py
# All experimental analyses for the LSMC project.
# Each experiment writes results to results/ as CSV for downstream plotting.
#
# Usage:
#   python3 experiments.py benchmark
#   python3 experiments.py convergence
#   python3 experiments.py hyperparam
#   python3 experiments.py runtime
#   python3 experiments.py optparams
#   python3 experiments.py all

import os
import sys
import csv
import time
import numpy as np

from lsmc import lsmc_american_put
from lsmc_ridge import lsmc_american_put_ridge
from lsmc_lasso import lsmc_american_put_lasso
from lsmc_random_forest import lsmc_american_put_rf
from lsmc_gradient_boosting import lsmc_american_put_gb


RESULTS_DIR = "results"
BENCHMARK = 4.478
SEED = 42

# standard params
BASE = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2, N=50)


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def write_csv(filename, rows, fieldnames):
    ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → saved {path}")


# -----------------------------------------------------------------------
# Experiment 1: Benchmark comparison table
# -----------------------------------------------------------------------
def run_benchmark():
    print("\n" + "=" * 60)
    print("Experiment 1: Benchmark Comparison (all methods, standard params)")
    print("=" * 60)

    configs = [
        ("OLS (deg 2)",       lsmc_american_put,       100_000, dict(degree=2, seed=SEED)),
        ("Ridge (α=1.0)",     lsmc_american_put_ridge,  100_000, dict(degree=2, alpha=1.0, seed=SEED)),
        ("Lasso (α=0.001)",   lsmc_american_put_lasso,  100_000, dict(degree=2, alpha=0.001, seed=SEED)),
        ("Random Forest",     lsmc_american_put_rf,      10_000,  dict(n_estimators=50, max_depth=4, seed=SEED)),
        ("Gradient Boosting", lsmc_american_put_gb,      10_000,  dict(n_estimators=100, max_depth=3, learning_rate=0.1, seed=SEED)),
    ]

    results = []
    header = f"{'Method':<22} {'Paths':>8} {'Price':>8} {'Error':>8} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    for name, fn, paths, kwargs in configs:
        t0 = time.perf_counter()
        price = fn(**BASE, paths=paths, **kwargs)
        elapsed = time.perf_counter() - t0
        err = price - BENCHMARK
        results.append(dict(method=name, paths=paths, price=round(price, 4),
                            error=round(err, 4), runtime_s=round(elapsed, 2)))
        print(f"{name:<22} {paths:>8,} {price:>8.4f} {err:>+8.4f} {elapsed:>8.2f}")

    write_csv("benchmark.csv", results, results[0].keys())
    return results


# -----------------------------------------------------------------------
# Experiment 2: Convergence analysis (price vs # paths)
# -----------------------------------------------------------------------
def run_convergence():
    print("\n" + "=" * 60)
    print("Experiment 2: Convergence Analysis (price vs # paths)")
    print("=" * 60)

    path_counts = [1_000, 5_000, 10_000, 50_000, 100_000]
    # tree methods capped at 10K for speed
    tree_max = 10_000

    methods = [
        ("OLS",    lsmc_american_put,       lambda p: dict(degree=2, seed=SEED)),
        ("Ridge",  lsmc_american_put_ridge,  lambda p: dict(degree=2, alpha=1.0, seed=SEED)),
        ("Lasso",  lsmc_american_put_lasso,  lambda p: dict(degree=2, alpha=0.001, seed=SEED)),
        ("RF",     lsmc_american_put_rf,      lambda p: dict(n_estimators=50, max_depth=4, seed=SEED)),
        ("GB",     lsmc_american_put_gb,      lambda p: dict(n_estimators=100, max_depth=3, learning_rate=0.1, seed=SEED)),
    ]

    results = []
    for name, fn, kw_fn in methods:
        max_p = tree_max if name in ("RF", "GB") else max(path_counts)
        valid_paths = [p for p in path_counts if p <= max_p]
        for p in valid_paths:
            t0 = time.perf_counter()
            price = fn(**BASE, paths=p, **kw_fn(p))
            elapsed = time.perf_counter() - t0
            results.append(dict(method=name, paths=p, price=round(price, 4),
                                runtime_s=round(elapsed, 2)))
            print(f"  {name:<8} paths={p:>8,}  price={price:.4f}  time={elapsed:.2f}s")

    write_csv("convergence.csv", results, results[0].keys())
    return results


# -----------------------------------------------------------------------
# Experiment 3: Hyperparameter sensitivity
# -----------------------------------------------------------------------
def run_hyperparam_sweep():
    print("\n" + "=" * 60)
    print("Experiment 3: Hyperparameter Sensitivity")
    print("=" * 60)

    results = []

    # Ridge: sweep alpha
    print("  Ridge — alpha sweep")
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        price = lsmc_american_put_ridge(**BASE, paths=100_000, degree=2, alpha=alpha, seed=SEED)
        results.append(dict(method="Ridge", param="alpha", value=alpha, price=round(price, 4)))
        print(f"    α={alpha:<8} → price={price:.4f}")

    # Lasso: sweep alpha
    print("  Lasso — alpha sweep")
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        price = lsmc_american_put_lasso(**BASE, paths=100_000, degree=2, alpha=alpha, seed=SEED)
        results.append(dict(method="Lasso", param="alpha", value=alpha, price=round(price, 4)))
        print(f"    α={alpha:<8} → price={price:.4f}")

    # RF: sweep n_estimators
    print("  Random Forest — n_estimators sweep")
    for n_est in [10, 50, 100, 200]:
        price = lsmc_american_put_rf(**BASE, paths=10_000, n_estimators=n_est, max_depth=4, seed=SEED)
        results.append(dict(method="RF", param="n_estimators", value=n_est, price=round(price, 4)))
        print(f"    n_est={n_est:<6} → price={price:.4f}")

    # RF: sweep max_depth
    print("  Random Forest — max_depth sweep")
    for depth in [2, 4, 6, 8]:
        price = lsmc_american_put_rf(**BASE, paths=10_000, n_estimators=50, max_depth=depth, seed=SEED)
        results.append(dict(method="RF", param="max_depth", value=depth, price=round(price, 4)))
        print(f"    depth={depth:<6} → price={price:.4f}")

    # GB: sweep learning_rate
    print("  Gradient Boosting — learning_rate sweep")
    for lr in [0.01, 0.05, 0.1, 0.3]:
        price = lsmc_american_put_gb(**BASE, paths=10_000, n_estimators=100, max_depth=3, learning_rate=lr, seed=SEED)
        results.append(dict(method="GB", param="learning_rate", value=lr, price=round(price, 4)))
        print(f"    lr={lr:<6} → price={price:.4f}")

    # GB: sweep n_estimators
    print("  Gradient Boosting — n_estimators sweep")
    for n_est in [50, 100, 200]:
        price = lsmc_american_put_gb(**BASE, paths=10_000, n_estimators=n_est, max_depth=3, learning_rate=0.1, seed=SEED)
        results.append(dict(method="GB", param="n_estimators", value=n_est, price=round(price, 4)))
        print(f"    n_est={n_est:<6} → price={price:.4f}")

    write_csv("hyperparam_sweep.csv", results, results[0].keys())
    return results


# -----------------------------------------------------------------------
# Experiment 4: Runtime comparison
# -----------------------------------------------------------------------
def run_runtime():
    print("\n" + "=" * 60)
    print("Experiment 4: Runtime Comparison")
    print("=" * 60)

    configs = [
        ("OLS", lsmc_american_put, 100_000, dict(degree=2, seed=SEED)),
        ("Ridge", lsmc_american_put_ridge, 100_000, dict(degree=2, alpha=1.0, seed=SEED)),
        ("Lasso", lsmc_american_put_lasso, 100_000, dict(degree=2, alpha=0.001, seed=SEED)),
        ("RF", lsmc_american_put_rf, 10_000, dict(n_estimators=50, max_depth=4, seed=SEED)),
        ("GB", lsmc_american_put_gb, 10_000, dict(n_estimators=100, max_depth=3, learning_rate=0.1, seed=SEED)),
    ]

    results = []
    # run each method 3 times and average
    n_runs = 3
    for name, fn, paths, kwargs in configs:
        times = []
        price = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            price = fn(**BASE, paths=paths, **kwargs)
            times.append(time.perf_counter() - t0)
        avg_t = np.mean(times)
        std_t = np.std(times)
        results.append(dict(method=name, paths=paths, price=round(price, 4),
                            avg_time_s=round(avg_t, 3), std_time_s=round(std_t, 3)))
        print(f"  {name:<8} paths={paths:>8,}  price={price:.4f}  "
              f"avg_time={avg_t:.3f}s ± {std_t:.3f}s")

    write_csv("runtime.csv", results, results[0].keys())
    return results


# -----------------------------------------------------------------------
# Experiment 5: Option parameter variation
# -----------------------------------------------------------------------
def run_option_params():
    print("\n" + "=" * 60)
    print("Experiment 5: Option Parameter Variation")
    print("=" * 60)

    scenarios = [
        ("Baseline (S0=36,K=40)",     dict(S0=36, K=40, T=1.0, r=0.06, sigma=0.2, N=50)),
        ("Deep ITM (S0=30,K=40)",     dict(S0=30, K=40, T=1.0, r=0.06, sigma=0.2, N=50)),
        ("ATM (S0=40,K=40)",          dict(S0=40, K=40, T=1.0, r=0.06, sigma=0.2, N=50)),
        ("Deep OTM (S0=50,K=40)",     dict(S0=50, K=40, T=1.0, r=0.06, sigma=0.2, N=50)),
        ("Long maturity (T=2)",       dict(S0=36, K=40, T=2.0, r=0.06, sigma=0.2, N=100)),
        ("High vol (σ=0.4)",          dict(S0=36, K=40, T=1.0, r=0.06, sigma=0.4, N=50)),
    ]

    methods = [
        ("OLS",   lsmc_american_put,       100_000, dict(degree=2, seed=SEED)),
        ("Ridge", lsmc_american_put_ridge,  100_000, dict(degree=2, alpha=1.0, seed=SEED)),
        ("Lasso", lsmc_american_put_lasso,  100_000, dict(degree=2, alpha=0.001, seed=SEED)),
        ("RF",    lsmc_american_put_rf,      10_000,  dict(n_estimators=50, max_depth=4, seed=SEED)),
        ("GB",    lsmc_american_put_gb,      10_000,  dict(n_estimators=100, max_depth=3, learning_rate=0.1, seed=SEED)),
    ]

    results = []
    for scenario_name, params in scenarios:
        print(f"\n  Scenario: {scenario_name}")
        for method_name, fn, paths, kwargs in methods:
            price = fn(**params, paths=paths, **kwargs)
            results.append(dict(scenario=scenario_name, method=method_name,
                                paths=paths, price=round(price, 4)))
            print(f"    {method_name:<8} → {price:.4f}")

    write_csv("option_params.csv", results, results[0].keys())
    return results


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
EXPERIMENTS = {
    "benchmark": run_benchmark,
    "convergence": run_convergence,
    "hyperparam": run_hyperparam_sweep,
    "runtime": run_runtime,
    "optparams": run_option_params,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        for exp_fn in EXPERIMENTS.values():
            exp_fn()
    elif sys.argv[1] in EXPERIMENTS:
        EXPERIMENTS[sys.argv[1]]()
    else:
        print(f"Unknown experiment: {sys.argv[1]}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}, all")
        sys.exit(1)
