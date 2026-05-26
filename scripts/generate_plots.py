# generate_plots.py
# Generate all visualizations from the experiment CSVs in output/.
#
# Usage:
#   python3 scripts/generate_plots.py              # generate all plots
#   python3 scripts/generate_plots.py convergence  # generate just one

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
BENCHMARK = 4.478


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found — run the experiment first")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig, name):
    ensure_dir()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# -----------------------------------------------------------------------
# Plot 1: Benchmark comparison bar chart
# -----------------------------------------------------------------------
def plot_benchmark():
    print("\nPlot: Benchmark Comparison")
    data = load_csv("benchmark.csv")
    if not data: return

    methods = [r["method"] for r in data]
    prices  = [float(r["price"]) for r in data]
    times   = [float(r["runtime_s"]) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # price bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(methods, prices, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(BENCHMARK, color="red", linestyle="--", linewidth=1, label=f"L&S benchmark={BENCHMARK}")
    ax1.set_ylabel("Estimated Price")
    ax1.set_title("American Put Price by Method")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=25)

    # runtime bar chart
    ax2.bar(methods, times, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Runtime by Method")
    ax2.tick_params(axis="x", rotation=25)

    fig.suptitle("LSMC Regression Methods — Benchmark Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "benchmark_comparison.png")


# -----------------------------------------------------------------------
# Plot 2: Convergence (price vs # paths)
# -----------------------------------------------------------------------
def plot_convergence():
    print("\nPlot: Convergence Analysis")
    data = load_csv("convergence.csv")
    if not data: return

    fig, ax = plt.subplots(figsize=(8, 5))
    methods_set = dict.fromkeys(r["method"] for r in data)

    for method in methods_set:
        subset = [r for r in data if r["method"] == method]
        paths  = [int(r["paths"]) for r in subset]
        prices = [float(r["price"]) for r in subset]
        ax.plot(paths, prices, "o-", label=method, markersize=5)

    ax.axhline(BENCHMARK, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Benchmark={BENCHMARK}")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Estimated Price")
    ax.set_title("Convergence: Price vs Number of Paths")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "convergence.png")


# -----------------------------------------------------------------------
# Plot 3: Hyperparameter sensitivity
# -----------------------------------------------------------------------
def plot_hyperparam():
    print("\nPlot: Hyperparameter Sensitivity")
    data = load_csv("hyperparam_sweep.csv")
    if not data: return

    groups = {}
    for r in data:
        key = (r["method"], r["param"])
        groups.setdefault(key, []).append(r)

    n = len(groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, ((method, param), subset) in enumerate(groups.items()):
        vals   = [float(r["value"]) for r in subset]
        prices = [float(r["price"]) for r in subset]
        axes[i].plot(vals, prices, "s-", color="steelblue", markersize=6)
        axes[i].axhline(BENCHMARK, color="red", linestyle="--", alpha=0.5)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Price")
        axes[i].set_title(f"{method} — {param}")
        if max(vals) / (min(vals) + 1e-12) > 50:
            axes[i].set_xscale("log")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Hyperparameter Sensitivity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "hyperparam_sensitivity.png")


# -----------------------------------------------------------------------
# Plot 4: Runtime comparison
# -----------------------------------------------------------------------
def plot_runtime():
    print("\nPlot: Runtime Comparison")
    data = load_csv("runtime.csv")
    if not data: return

    methods = [r["method"] for r in data]
    prices  = [float(r["price"]) for r in data]
    times   = [float(r["avg_time_s"]) for r in data]
    errors  = [float(r["std_time_s"]) for r in data]
    abs_errs = [abs(p - BENCHMARK) for p in prices]

    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]
    cmap = plt.cm.tab10
    offsets = [(10, 8), (10, -14), (10, 18), (10, 8), (10, -14)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (m, t, err) in enumerate(zip(methods, times, abs_errs)):
        ax.scatter(t, err, s=160, color=cmap(i), edgecolors="black",
                   linewidth=0.8, zorder=3, marker=markers[i % len(markers)],
                   label=m)
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(m, (t, err), textcoords="offset points",
                    xytext=(ox, oy), fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    ax.set_xlabel("Avg Runtime (seconds)", fontsize=11)
    ax.set_ylabel("|Price Error| vs Benchmark", fontsize=11)
    ax.set_title("Accuracy vs Speed Tradeoff", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    close_idx = [i for i, t in enumerate(times) if t < 1.0]
    if len(close_idx) >= 2:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        axins = ax.inset_axes([0.28, 0.55, 0.33, 0.35])

        for i in close_idx:
            axins.scatter(times[i], abs_errs[i], s=180,
                          color=cmap(i), edgecolors="black", linewidth=0.8,
                          marker=markers[i % len(markers)], zorder=3)
            axins.annotate(methods[i], (times[i], abs_errs[i]),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=9, fontweight="bold")

        t_vals = [times[i] for i in close_idx]
        e_vals = [abs_errs[i] for i in close_idx]
        t_pad = (max(t_vals) - min(t_vals)) * 0.4 + 0.02
        e_pad = (max(e_vals) - min(e_vals)) * 0.6 + 0.002
        axins.set_xlim(min(t_vals) - t_pad, max(t_vals) + t_pad)
        axins.set_ylim(min(e_vals) - e_pad, max(e_vals) + e_pad)
        axins.set_title("Zoom: Linear Models", fontsize=9, fontstyle="italic")
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=8)

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",
                   linestyle="--", linewidth=0.8)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
    save(fig, "runtime_tradeoff.png")


# -----------------------------------------------------------------------
# Plot 5: Option parameter variation heatmap
# -----------------------------------------------------------------------
def plot_option_params():
    print("\nPlot: Option Parameter Variation")
    data = load_csv("option_params.csv")
    if not data: return

    scenarios = list(dict.fromkeys(r["scenario"] for r in data))
    methods   = list(dict.fromkeys(r["method"] for r in data))

    matrix = np.zeros((len(scenarios), len(methods)))
    for r in data:
        i = scenarios.index(r["scenario"])
        j = methods.index(r["method"])
        matrix[i, j] = float(r["price"])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=9)

    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Price")
    ax.set_title("American Put Price Across Scenarios and Methods")
    fig.tight_layout()
    save(fig, "option_params_heatmap.png")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
PLOTS = {
    "benchmark": plot_benchmark,
    "convergence": plot_convergence,
    "hyperparam": plot_hyperparam,
    "runtime": plot_runtime,
    "optparams": plot_option_params,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        for fn in PLOTS.values():
            fn()
    elif sys.argv[1] in PLOTS:
        PLOTS[sys.argv[1]]()
    else:
        print(f"Unknown plot: {sys.argv[1]}")
        print(f"Available: {', '.join(PLOTS.keys())}, all")
        sys.exit(1)
