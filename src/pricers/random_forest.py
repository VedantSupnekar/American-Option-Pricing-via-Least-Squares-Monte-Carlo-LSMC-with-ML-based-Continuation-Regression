# lsmc_random_forest.py
# LSMC American put pricing using Random Forest for the continuation value
# switching from linear regression to a tree ensemble — no need for polynomial
# basis or scaling, trees handle non-linearity automatically
#
# idea: at each time step, fit a random forest on (S_t, discounted future payoff)
# for in-the-money paths, then use predictions as continuation values.
# should capture non-linear payoff structure better than OLS/Ridge/Lasso
#
# tradeoff: much slower than linear methods — fitting 100 trees at every single
# time step gets expensive. using fewer paths and capping tree depth.
#
# Reference: Longstaff & Schwartz (2001)

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def lsmc_american_put_rf(S0, K, T, r, sigma, N, paths, n_estimators=50, max_depth=4, seed=None):
    """
    American put pricer using LSMC with Random Forest continuation estimator.

    Args:
        S0, K, T, r, sigma : standard option params
        N            : time steps
        paths        : number of Monte Carlo paths
        n_estimators : number of trees in the forest (default 50)
        max_depth    : max depth per tree (default 4, keeps it from overfitting)
        seed         : random seed
    Returns:
        estimated price at t=0
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    # simulate GBM paths
    S = np.zeros((N + 1, paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # terminal payoff
    V = np.maximum(K - S[-1], 0)
    df = np.exp(-r * dt)

    # backward induction
    for t in range(N - 1, 0, -1):
        V = V * df

        St = S[t]
        itm_idx = np.where(St < K)[0]

        if len(itm_idx) == 0:
            continue

        X_itm = St[itm_idx].reshape(-1, 1)  # RF expects 2D input
        Y_itm = V[itm_idx]

        # random forest to estimate continuation value
        # no scaling needed for trees
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,           # use all cores
            random_state=42,
        )
        rf.fit(X_itm, Y_itm)
        cont_val = rf.predict(X_itm)

        # exercise if immediate payoff > expected continuation
        exercise_val = K - St[itm_idx]
        exercise_idx = itm_idx[exercise_val > cont_val]
        V[exercise_idx] = K - St[exercise_idx]

    return np.mean(V * df)


if __name__ == "__main__":
    # same params as Longstaff & Schwartz (2001), Table 1
    # expected price ~4.47
    S0    = 36.0
    K     = 40.0
    T     = 1.0
    r     = 0.06
    sigma = 0.2
    N     = 50

    print("LSMC American Put w/ Random Forest regression")
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, N={N}\n")

    # tree methods are slow — using fewer paths than linear methods
    # fitting 50 trees * 50 time steps adds up
    print(f"{'paths':>8}  {'n_trees':>8}  {'price':>8}")
    print("-" * 30)
    for paths, n_est in [(5_000, 50), (10_000, 50), (10_000, 100)]:
        price = lsmc_american_put_rf(S0, K, T, r, sigma, N, paths,
                                     n_estimators=n_est, max_depth=4, seed=42)
        print(f"{paths:>8,}  {n_est:>8}  {price:>8.4f}")

    print("\npaper benchmark ~4.47")
    print("more paths = better estimate but slower runtime")
