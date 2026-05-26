# LSMC American put pricing but using Ridge regression instead of OLS
# for estimating the continuation value at each time step
#
# idea: the hard part of LSMC is estimating E[V | S_t], the expected
# continuation value. OLS works but can overfit with small ITM samples
# or high-degree basis. Ridge adds L2 penalty to shrink coefficients,
# which should help stability.
#
# Reference: Longstaff & Schwartz (2001)

import warnings
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


def lsmc_american_put_ridge(S0, K, T, r, sigma, N, paths, degree=2, alpha=1.0, seed=None):
    """
    American put pricer using LSMC with Ridge regression continuation estimator.

    Args:
        S0, K, T, r, sigma : standard option params
        N      : time steps
        paths  : number of Monte Carlo paths
        degree : polynomial degree for basis features (default 2)
        alpha  : Ridge regularization strength (default 1.0)
        seed   : random seed
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

    # terminal payoff for American put
    V = np.maximum(K - S[-1], 0)
    df = np.exp(-r * dt)  # discount per step

    # backward induction
    for t in range(N - 1, 0, -1):
        V = V * df  # discount future cash flows one step

        St = S[t]
        itm_idx = np.where(St < K)[0]  # only care about in the money paths

        if len(itm_idx) == 0:
            continue

        X_itm = St[itm_idx]
        Y_itm = V[itm_idx]

        # Ridge regression to estimate continuation value
        # need to scale first otherwise the L2 penalty is weirdly biased
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pipe.fit(X_itm.reshape(-1, 1), Y_itm)
            cont_val = pipe.predict(X_itm.reshape(-1, 1))

        # exercise if immediate payoff > expected continuation
        exercise_val = K - X_itm
        exercise_idx = itm_idx[exercise_val > cont_val]
        V[exercise_idx] = K - St[exercise_idx]

    return np.mean(V * df)


if __name__ == "__main__":
    # using same params as Table 1 in Longstaff & Schwartz (2001)
    # expected price ~4.47
    S0    = 36.0
    K     = 40.0
    T     = 1.0
    r     = 0.06
    sigma = 0.2
    N     = 50
    paths = 100_000

    print("LSMC American Put w/ Ridge regression")
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, N={N}, paths={paths}\n")

    # sweep a few alpha values to see how regularization affects price
    print(f"{'alpha':>8}  {'price':>8}")
    print("-" * 20)
    for a in [0.01, 0.1, 1.0, 10.0]:
        price = lsmc_american_put_ridge(S0, K, T, r, sigma, N, paths, degree=2, alpha=a, seed=42)
        print(f"{a:>8.3f}  {price:>8.4f}")

    print("\npaper benchmark ~4.47")
