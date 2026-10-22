"""Discrete delta hedge of a short European call.

The call is sold at the Black-Scholes price computed with the implied
volatility ``sigma_i``. The hedge delta is recomputed with that same
volatility. The spot path is a geometric Brownian motion with realized
volatility ``sigma_r``. Cash is compounded continuously at the rate ``r``.

The value returned by ``hedge_pnl`` is the cash left at maturity, after the
call payoff has been paid and the shares have been sold. That cash is the
P&L of the hedged sale.
"""

import numpy as np

from bs import bs_call
from greeks import delta_call


def simulate_spots(
    S0: float,
    r: float,
    sigma_r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """GBM spot paths. Shape ``(n_paths, n_steps + 1)``, column 0 is ``S0``.

    ``n_paths`` is the number of scenarios. ``n_steps`` is the number of
    intervals from 0 to ``T``, so the hedge is rebalanced ``n_steps`` times
    including inception. ``sigma_r`` is the realized volatility of the path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    spots = np.empty((n_paths, n_steps + 1))
    spots[:, 0] = S0
    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        spots[:, i + 1] = spots[:, i] * np.exp((r - 0.5 * sigma_r**2) * dt + sigma_r * np.sqrt(dt) * Z)
    return spots


def hedge_pnl(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_i: float,
    sigma_r: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    k: float = 0.0,
) -> np.ndarray:
    """P&L at maturity of a delta-hedged short call, one value per path.

    ``k`` is a proportional transaction cost. Buying or selling ``q`` shares
    removes ``k * abs(q) * S`` from the cash account, including the initial
    hedge and the final unwind. ``k = 0`` is a hedge with no costs.

    Delta is not recomputed on the last date: time left would be zero. The
    shares held into expiry are the position from the previous rebalance.
    """
    spots = simulate_spots(S0, r, sigma_r, T, n_steps, n_paths, seed)
    dt = T / n_steps
    premium = bs_call(S0, K, T, r, sigma_i)
    delta = delta_call(S0, K, T, r, sigma_i)
    cash = premium - spots[:, 0] * delta - k * np.abs(delta) * spots[:, 0]

    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt)
        t_left = T - i * dt
        delta_new = delta_call(spots[:, i], K, t_left, r, sigma_i)
        traded = delta_new - delta
        cash = cash - traded * spots[:, i] - k * np.abs(traded) * spots[:, i]
        delta = delta_new

    cash = cash * np.exp(r * dt)
    payoff = np.maximum(spots[:, -1] - K, 0.0)
    cash = cash - payoff + delta * spots[:, -1] - k * np.abs(delta) * spots[:, -1]
    return cash


if __name__ == "__main__":
    S0, r, sigma_r, T, seed = 100.0, 0.05, 0.2, 1.0, 0
    n_paths = 1000
    paths = simulate_spots(S0, r, sigma_r, T, n_steps=1, n_paths=n_paths, seed=seed)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma_r**2) * T + sigma_r * np.sqrt(T) * Z)

    gap_start = np.max(np.abs(paths[:, 0] - S0))
    gap_end = np.max(np.abs(paths[:, -1] - S_T))
    print(f"{'OK' if paths.shape == (n_paths, 2) else 'FAIL'} shape {paths.shape}")
    print(f"{'OK' if gap_start == 0.0 else 'FAIL'} column 0 is S0  |gap|={gap_start:.3e}")
    print(f"{'OK' if gap_end == 0.0 else 'FAIL'} one step matches the closed GBM  |gap|={gap_end:.3e}")
