"""Black-Scholes prices for European calls and puts, without dividends.

Time is in years. The rate ``r`` is continuously compounded. Volatility is a
decimal (``0.2`` for 20%).
"""

import numpy as np
from scipy.stats import norm


def _broadcast(S, K, T, r, sigma):
    """Return a scalar flag and the five inputs broadcast to a common shape."""
    scalar = all(np.ndim(x) == 0 for x in (S, K, T, r, sigma))
    arrays = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, dtype=float)) for x in (S, K, T, r, sigma))
    )
    return scalar, arrays


def _d1_d2(S, K, T, r, sigma):
    """Black-Scholes d1 and d2. Inputs must already be masked to T > 0 and sigma > 0."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Price of a European call.

    ``T = 0`` returns the payoff. ``sigma = 0`` returns the discounted payoff
    of the certain forward. A scalar input returns a float.
    """
    scalar, (S, K, T, r, sigma) = _broadcast(S, K, T, r, sigma)
    price = np.empty(S.shape, dtype=float)

    expired = T == 0
    zero_vol = (sigma == 0) & ~expired
    regular = ~(expired | zero_vol)

    price[expired] = np.maximum(S[expired] - K[expired], 0.0)
    price[zero_vol] = np.maximum(
        S[zero_vol] - K[zero_vol] * np.exp(-r[zero_vol] * T[zero_vol]), 0.0
    )
    if np.any(regular):
        d1, d2 = _d1_d2(S[regular], K[regular], T[regular], r[regular], sigma[regular])
        price[regular] = (
            S[regular] * norm.cdf(d1)
            - K[regular] * np.exp(-r[regular] * T[regular]) * norm.cdf(d2)
        )

    if scalar:
        return float(price[0])
    return price


def bs_put(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Price of a European put.

    Written from the closed form, not from put-call parity, so the parity
    check below is an independent test. Same edge cases as ``bs_call``.
    """
    scalar, (S, K, T, r, sigma) = _broadcast(S, K, T, r, sigma)
    price = np.empty(S.shape, dtype=float)

    expired = T == 0
    zero_vol = (sigma == 0) & ~expired
    regular = ~(expired | zero_vol)

    price[expired] = np.maximum(K[expired] - S[expired], 0.0)
    price[zero_vol] = np.maximum(
        K[zero_vol] * np.exp(-r[zero_vol] * T[zero_vol]) - S[zero_vol], 0.0
    )
    if np.any(regular):
        d1, d2 = _d1_d2(S[regular], K[regular], T[regular], r[regular], sigma[regular])
        price[regular] = (
            K[regular] * np.exp(-r[regular] * T[regular]) * norm.cdf(-d2)
            - S[regular] * norm.cdf(-d1)
        )

    if scalar:
        return float(price[0])
    return price


def parity_holds(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    tol: float = 1e-10,
) -> bool:
    """True when |C - P - (S - K e^{-rT})| stays within ``tol`` on every entry."""
    gap = np.abs(bs_call(S, K, T, r, sigma) - bs_put(S, K, T, r, sigma) - S + K * np.exp(-r * T))
    return bool(np.max(gap) <= tol)


if __name__ == "__main__":
    cases = {
        "OTM": (80, 100, 1.0, 0.05, 0.2),
        "ATM": (100, 100, 1.0, 0.05, 0.2),
        "ITM": (120, 100, 1.0, 0.05, 0.2),
        "expiry": (110, 100, 0.0, 0.05, 0.2),
        "zero vol": (100, 100, 1.0, 0.05, 0.0),
    }
    for name, args in cases.items():
        print(f"{'OK' if parity_holds(*args) else 'FAIL'} parity {name}")
    mixed = parity_holds(100, 100, np.array([0.0, 0.5, 1.0]), 0.05, 0.2)
    print(f"{'OK' if mixed else 'FAIL'} parity mixed maturities")
    print(f"ATM call {bs_call(100, 100, 1, 0.05, 0.2):.8f}")
