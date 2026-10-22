"""Monte Carlo check of the Black-Scholes call.

Both numbers estimate the same quantity: the discounted risk-neutral
expectation of the call payoff under geometric Brownian motion. The closed
form is exact. The Monte Carlo price is a noisy average. The 95% interval
is the noise of that average. Neither number is a market price.
"""

import numpy as np

from bs import bs_call


def monte_carlo_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int = 10**6,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Monte Carlo call price and the two ends of the 95% interval.

    The spot is drawn at maturity only. ``seed`` fixes the generator, so a
    second call with the same arguments returns the same interval.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    discounted_payoff = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    price = float(np.mean(discounted_payoff))
    half = 1.96 * float(np.std(discounted_payoff)) / np.sqrt(n)
    return price, price - half, price + half


def call_price_in_interval(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int = 10**6,
    seed: int = 0,
) -> bool:
    """True when the Black-Scholes call lies inside the Monte Carlo interval."""
    _, low, high = monte_carlo_call(S, K, T, r, sigma, n, seed)
    bs_price = bs_call(S, K, T, r, sigma)
    return bool(low <= bs_price <= high)


if __name__ == "__main__":
    price, low, high = monte_carlo_call(100, 100, 1, 0.05, 0.2)
    bs_price = bs_call(100, 100, 1, 0.05, 0.2)
    inside = low <= bs_price <= high
    print(f"Monte Carlo {price:.6f}  interval [{low:.6f}, {high:.6f}]")
    print(f"Black-Scholes {bs_price:.6f}")
    print("OK" if inside else "FAIL")
