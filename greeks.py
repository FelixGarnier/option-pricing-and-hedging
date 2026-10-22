"""Closed-form Black-Scholes Greeks. No dividends, T > 0, sigma > 0.

Delta is a number of shares per option. Gamma is the change in delta per
unit of spot. Vega is dV/dsigma with sigma in decimal, so a one-point move
in volatility is vega / 100. Theta is dV/dt in price per year: calendar time
advances and maturity shrinks, so theta = -dV/dT. Rho is dV/dr with r in
decimal.
"""

import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call delta, N(d1)."""
    return norm.cdf(_d1(S, K, T, r, sigma))


def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta, N(d1) - 1."""
    return norm.cdf(_d1(S, K, T, r, sigma)) - 1


def gamma_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma, shared by the call and the put."""
    return norm.pdf(_d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


gamma_put = gamma_call


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega, shared by the call and the put. Per 1.00 of volatility, not per vol point."""
    return S * norm.pdf(_d1(S, K, T, r, sigma)) * np.sqrt(T)


def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call theta, dV/dt in price per year."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)


def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put theta, dV/dt in price per year."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call rho, dV/dr with r in decimal."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def rho_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put rho, dV/dr with r in decimal."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


if __name__ == "__main__":
    from bs import bs_call, bs_put

    K, T, r, sigma = 100.0, 1.0, 0.05, 0.2
    spots = (80.0, 100.0, 120.0)
    h_S, h_G = 1e-4, 1e-2
    h_sig, h_T, h_r = 1e-4, 1e-5, 1e-4

    def report(name: str, gap: float, tol: float) -> bool:
        status = "OK" if gap <= tol else "FAIL"
        print(f"{status:4} {name:24} |gap|={gap:.3e}  tol={tol:.0e}")
        return gap <= tol

    passed = True
    for S in spots:
        d_call = delta_call(S, K, T, r, sigma)
        d_put = delta_put(S, K, T, r, sigma)
        fd_call = (bs_call(S + h_S, K, T, r, sigma) - bs_call(S - h_S, K, T, r, sigma)) / (2 * h_S)
        fd_put = (bs_put(S + h_S, K, T, r, sigma) - bs_put(S - h_S, K, T, r, sigma)) / (2 * h_S)
        passed &= report(f"delta_call S={S:.0f}", abs(d_call - fd_call), 1e-8)
        passed &= report(f"delta_put S={S:.0f}", abs(d_put - fd_put), 1e-8)
        passed &= report(f"delta call-put S={S:.0f}", abs(d_call - d_put - 1.0), 1e-12)

        g_call = gamma_call(S, K, T, r, sigma)
        g_put = gamma_put(S, K, T, r, sigma)
        fd_g_call = (
            bs_call(S + h_G, K, T, r, sigma) - 2 * bs_call(S, K, T, r, sigma) + bs_call(S - h_G, K, T, r, sigma)
        ) / h_G**2
        fd_g_put = (
            bs_put(S + h_G, K, T, r, sigma) - 2 * bs_put(S, K, T, r, sigma) + bs_put(S - h_G, K, T, r, sigma)
        ) / h_G**2
        passed &= report(f"gamma_call S={S:.0f}", abs(g_call - fd_g_call), 1e-6)
        passed &= report(f"gamma_put S={S:.0f}", abs(g_put - fd_g_put), 1e-6)
        passed &= report(f"gamma call-put S={S:.0f}", abs(g_call - g_put), 1e-12)

        # Bumping maturity in the pricer returns dV/dT. Theta is dV/dt = -dV/dT.
        fd_th_call = -(bs_call(S, K, T + h_T, r, sigma) - bs_call(S, K, T - h_T, r, sigma)) / (2 * h_T)
        fd_th_put = -(bs_put(S, K, T + h_T, r, sigma) - bs_put(S, K, T - h_T, r, sigma)) / (2 * h_T)
        passed &= report(f"theta_call S={S:.0f}", abs(theta_call(S, K, T, r, sigma) - fd_th_call), 1e-6)
        passed &= report(f"theta_put S={S:.0f}", abs(theta_put(S, K, T, r, sigma) - fd_th_put), 1e-6)

        fd_rho_call = (bs_call(S, K, T, r + h_r, sigma) - bs_call(S, K, T, r - h_r, sigma)) / (2 * h_r)
        fd_rho_put = (bs_put(S, K, T, r + h_r, sigma) - bs_put(S, K, T, r - h_r, sigma)) / (2 * h_r)
        passed &= report(f"rho_call S={S:.0f}", abs(rho_call(S, K, T, r, sigma) - fd_rho_call), 1e-4)
        passed &= report(f"rho_put S={S:.0f}", abs(rho_put(S, K, T, r, sigma) - fd_rho_put), 1e-4)

        fd_v_call = (bs_call(S, K, T, r, sigma + h_sig) - bs_call(S, K, T, r, sigma - h_sig)) / (2 * h_sig)
        fd_v_put = (bs_put(S, K, T, r, sigma + h_sig) - bs_put(S, K, T, r, sigma - h_sig)) / (2 * h_sig)
        passed &= report(f"vega vs call S={S:.0f}", abs(vega(S, K, T, r, sigma) - fd_v_call), 1e-4)
        passed &= report(f"vega vs put S={S:.0f}", abs(vega(S, K, T, r, sigma) - fd_v_put), 1e-4)

    print("PASS" if passed else "FAIL")
