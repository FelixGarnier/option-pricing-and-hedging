# Pricing and hedging, notes

European options, no dividends, constant continuous rate, constant volatility. Time is in years. A volatility of 20% is written `0.2`.

## 1. The model

The spot follows a geometric Brownian motion under the risk-neutral measure

\[
dS_t = r S_t \, dt + \sigma S_t \, dW_t.
\]

The drift is the risk-free rate because, under this measure, a price is the expectation of a discounted payoff. It is not a forecast of the stock's real return.

Itô's formula gives the log

\[
d \log S_t = \big(r - \tfrac12 \sigma^2\big) dt + \sigma \, dW_t.
\]

The \(-\tfrac12\sigma^2\) term is the convexity of the logarithm. The second derivative of \(\log S\) is \(-1/S^2\), and the Itô correction \(\tfrac12 \sigma^2 S^2 \times (-1/S^2)\) equals \(-\tfrac12\sigma^2\). It enforces

\[
\mathbb{E}[S_T] = S_0 e^{rT}.
\]

A log drift of \(r\) instead of \(r - \tfrac12\sigma^2\) would push that expectation up to \(S_0 e^{(r + \tfrac12\sigma^2)T}\) and make a Monte Carlo call too expensive.

Integrated to the horizon \(T\),

\[
S_T = S_0 \exp\big((r - \tfrac12\sigma^2)T + \sigma\sqrt{T}\, Z\big), \qquad Z \sim \mathcal{N}(0,1).
\]

## 2. The Black-Scholes price

\[
V_0 = e^{-rT} \, \mathbb{E}\big[\mathrm{payoff}(S_T)\big].
\]

The call payoff is \(\max(S_T - K, 0)\), the put payoff \(\max(K - S_T, 0)\). With

\[
d_1 = \frac{\ln(S/K) + (r + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T},
\]

and \(N\) the standard normal cdf,

\[
C = S\, N(d_1) - K e^{-rT} N(d_2),
\]

\[
P = K e^{-rT} N(-d_2) - S\, N(-d_1).
\]

\(N(d_2)\) is the risk-neutral probability of finishing in the money. \(N(d_1)\) is the call delta. The two differ by the same convexity that appears in the log drift.

Edge cases, both implemented in `bs.py`:

- \(T = 0\): the price is the payoff.
- \(\sigma = 0\) and \(T > 0\): the spot arrives at \(S e^{rT}\) for sure, and the price is the discounted payoff of that number.

Put-call parity does not go back through \(d_1\):

\[
C - P = S - K e^{-rT}.
\]

Call and put are coded separately. Building the put from the call by parity would make the parity test pass even with a wrong \(d_1\).

`monte_carlo_check.py` compares two computations of the same expectation. The formula is exact. The Monte Carlo price is a noisy average of \(10^6\) discounted payoffs. The 95% interval is the noise of that average. The test asks the exact number to fall inside the interval.

## 3. Greeks

\(n\) is the standard normal density. Formulas assume no dividends.

| Name | Derivative | Call | Put |
|---|---|---|---|
| Delta | \(\partial V / \partial S\) | \(N(d_1)\) | \(N(d_1) - 1\) |
| Gamma | \(\partial^2 V / \partial S^2\) | \(n(d_1) / (S \sigma \sqrt{T})\) | same |
| Vega | \(\partial V / \partial \sigma\) | \(S\, n(d_1) \sqrt{T}\) | same |
| Theta | \(\partial V / \partial t\) | \(-\dfrac{S n(d_1)\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)\) | \(-\dfrac{S n(d_1)\sigma}{2\sqrt{T}} + r K e^{-rT} N(-d_2)\) |
| Rho | \(\partial V / \partial r\) | \(K T e^{-rT} N(d_2)\) | \(-K T e^{-rT} N(-d_2)\) |

Units:

- Delta is a number of shares. An at-the-money one-year call has a delta near 0.5 to 0.6. A put delta is negative.
- Call and put share the gamma, because \(C - P\) is linear in \(S\).
- Vega above is for sigma in decimal. A move from 20% to 21% is a shock of 0.01.
- Theta above is price per year. A per-day theta is that number divided by 365. Bumping \(T\) in the pricer returns \(\partial V/\partial T = -\Theta\).
- Rho is for \(r\) in decimal. One basis point is a shock of 0.0001.

Near the money, as \(T \to 0\), \(d_1 \to 0\) when \(S\) stays on \(K\), so \(n(d_1)\) stays near 0.4. Gamma has \(\sqrt{T}\) in the denominator and blows up. The call delta has to rise from 0 to 1, and at short maturity that rise happens over a narrow range of spots, so its slope is steep. Vega has \(\sqrt{T}\) in the numerator and dies out.

Differentiating parity, with no dividend:

\[
\Delta_C - \Delta_P = 1, \qquad \Gamma_C = \Gamma_P, \qquad \mathrm{vega}_C = \mathrm{vega}_P,
\]

\[
\rho_C - \rho_P = K T e^{-rT}, \qquad \Theta_C - \Theta_P = - r K e^{-rT}.
\]

The numerical check is a central finite difference. For delta,

\[
\Delta \approx \frac{V(S+h) - V(S-h)}{2h}.
\]

Gamma is the second difference \((V(S+h) - 2V(S) + V(S-h))/h^2\).

## 4. The delta hedge

A market maker who sells an option is short the spot until the delta is covered.

The call is sold at the Black-Scholes price computed with the implied volatility \(\sigma_i\). The option position is worth \(-\Delta\) shares, with \(\Delta = N(d_1)\) also computed with \(\sigma_i\). Covering it means buying \(\Delta\) shares. The opening cash is

\[
\mathrm{cash}_0 = C - \Delta_0 S_0.
\]

That cash is often negative: the shares cost more than the premium, and the difference is borrowed. The marked portfolio is worth zero at inception, because the option was sold at its model price. The P&L appears later.

Between two dates the share position is constant and the cash is multiplied by \(e^{r \Delta t}\).

On the next date the delta is recomputed from the new spot, the time left, and again \(\sigma_i\). Realized volatility \(\sigma_r\) is used only to draw the spot path. The cash pays \((\Delta_{\mathrm{new}} - \Delta_{\mathrm{old}}) S\) for the shares traded.

At maturity the call payoff is paid, the shares are sold, and the cash that remains is the P&L. It is already a value at date \(T\), because the cash was compounded along the way. Delta is not recomputed on the expiry date: the time left is zero, and the position sold is the one left from the previous rebalance.

The spot path uses drift \(r\) and volatility \(\sigma_r\). Only the volatility differs between the pricing model and the realized path. The generator seed is fixed.

With a continuous hedge and \(\sigma_r = \sigma_i\), the P&L is zero: theta and gamma cancel, which is the Black-Scholes equation. With \(N\) rebalance dates a residual remains. When \(\sigma_r = \sigma_i\) its mean is close to 0 and its standard deviation falls like \(1/\sqrt{N}\).

## 5. Where the P&L comes from

Over a short interval,

\[
dV \approx \Theta \, dt + \Delta \, dS + \tfrac12 \Gamma \, (dS)^2.
\]

The shares cancel the \(dS\) term. The price was built so that

\[
\Theta \, dt + \tfrac12 \Gamma S^2 \sigma_i^2 \, dt
\]

is exactly the financing at rate \(r\). Replacing \((dS)^2\) by what actually happens, \(S^2 \sigma_r^2 dt\), the excess P&L of a long option is about

\[
\tfrac12 \Gamma S^2 \big(\sigma_r^2 - \sigma_i^2\big) \, dt.
\]

That is gamma scalping. This project does the opposite: the call is sold, so the position gamma is \(-\Gamma\),

\[
\mathbb{E}[\mathrm{P\&L}] \approx -\int_0^T \tfrac12 \Gamma_t S_t^2 \big(\sigma_r^2 - \sigma_i^2\big) \, dt.
\]

\(\sigma_r > \sigma_i\): the volatility was sold too cheap and the mean P&L is negative. \(\sigma_r < \sigma_i\): it was sold too rich and the mean P&L is positive.

Each trade can cost \(k \times |\text{shares traded}| \times S\), including the opening hedge and the final unwind. The standard deviation of the hedging error falls like \(1/\sqrt{N}\). Cumulative costs grow like \(\sqrt{N}\). The useful frequency is a compromise between the two.

## 6. Code map

| File | Role |
|---|---|
| `bs.py` | Call and put prices, put-call parity check |
| `greeks.py` | Delta, gamma, vega, theta, rho, finite-difference checks |
| `monte_carlo_check.py` | Closed form against a Monte Carlo confidence interval |
| `hedge_sim.py` | Spot paths and the P&L of the hedged short call |
| `experiments.ipynb` | Greeks, hedge error against \(N\), P&L against the vol gap, costs |
| `docs/notes.md` | These notes |
