"""
black_scholes.py

A from-scratch implementation of the Black-Scholes-Merton option pricing model,
including analytical Greeks and a numerical implied volatility solver.

Author: Sa'Mara Roberts
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """
    Compute d1 and d2, the two standardized normal arguments used throughout
    the Black-Scholes formula.

    S : spot price of the underlying
    K : strike price
    T : time to expiration, in years
    r : risk-free interest rate (annualized, continuously compounded)
    sigma : volatility of the underlying (annualized)
    q : continuous dividend yield (default 0)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must both be positive.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes-Merton price of a European call or put.

    Returns the theoretical option price.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Analytical Greeks for a European option under Black-Scholes.

    Returns a dict with delta, gamma, vega, theta, rho.
    Theta is expressed per calendar day (divided by 365) since that's how
    traders actually think about time decay day-to-day.
    Vega is expressed per 1% change in vol (divided by 100).
    Rho is expressed per 1% change in rates (divided by 100).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                 vol_lower=1e-4, vol_upper=5.0):
    """
    Solve for the implied volatility that makes the Black-Scholes price equal
    to the observed market price, using Brent's method (bisection-style root
    finding, robust and doesn't need a derivative).

    This is the direction real trading desks actually use the model in
    practice: prices are set by supply and demand in the market, and traders
    back out an implied vol to compare options across strikes/expiries on
    a common scale.
    """

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    # Sanity check: make sure a root actually exists in the bracket.
    lo, hi = objective(vol_lower), objective(vol_upper)
    if lo * hi > 0:
        raise ValueError(
            "No implied vol found in [%.4f, %.2f] for the given market price. "
            "Check that market_price is within a valid arbitrage-free range."
            % (vol_lower, vol_upper)
        )

    return brentq(objective, vol_lower, vol_upper, xtol=1e-8)


if __name__ == "__main__":
    # Quick sanity check when run directly.
    S, K, T, r, sigma = 100, 100, 0.5, 0.04, 0.20

    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    greeks = bs_greeks(S, K, T, r, sigma, "call")

    print(f"At-the-money call price: {call_price:.4f}")
    print(f"At-the-money put price:  {put_price:.4f}")
    print("Call Greeks:", {k: round(v, 4) for k, v in greeks.items()})

    # Put-call parity check: C - P should equal S*e^-qT - K*e^-rT
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-call parity check: {parity_lhs:.4f} vs {parity_rhs:.4f} "
          f"(diff = {abs(parity_lhs - parity_rhs):.2e})")

    # Recover implied vol from the price we just generated -- should return
    # ~0.20 exactly, confirming the solver is self-consistent with the pricer.
    recovered_vol = implied_vol(call_price, S, K, T, r, "call")
    print(f"\nRecovered implied vol from price: {recovered_vol:.6f} "
          f"(input was {sigma})")
