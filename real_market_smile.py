"""
real_market_smile.py

Pulls a REAL, live option chain (via Yahoo Finance) for a given ticker,
computes implied volatility from actual market prices using the implied_vol()
solver built in black_scholes.py, and compares the resulting real-world
volatility smile against the model's theoretical assumptions.

This is the piece that turns the project from "I implemented Black-Scholes
correctly" into "I used it to analyze real market behavior" -- the smile
here isn't simulated, it's whatever the market is actually pricing in right
now, for a real, currently-trading stock.

Requires: pip install yfinance

Author: Sa'Mara Roberts
"""

import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from black_scholes import implied_vol, bs_price


# A reasonable proxy for the current risk-free rate. This is not fetched
# live -- swap in the current ~3-month T-bill yield if you want it precise.
# As of early 2026 the short end of the curve has been roughly in this range.
RISK_FREE_RATE = 0.045


def get_real_option_chain(ticker_symbol, expiration_index=2):
    """
    Fetch a real, live option chain for a given ticker.

    expiration_index picks which upcoming expiration date to use (0 = nearest,
    which is often too close to expiry / too illiquid to give clean implied
    vols -- 2 or 3 usually lands on something with reasonable volume).

    Returns a dict with spot price, expiration date, time-to-expiry, and the
    raw calls dataframe from yfinance.
    """
    ticker = yf.Ticker(ticker_symbol)

    available_expirations = ticker.options
    if not available_expirations:
        raise ValueError(
            f"No option expirations found for {ticker_symbol}. "
            "The ticker symbol may be wrong, or it may not have listed options."
        )

    expiration_index = min(expiration_index, len(available_expirations) - 1)
    expiration_date = available_expirations[expiration_index]

    # Spot price: use the most recent close as a simple, robust proxy.
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Could not fetch recent price history for {ticker_symbol}.")
    spot = history["Close"].iloc[-1]

    chain = ticker.option_chain(expiration_date)
    calls = chain.calls

    days_to_expiry = (
        datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.now()
    ).days
    T = max(days_to_expiry, 1) / 365  # floor at 1 day to avoid T=0 errors

    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "expiration_date": expiration_date,
        "T": T,
        "calls": calls,
    }


def compute_real_implied_vols(data, moneyness_range=(0.85, 1.15)):
    """
    For each call option in the chain within the given moneyness window,
    back out the market-implied volatility from its actual traded price.

    Uses mid-price (bid+ask)/2 where available, since last-traded price can
    be stale, especially for less liquid strikes; falls back to lastPrice
    when bid/ask are both zero (illiquid or after-hours).

    Skips any strike where the solver can't find a valid implied vol -- this
    happens for deep ITM/OTM options with wide or stale quotes that violate
    no-arbitrage bounds, which is itself a realistic data-quality issue worth
    knowing about, not a bug.
    """
    S = data["spot"]
    T = data["T"]
    calls = data["calls"]

    lo, hi = moneyness_range
    strikes_out, market_prices_out, implied_vols_out = [], [], []
    skipped = 0

    for _, row in calls.iterrows():
        K = row["strike"]
        if not (lo * S <= K <= hi * S):
            continue

        bid, ask, last = row.get("bid", 0), row.get("ask", 0), row.get("lastPrice", 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last

        if mid <= 0:
            skipped += 1
            continue

        try:
            iv = implied_vol(mid, S, K, T, RISK_FREE_RATE, option_type="call")
        except ValueError:
            skipped += 1
            continue

        strikes_out.append(K)
        market_prices_out.append(mid)
        implied_vols_out.append(iv)

    return {
        "strikes": np.array(strikes_out),
        "market_prices": np.array(market_prices_out),
        "implied_vols": np.array(implied_vols_out),
        "skipped": skipped,
        "spot": S,
    }


def plot_real_smile(data, result, save_path="real_market_smile.png"):
    ticker = data["ticker"]
    expiration = data["expiration_date"]
    S = result["spot"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(result["strikes"], result["implied_vols"] * 100, 'o-',
             color='#c0392b', markersize=5, linewidth=1.5)
    ax.axvline(S, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f"Spot = ${S:.2f}")
    ax.set_xlabel("Strike Price ($)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"{ticker} Real Market Volatility Smile — Expiring {expiration}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    ticker_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"Fetching live option chain for {ticker_symbol}...")
    data = get_real_option_chain(ticker_symbol)
    print(f"Spot price: ${data['spot']:.2f}")
    print(f"Expiration: {data['expiration_date']}  ({data['T']*365:.0f} days out)")

    result = compute_real_implied_vols(data)
    n = len(result["strikes"])
    print(f"\nSolved implied vol for {n} strikes "
          f"({result['skipped']} skipped due to bad/missing quotes)")

    if n == 0:
        print("No valid strikes found -- try a more liquid ticker or a "
              "different expiration_index.")
        sys.exit(1)

    print("\nSample of the real market smile:")
    for k, iv in list(zip(result["strikes"], result["implied_vols"]))[:8]:
        print(f"  Strike ${k:>7.2f}  ->  IV {iv*100:.2f}%")

    plot_real_smile(data, result)
