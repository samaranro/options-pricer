"""
vol_smile.py

Demonstrates the "volatility smile" -- the well-known empirical pattern where
implied volatility varies by strike price, even though Black-Scholes assumes
a single constant volatility for all strikes.

Since this project doesn't pull live market data, we simulate a realistic
smile using a simple parametric model (an SVI-style skew), then run OUR
implied_vol() solver on the simulated market prices to recover the smile --
proving the whole pipeline (pricing -> market prices -> implied vol solving)
works end to end, the same way a trading desk would use it on real quotes.

Author: Sa'Mara Roberts
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import bs_price, implied_vol


def simulate_market_vol(strike, spot, atm_vol=0.20, skew=-0.15, curvature=0.10):
    """
    A simple parametric smile: vol rises as strikes move away from the money,
    with negative skew (puts trade at higher implied vol than calls) -- the
    classic equity index pattern that's been persistent since the 1987 crash,
    when the market started pricing in crash/tail risk asymmetrically.

    This is a simulation for demonstration purposes, not a calibrated model.
    """
    moneyness = np.log(strike / spot)
    return atm_vol + skew * moneyness + curvature * moneyness ** 2


def build_smile(spot=100, r=0.04, T=0.25, strikes=None):
    """
    For a range of strikes:
      1. Assign each strike a "true" simulated market vol (the smile we're
         trying to recover).
      2. Price a call at that vol to get a synthetic "market price".
      3. Feed that market price back into our implied_vol() solver.
      4. Compare recovered vol to the true simulated vol.

    Returns a dict of arrays: strikes, true_vols, market_prices, implied_vols.
    """
    if strikes is None:
        strikes = np.linspace(70, 130, 25)

    true_vols = np.array([simulate_market_vol(k, spot) for k in strikes])
    market_prices = np.array([
        bs_price(spot, k, T, r, v, "call") for k, v in zip(strikes, true_vols)
    ])
    recovered_vols = np.array([
        implied_vol(p, spot, k, T, r, "call")
        for p, k in zip(market_prices, strikes)
    ])

    return {
        "strikes": strikes,
        "true_vols": true_vols,
        "market_prices": market_prices,
        "implied_vols": recovered_vols,
        "spot": spot,
    }


def plot_smile(data, save_path="vol_smile.png"):
    strikes = data["strikes"]
    spot = data["spot"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: the smile itself ---
    ax = axes[0]
    ax.plot(strikes, data["true_vols"] * 100, 'o', color='#2c3e50',
             markersize=5, label="Simulated market vol", zorder=3)
    ax.plot(strikes, data["implied_vols"] * 100, '-', color='#c0392b',
             linewidth=2, label="Recovered via implied_vol() solver", zorder=2)
    ax.axvline(spot, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f"Spot = {spot}")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Volatility Smile: Simulated vs. Recovered")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Right panel: option price curve for context ---
    ax2 = axes[1]
    ax2.plot(strikes, data["market_prices"], color='#27ae60', linewidth=2)
    ax2.axvline(spot, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel("Strike Price")
    ax2.set_ylabel("Call Option Price ($)")
    ax2.set_title("Corresponding Call Prices Across Strikes")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    max_diff = np.max(np.abs(data["true_vols"] - data["implied_vols"]))
    print(f"Max difference between simulated and recovered vol: {max_diff:.2e} "
          f"(confirms the solver recovers the smile essentially exactly)")


if __name__ == "__main__":
    data = build_smile()
    plot_smile(data)
