"""
greeks_analysis.py

Visualizes how the Greeks behave as the underlying moves and as expiration
approaches. This is the kind of intuition-building traders actually rely on:
knowing THE SHAPE of your risk, not just a single number.

Author: Sa'Mara Roberts
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import bs_greeks


def greeks_vs_spot(K=100, T=0.25, r=0.04, sigma=0.20, spot_range=None):
    if spot_range is None:
        spot_range = np.linspace(60, 140, 200)

    results = {g: [] for g in ["delta", "gamma", "vega", "theta"]}
    for S in spot_range:
        greeks = bs_greeks(S, K, T, r, sigma, "call")
        for g in results:
            results[g].append(greeks[g])

    return spot_range, {g: np.array(v) for g, v in results.items()}


def plot_greeks_vs_spot(save_path="greeks_vs_spot.png"):
    spot_range, results = greeks_vs_spot()
    K = 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = {
        "delta": "Delta (directional exposure)",
        "gamma": "Gamma (delta's rate of change)",
        "vega": "Vega (sensitivity to vol, per 1%)",
        "theta": "Theta (daily time decay, per day)",
    }
    colors = {"delta": "#2980b9", "gamma": "#c0392b",
              "vega": "#27ae60", "theta": "#8e44ad"}

    for ax, (g, title) in zip(axes.flat, titles.items()):
        ax.plot(spot_range, results[g], color=colors[g], linewidth=2)
        ax.axvline(K, color='gray', linestyle='--', linewidth=1, alpha=0.6,
                    label=f"Strike = {K}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Spot Price")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Call Option Greeks vs. Underlying Price (K=100, T=3mo, σ=20%)",
                  fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    # A couple of the intuitions worth being able to say out loud in an
    # interview -- printed here so running the script doubles as a study aid.
    atm_idx = np.argmin(np.abs(spot_range - K))
    print(f"\nAt-the-money (S=K={K}):")
    print(f"  Delta ≈ {results['delta'][atm_idx]:.3f}  "
          f"(near 0.5 -- classic ATM behavior)")
    print(f"  Gamma is at its MAXIMUM here ({results['gamma'][atm_idx]:.4f}) "
          f"-- delta changes fastest when you're at the money")
    print(f"  Theta is at its most NEGATIVE here "
          f"({results['theta'][atm_idx]:.4f}/day) -- ATM options decay "
          f"fastest since they have the most extrinsic value to lose")


if __name__ == "__main__":
    plot_greeks_vs_spot()
