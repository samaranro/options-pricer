"""
test_black_scholes.py

Validates the pricer against known analytical properties rather than just
checking that the code runs. Run with:  python3 -m pytest test_black_scholes.py -v
"""

import numpy as np
import pytest
from black_scholes import bs_price, bs_greeks, implied_vol


S, K, T, r, sigma = 100, 100, 0.5, 0.04, 0.20


def test_put_call_parity():
    """C - P = S*e^-qT - K*e^-rT must hold exactly (up to floating point)."""
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-9


def test_prices_are_positive():
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    assert call > 0
    assert put > 0


def test_deep_itm_call_approaches_intrinsic():
    """A deep in-the-money call should behave like S - K*e^-rT (its intrinsic
    value under the forward), since the optionality has almost no value left."""
    deep_itm_call = bs_price(S=200, K=100, T=T, r=r, sigma=sigma, option_type="call")
    intrinsic = 200 - 100 * np.exp(-r * T)
    assert abs(deep_itm_call - intrinsic) < 0.5


def test_deep_otm_call_approaches_zero():
    otm_call = bs_price(S=100, K=300, T=T, r=r, sigma=sigma, option_type="call")
    assert otm_call < 0.01


def test_call_delta_bounded():
    """Call delta must always be in [0, 1]."""
    for spot in [50, 80, 100, 120, 200]:
        greeks = bs_greeks(spot, K, T, r, sigma, "call")
        assert 0 <= greeks["delta"] <= 1


def test_put_delta_bounded():
    """Put delta must always be in [-1, 0]."""
    for spot in [50, 80, 100, 120, 200]:
        greeks = bs_greeks(spot, K, T, r, sigma, "put")
        assert -1 <= greeks["delta"] <= 0


def test_gamma_positive_and_symmetric_role():
    """Gamma must always be positive for both calls and puts (long options
    are always long gamma under Black-Scholes)."""
    call_greeks = bs_greeks(S, K, T, r, sigma, "call")
    put_greeks = bs_greeks(S, K, T, r, sigma, "put")
    assert call_greeks["gamma"] > 0
    # Gamma is identical for calls and puts at the same strike/expiry --
    # a useful fact to know cold in an interview.
    assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-9


def test_vega_positive():
    """Vega must always be positive: more volatility always adds option value."""
    greeks = bs_greeks(S, K, T, r, sigma, "call")
    assert greeks["vega"] > 0


def test_implied_vol_recovers_input():
    """Round-trip: price at a known vol, then solve for implied vol, should
    return (approximately) the original vol."""
    true_vol = 0.35
    price = bs_price(S, K, T, r, true_vol, "call")
    recovered = implied_vol(price, S, K, T, r, "call")
    assert abs(recovered - true_vol) < 1e-6


def test_implied_vol_raises_on_impossible_price():
    """A price above the theoretical max (S) should raise, not silently
    return garbage -- this matters because bad market data happens."""
    with pytest.raises(ValueError):
        implied_vol(market_price=S * 2, S=S, K=K, T=T, r=r, option_type="call")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
