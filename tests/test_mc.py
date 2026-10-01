import numpy as np
import pytest

from src.monte_carlo import price_european_mc
from src.heston import HestonParams, simulate_heston_paths
from src.black76 import black76_price


def test_call_and_antithetic_error():
    result = price_european_mc(np.array([90, 110, 95, 105]), 100, 1, 0, "call", antithetic=True)
    assert result.price == pytest.approx(3.75)
    assert result.standard_error >= 0


def test_invalid_terminal_prices_rejected():
    with pytest.raises(ValueError):
        price_european_mc(np.array([100, np.nan]), 100, 1, .02)


def test_constant_variance_mc_matches_black76_within_sampling_error():
    s0, r, q, t, k, vol = 100.0, .02, .01, 1.0, 102.0, .20
    spots, _ = simulate_heston_paths(s0, HestonParams(2, vol**2, 0, -.5, vol**2),
                                     r, q, t, 64, 40_000, seed=17)
    mc = price_european_mc(spots[:, -1], k, t, r, "call", antithetic=True)
    fwd = s0 * np.exp((r-q)*t)
    benchmark = black76_price(fwd, k, t, vol, np.exp(-r*t), "call")
    assert abs(mc.price - benchmark) < 4 * mc.standard_error
