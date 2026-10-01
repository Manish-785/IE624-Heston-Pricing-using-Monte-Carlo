import numpy as np
import pytest

from src.black76 import black76_price, implied_volatility


def test_prices_parity_and_iv_round_trip():
    f, k, t, vol, d = 105.0, 100.0, 0.75, 0.27, 0.98
    call = black76_price(f, k, t, vol, d, "call")
    put = black76_price(f, k, t, vol, d, "put")
    assert call - put == pytest.approx(d * (f - k))
    assert implied_volatility(call, f, k, t, d, "call") == pytest.approx(vol, abs=1e-8)
    assert implied_volatility(put, f, k, t, d, "put") == pytest.approx(vol, abs=1e-8)


def test_invalid_price_is_rejected():
    with pytest.raises(ValueError):
        implied_volatility(200, 100, 100, 1)


def test_call_decreases_with_strike():
    prices = [black76_price(100, k, 1, 0.2) for k in (80, 100, 120)]
    assert np.all(np.diff(prices) < 0)
