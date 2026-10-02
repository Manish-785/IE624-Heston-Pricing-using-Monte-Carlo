import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from src.calibration import drop_iv_outliers
from src.heston import HestonParams, simulate_heston_paths
from src.heston_cf import _log_cf, heston_call_cf
from src.market_data import forward_and_discount
from src.smile import heston_smile_cf, heston_smile_mc


def _reference_call(F, K, T, p):
    k = np.log(F / K)
    f = lambda u: np.real(np.exp(1j * u * k) * _log_cf(np.array([u - 0.5j]), T, p)[0]) / (u * u + 0.25)
    return F - np.sqrt(F * K) / np.pi * quad(f, 0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-12)[0]


@pytest.mark.parametrize("T", [1 / 52, 0.1, 1.0, 3.0])
@pytest.mark.parametrize("p", [HestonParams(2, .04, .6, -.7, .09), HestonParams(6.4, .044, 1.83, -.66, .017)])
def test_cf_quadrature_accurate_across_maturities(T, p):
    for K in (70.0, 100.0, 130.0):
        assert abs(heston_call_cf(100.0, np.array([K]), T, p)[0] - _reference_call(100.0, K, T, p)) < 5e-4


def test_cf_prices_respect_no_arbitrage_bounds():
    p = HestonParams(6.4, .044, 1.83, -.66, .017)
    K = np.linspace(50, 200, 40)
    c = heston_call_cf(100.0, K, 0.05, p, 0.99)
    assert np.all(c >= 0.99 * np.maximum(100 - K, 0) - 1e-12) and np.all(c <= 99.0 + 1e-12)


def test_qe_emits_no_numpy_warnings_for_extreme_params():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for p in (HestonParams(15, .04, 3, -.99, .001), HestonParams(.1, .01, 3, -.9, .5)):
            s, v = simulate_heston_paths(100.0, p, .03, 0, 0.5, 25, 4000, scheme="qe")
            assert np.all(np.isfinite(s)) and np.all(v >= 0)


def test_mc_smile_matches_semi_analytic_smile():
    p = HestonParams(6.4, .044, 1.83, -.66, .017)
    T, d = 0.5, np.exp(-0.03 * 0.5)
    F = 100 * np.exp(0.015)
    K = F * np.exp(np.linspace(-0.2, 0.1, 6))
    mc = heston_smile_mc(100.0, F, K, T, p, d, n_paths=100_000)
    cf = heston_smile_cf(F, K, T, p, d)
    assert np.nanmax(np.abs(mc - cf)) < 0.006  # 0.6 vol points


def test_outlier_filter_removes_isolated_spike():
    k = np.linspace(80, 120, 41)
    iv = 0.2 - 0.001 * (k - 100)
    iv[20] += 0.08
    keep = drop_iv_outliers(k, iv)
    assert not keep[20] and keep.sum() >= 38


def test_forward_extraction_ignores_bad_quote_and_rejects_garbage():
    K = np.arange(90.0, 111.0)
    D, F = 0.97, 101.0
    c = np.maximum(D * (F - K), 0) + 3.0
    p = c - D * (F - K)
    rows = [dict(strike=k, type="call", mid=a) for k, a in zip(K, c)] + \
           [dict(strike=k, type="put", mid=b) for k, b in zip(K, p)]
    df = pd.DataFrame(rows)
    df.loc[(df.type == "call") & (df.strike == 100.0), "mid"] += 5.0  # stale quote
    f, d = forward_and_discount(df, spot=100.0)
    assert abs(f - F) < 0.05 and abs(d - D) < 1e-3
    df["mid"] = np.where(df.type == "call", 5.0, 5.0)  # C-P = 0 -> no valid parity
    with pytest.raises(ValueError):
        forward_and_discount(df, spot=100.0)
