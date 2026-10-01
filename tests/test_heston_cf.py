import numpy as np

from src.heston import HestonParams, simulate_heston_paths
from src.heston_cf import heston_call_cf
from src.monte_carlo import price_european_mc

P = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.09)


def test_cf_matches_monte_carlo():
    s, _ = simulate_heston_paths(100.0, P, 0.03, 0.0, 1.0, 400, 200_000, seed=7)
    for k in (90.0, 100.0, 115.0):
        mc = price_european_mc(s[:, -1], k, 1.0, 0.03, "call", antithetic=True)
        cf = heston_call_cf(100 * np.exp(0.03), np.array([k]), 1.0, P, np.exp(-0.03))[0]
        assert abs(cf - mc.price) < 4 * mc.standard_error + 0.05


def test_cf_reduces_to_black_when_no_vol_of_vol():
    from src.black76 import black76_price
    p = HestonParams(kappa=2.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    cf = heston_call_cf(100.0, np.array([95.0, 105.0]), 1.0, p)
    bs = [black76_price(100.0, k, 1.0, 0.2) for k in (95.0, 105.0)]
    assert np.allclose(cf, bs, atol=1e-3)


def test_qe_matches_cf_when_feller_violated():
    p = HestonParams(kappa=6.0, theta=0.045, xi=1.8, rho=-0.65, v0=0.02)
    f, d = 100 * np.exp(0.015), np.exp(-0.015)
    s, _ = simulate_heston_paths(100.0, p, 0.03, 0.0, 0.5, 25, 200_000, seed=5, scheme="qe")
    assert abs(s[:, -1].mean() / f - 1) < 3e-3
    for k in (85.0, 100.0, 115.0):
        mc = price_european_mc(s[:, -1], k, 0.5, 0.03, "call", antithetic=True)
        cf = heston_call_cf(f, np.array([k]), 0.5, p, d)[0]
        assert abs(cf - mc.price) < 4 * mc.standard_error + 0.02
