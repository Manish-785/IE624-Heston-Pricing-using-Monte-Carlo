"""Calibrate Heston parameters to market implied volatilities."""
import numpy as np
from scipy.optimize import least_squares

from .black76 import black76_price, implied_volatility
from .heston import HestonParams
from .heston_cf import heston_call_cf

BOUNDS = ([0.05, 1e-3, 0.05, -0.99, 1e-3], [15.0, 1.0, 3.0, -0.01, 1.0])


def otm_quotes(chain, forward: float):
    """OTM options (puts below forward, calls above), converted to call-equivalent prices."""
    rows = chain[((chain.type == "put") & (chain.strike < forward)) |
                 ((chain.type == "call") & (chain.strike >= forward))]
    return rows.sort_values("strike")


def market_ivs(quotes, forward: float, T: float, disc: float):
    ivs = []
    for r in quotes.itertuples():
        try:
            ivs.append(implied_volatility(r.mid, forward, r.strike, T, disc, r.type))
        except (ValueError, RuntimeError):
            ivs.append(np.nan)
    return np.array(ivs)


def vega(forward, strike, T, vol, disc):
    h = 1e-4
    up = black76_price(forward, strike, T, vol + h, disc)
    dn = black76_price(forward, strike, T, max(vol - h, 1e-6), disc)
    return (up - dn) / (vol + h - max(vol - h, 1e-6))


def calibrate(slices: list[dict]) -> tuple[HestonParams, float]:
    """slices: dicts with F, D, T, K, iv (market IVs). Minimises vega-weighted price error,
    which approximates an implied-vol-space fit. Returns (params, RMSE in vol points)."""
    prepped = []
    for s in slices:
        ok = np.isfinite(s["iv"])
        K, iv = s["K"][ok], s["iv"][ok]
        w = np.array([vega(s["F"], k, s["T"], v, s["D"]) for k, v in zip(K, iv)])
        mkt = np.array([black76_price(s["F"], k, s["T"], v, s["D"], "call") for k, v in zip(K, iv)])
        prepped.append((s["F"], s["D"], s["T"], K, mkt, np.maximum(w, 1e-8)))

    def resid(x):
        p = HestonParams(*x)
        out = []
        for F, D, T, K, mkt, w in prepped:
            out.append((heston_call_cf(F, K, T, p, D) - mkt) / w)
        return np.concatenate(out)

    best = None
    for x0 in ([2.0, 0.04, 0.6, -0.7, 0.04], [5.0, 0.06, 1.0, -0.8, 0.02], [1.0, 0.03, 0.4, -0.5, 0.05]):
        sol = least_squares(resid, x0, bounds=BOUNDS, x_scale=[1, 0.05, 0.5, 0.5, 0.05])
        if best is None or sol.cost < best.cost:
            best = sol
    rmse = float(np.sqrt(np.mean(best.fun**2)))
    return HestonParams(*best.x), rmse
