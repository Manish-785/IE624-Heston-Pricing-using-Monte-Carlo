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


def drop_iv_outliers(strikes: np.ndarray, iv: np.ndarray, window: int = 7, n_mad: float = 5.0,
                     floor: float = 0.005) -> np.ndarray:
    """Boolean mask of quotes consistent with their neighbours (rolling-median test).

    Stale or mispriced mids show up as isolated spikes in the smile; they are removed
    before calibration so a handful of bad quotes cannot drag the fit.
    """
    order = np.argsort(strikes)
    x = iv[order]
    ok = np.isfinite(x)
    keep = np.zeros(len(x), dtype=bool)
    half = window // 2
    for i in range(len(x)):
        nb = x[max(0, i - half): i + half + 1]
        nb = nb[np.isfinite(nb)]
        med = np.median(nb)
        mad = max(np.median(np.abs(nb - med)) * 1.4826, floor)
        keep[i] = ok[i] and abs(x[i] - med) <= n_mad * mad
    out = np.zeros(len(x), dtype=bool)
    out[order] = keep
    return out


def build_slice(chain, spot: float, days: int) -> dict:
    """One expiry's market smile: parity forward/discount, OTM mid IVs with bad quotes removed."""
    from .market_data import forward_and_discount
    c = chain[chain.days == days]
    T = days / 365.0
    F, D = forward_and_discount(c, spot)
    q = otm_quotes(c, F)
    iv = market_ivs(q, F, T, D)
    keep = np.isfinite(iv) & (iv > 0.02)
    keep[keep] = drop_iv_outliers(q.strike.values[keep], iv[keep])
    return dict(days=days, T=T, F=F, D=D, K=q.strike.values[keep], iv=iv[keep],
                bid=q.bid.values[keep], ask=q.ask.values[keep], kind=q.type.values[keep])


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
        if len(K) < 5:
            raise ValueError(f"Slice T={s['T']:.3f} has only {len(K)} usable quotes; need >= 5.")
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
    if not np.isfinite(rmse):
        raise RuntimeError("Calibration produced a non-finite objective.")
    at_bound = [n for n, x, lo, hi in zip(("kappa", "theta", "xi", "rho", "v0"), best.x, *BOUNDS)
                if min(x - lo, hi - x) < 1e-6 * (hi - lo)]
    if at_bound:
        print(f"WARNING: calibrated parameter(s) hit a bound: {', '.join(at_bound)}")
    return HestonParams(*best.x), rmse
