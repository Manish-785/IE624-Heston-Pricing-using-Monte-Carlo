"""Calibrate Heston to live SPX options and compare market vs fitted smiles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black76 import implied_volatility
from src.calibration import calibrate, market_ivs, otm_quotes
from src.heston import simulate_heston_paths
from src.heston_cf import heston_call_cf
from src.market_data import clean_chain, fetch_spx_chain, forward_and_discount
from src.monte_carlo import price_european_mc

OUT = Path(__file__).resolve().parents[1] / "outputs"
N_EXPIRIES = 4          # calibrate on this many maturities spread over the term structure
MC_PATHS = 100_000
MC_STEPS_PER_YEAR = 100


def pick_expiries(chain, n):
    counts = chain.groupby("days").size()
    days = counts[counts >= 20].index.values
    days = days[days >= 30]
    idx = np.unique(np.linspace(0, len(days) - 1, n).round().astype(int))
    return list(days[idx])


def main():
    OUT.mkdir(exist_ok=True)
    raw, spot, asof = fetch_spx_chain()
    chain = clean_chain(raw, spot)
    print(f"As of {asof}: SPX spot {spot:.2f}, {len(chain)} clean quotes")

    slices = []
    for days in pick_expiries(chain, N_EXPIRIES):
        c = chain[chain.days == days]
        T = days / 365.0
        F, D = forward_and_discount(c, spot)
        q = otm_quotes(c, F)
        iv = market_ivs(q, F, T, D)
        keep = np.isfinite(iv) & (iv > 0.02)
        slices.append(dict(days=days, T=T, F=F, D=D, K=q.strike.values[keep], iv=iv[keep],
                           bid=q.bid.values[keep], ask=q.ask.values[keep],
                           kind=q.type.values[keep]))
        print(f"  {days:4d}d: F={F:.1f}  r_implied={-np.log(D)/T:.3%}  n={keep.sum()}")

    params, rmse = calibrate(slices)
    print(f"\nCalibrated: {params}\n  vega-weighted RMSE (price/vega, ~vol units): {rmse:.4f}")
    print(f"  Feller 2*kappa*theta - xi^2 = {2*params.kappa*params.theta - params.xi**2:+.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    rows = []
    for ax, s in zip(axes.ravel(), slices):
        F, D, T = s["F"], s["D"], s["T"]
        # Fitted smile (semi-analytic) on a fine grid
        kgrid = np.linspace(s["K"].min(), s["K"].max(), 120)
        cgrid = heston_call_cf(F, kgrid, T, params, D)
        fit_iv = []
        for k, cp in zip(kgrid, cgrid):
            try:
                fit_iv.append(implied_volatility(cp, F, k, T, D, "call"))
            except (ValueError, RuntimeError):
                fit_iv.append(np.nan)
        # Monte Carlo cross-check at the same strikes (OTM options, as for the market)
        steps = max(25, int(MC_STEPS_PER_YEAR * T))
        r = -np.log(D) / T
        q_div = r - np.log(F / spot) / T  # so that spot*exp((r-q)T) = F
        st, _ = simulate_heston_paths(spot, params, r, q_div, T, steps, MC_PATHS, seed=11, scheme="qe")
        sel = slice(None, None, 2)
        mc_iv = []
        for k, kind in zip(s["K"][sel], s["kind"][sel]):
            res = price_european_mc(st[:, -1], k, T, r, kind, antithetic=True)
            try:
                mc_iv.append(implied_volatility(res.price, F, k, T, D, kind))
            except (ValueError, RuntimeError):
                mc_iv.append(np.nan)

        m = np.log(s["K"] / F)
        ax.plot(m, s["iv"] * 100, "o", ms=3.5, color="#1f77b4", label="Market (mid)")
        ax.plot(np.log(kgrid / F), np.array(fit_iv) * 100, "-", color="#d62728", lw=2, label="Heston fit (semi-analytic)")
        ax.plot(np.log(s["K"][sel] / F), np.array(mc_iv) * 100, "x", color="black", ms=5, label="Heston Monte Carlo")
        ax.set_title(f"{s['days']} days (T={T:.2f}y)")
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("Implied vol (%)")
        ax.grid(alpha=0.3)
        fit_at_mkt = []
        cm = heston_call_cf(F, s["K"], T, params, D)
        for k, cp, kind in zip(s["K"], cm, s["kind"]):
            pr = cp if kind == "call" else cp - D * (F - k)
            try:
                fit_at_mkt.append(implied_volatility(pr, F, k, T, D, kind))
            except (ValueError, RuntimeError):
                fit_at_mkt.append(np.nan)
        err = np.array(fit_at_mkt) - s["iv"]
        rows.append(dict(days=s["days"], rmse_vol_pts=100 * np.sqrt(np.nanmean(err**2)),
                         mean_err_vol_pts=100 * np.nanmean(err),
                         ))
        pd.DataFrame(dict(strike=s["K"], type=s["kind"], market_iv=s["iv"], fit_iv=fit_at_mkt)
                     ).to_csv(OUT / f"spx_smile_{s['days']}d.csv", index=False)
    axes[0, 0].legend()
    fig.suptitle(f"SPX implied-volatility smile: market vs calibrated Heston ({asof}, spot {spot:.0f})")
    fig.tight_layout()
    fig.savefig(OUT / "spx_smile_market_vs_heston.png", dpi=150)

    summary = pd.DataFrame(rows)
    print("\nFit error by maturity (vol points):\n", summary.round(3).to_string(index=False))
    summary.to_csv(OUT / "spx_fit_summary.csv", index=False)
    pd.DataFrame([params.__dict__]).to_csv(OUT / "spx_calibrated_params.csv", index=False)


if __name__ == "__main__":
    main()
