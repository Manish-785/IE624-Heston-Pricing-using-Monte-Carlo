"""Calibrate Heston to live SPX options and compare market vs fitted smiles."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import build_slice, calibrate
from src.market_data import clean_chain, fetch_spx_chain, forward_and_discount
from src.smile import _invert as _iv, heston_smile_cf, heston_smile_mc

OUT = Path(__file__).resolve().parents[1] / "outputs"
N_EXPIRIES = 4          # calibrate on this many maturities spread over the term structure
MC_PATHS = 200_000
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
        sl = build_slice(chain, spot, days)
        slices.append(sl)
        F, D, T = sl["F"], sl["D"], sl["T"]
        print(f"  {days:4d}d: F={F:.1f}  r_implied={-np.log(D)/T:.3%}  n={len(sl['K'])}")

    params, rmse = calibrate(slices)
    print(f"\nCalibrated: {params}\n  vega-weighted RMSE (price/vega, ~vol units): {rmse:.4f}")
    print(f"  Feller 2*kappa*theta - xi^2 = {2*params.kappa*params.theta - params.xi**2:+.4f}")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, RED = "#1f77b4", "#d62728"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
    rows = []
    for ax, ax2, s in zip(axes.ravel(), axes2.ravel(), slices):
        F, D, T = s["F"], s["D"], s["T"]
        kgrid = np.linspace(s["K"].min(), s["K"].max(), 150)
        fit_iv = heston_smile_cf(F, kgrid, T, params, D)
        fit_at_mkt = heston_smile_cf(F, s["K"], T, params, D, kinds=s["kind"])
        lo = np.array([_iv(b, F, k, T, D, kd) for b, k, kd in zip(s["bid"], s["K"], s["kind"])])
        hi = np.array([_iv(a, F, k, T, D, kd) for a, k, kd in zip(s["ask"], s["K"], s["kind"])])
        err = fit_at_mkt - s["iv"]
        m = np.log(s["K"] / F)
        # Thin dense chains to ~35 evenly spaced quotes for display (all quotes still used in the fit)
        show = np.unique(np.linspace(0, len(m) - 1, min(35, len(m))).round().astype(int))
        # Primary curve: Monte Carlo (QE) at the displayed strikes, all from one set of paths
        mc_iv = heston_smile_mc(spot, F, s["K"][show], T, params, D, MC_PATHS, MC_STEPS_PER_YEAR,
                                kinds=s["kind"][show])
        mc_err = mc_iv - s["iv"][show]

        ax.plot(m[show], s["iv"][show] * 100, "o", ms=5, color=BLUE, alpha=0.7, label="Market (mid)")
        ax.plot(m[show], mc_iv * 100, "-s", ms=4, color=RED, lw=2, label="Heston Monte Carlo")
        ax.plot(np.log(kgrid / F), fit_iv * 100, "--", color="black", lw=1.2, label="Heston semi-analytic (reference)")
        ax.set_title(f"{s['days']} days to expiry", fontweight="bold")
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("Implied vol (%)")
        ax.grid(alpha=0.25)

        ax2.axhline(0, color="0.4", lw=1)
        ax2.plot(m[show], mc_err * 100, "-s", ms=4, color=RED, lw=1.8, label="Monte Carlo - market")
        ax2.plot(m[show], err[show] * 100, "--", color="black", lw=1.2, label="Semi-analytic - market")
        ax2.set_title(f"{s['days']} days: MC RMSE {100*np.sqrt(np.nanmean(mc_err**2)):.2f} vol pts", fontweight="bold")
        ax2.set_xlabel("log-moneyness ln(K/F)")
        ax2.set_ylabel("Heston - market (vol pts)")
        ax2.grid(alpha=0.25)
        inside = np.nanmean((fit_at_mkt >= lo) & (fit_at_mkt <= hi))
        rows.append(dict(days=s["days"], n_quotes=len(s["K"]),
                         rmse_vol_pts=100 * np.sqrt(np.nanmean(err**2)),
                         mc_rmse_vol_pts=100 * np.sqrt(np.nanmean(mc_err**2)),
                         mean_err_vol_pts=100 * np.nanmean(err),
                         max_abs_err_vol_pts=100 * np.nanmax(np.abs(err)),
                         pct_within_bid_ask=100 * inside))
        pd.DataFrame(dict(strike=s["K"], type=s["kind"], market_iv=s["iv"], bid_iv=lo, ask_iv=hi,
                          fit_iv=fit_at_mkt)).to_csv(OUT / f"spx_smile_{s['days']}d.csv", index=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    axes2[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"SPX implied-volatility smile: market vs Heston ({asof})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "spx_smile_market_vs_heston.png", dpi=150)
    fig2.suptitle("Heston minus market implied vol, Monte Carlo (positive = model above market)", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.savefig(OUT / "spx_smile_difference.png", dpi=150)

    summary = pd.DataFrame(rows)
    print("\nFit error by maturity (vol points):\n", summary.round(3).to_string(index=False))
    summary.to_csv(OUT / "spx_fit_summary.csv", index=False)
    pd.DataFrame([params.__dict__]).to_csv(OUT / "spx_calibrated_params.csv", index=False)


if __name__ == "__main__":
    main()
