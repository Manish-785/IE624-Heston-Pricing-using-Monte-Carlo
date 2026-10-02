"""How each Heston parameter reshapes the implied-volatility smile.

One parameter is varied at a time around a base set (the SPX calibration if available,
otherwise the illustrative demo parameters). Curves are semi-analytic; Monte Carlo (QE)
markers on the base case confirm the simulator agrees.
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.heston import HestonParams
from src.smile import heston_smile_cf, heston_smile_mc

OUT = Path(__file__).resolve().parents[1] / "outputs"
S0, R, Q = 100.0, 0.03, 0.0
MC_PATHS = 100_000
MATURITIES = (0.25, 1.0)
GRID = {  # parameter -> (values, label)
    "rho":   ([-0.9, -0.65, -0.3, 0.0], "rho: spot-vol correlation (skew)"),
    "xi":    ([0.3, 0.8, 1.4, 2.0], "xi: vol of vol (curvature / wings)"),
    "v0":    ([0.01, 0.02, 0.04, 0.08], "v0: initial variance (short-end level)"),
    "theta": ([0.02, 0.03, 0.045, 0.07], "theta: long-run variance (long-end level)"),
    "kappa": ([0.5, 1.5, 3.0, 6.5], "kappa: mean reversion (term structure)"),
}


def base_params() -> HestonParams:
    path = OUT / "spx_calibrated_params.csv"
    if path.exists():
        return HestonParams(**pd.read_csv(path).iloc[0].to_dict())
    return HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.09)


def main():
    OUT.mkdir(exist_ok=True)
    base = base_params()
    print(f"Base parameters: {base}")
    rows = []
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    m = np.linspace(-0.3, 0.2, 31)  # log-moneyness grid ln(K/F)
    for T in MATURITIES:
        disc = np.exp(-R * T)
        fwd = S0 * np.exp((R - Q) * T)
        strikes = fwd * np.exp(m)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (name, (values, label)) in zip(axes.ravel(), GRID.items()):
            shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(values)))
            for val, c in zip(values, shades):
                p = replace(base, **{name: val})
                iv = heston_smile_mc(S0, fwd, strikes, T, p, disc, n_paths=MC_PATHS, seed=7)  # common random numbers
                ax.plot(m, iv * 100, color=c, lw=2.2, label=f"{name} = {val:g}")
                atm = float(np.interp(0.0, m, iv))
                rows.append(dict(maturity=T, param=name, value=val, atm_vol=atm,
                                 skew=float((np.interp(-0.1, m, iv) - np.interp(0.1, m, iv)) / 0.2),
                                 curvature=float(np.interp(-0.2, m, iv) + np.interp(0.2, m, iv) - 2 * atm)))
            ax.set_title(label, fontweight="bold", fontsize=11)
            ax.set_xlabel("log-moneyness ln(K/F)")
            ax.set_ylabel("Implied vol (%)")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=9)
        info = axes.ravel()[-1]
        info.axis("off")
        lines = [f"Maturity T = {T:g}y", "", "Base parameters", "(one varied per panel,", "others held fixed):", "",
                 f"kappa = {base.kappa:.2f}", f"theta = {base.theta:.3f}", f"xi    = {base.xi:.2f}",
                 f"rho   = {base.rho:.2f}", f"v0    = {base.v0:.3f}"]
        info.text(0, 0.95, chr(10).join(lines), va="top", fontsize=12, family="monospace")
        fig.suptitle(f"Effect of each Heston parameter on the implied-vol smile (Monte Carlo, T = {T:g}y)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"heston_param_sensitivity_T{T:g}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "heston_param_sensitivity.csv", index=False)

    # Monte Carlo cross-check of the base case (reported, not plotted)
    T = 1.0
    fwd, disc = S0 * np.exp((R - Q) * T), np.exp(-R * T)
    mm = np.linspace(-0.3, 0.2, 11)
    mc = heston_smile_mc(S0, fwd, fwd * np.exp(mm), T, base, disc, n_paths=100_000)
    cf = heston_smile_cf(fwd, fwd * np.exp(mm), T, base, disc)
    print(f"Monte Carlo vs semi-analytic, base case T=1: max |diff| = {100*np.nanmax(np.abs(mc-cf)):.2f} vol pts")


if __name__ == "__main__":
    main()
