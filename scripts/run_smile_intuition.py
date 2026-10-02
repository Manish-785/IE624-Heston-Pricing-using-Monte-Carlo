"""Part 3: why a volatility curve must exist -- illustrated on the S&P 500.

Black-Scholes prices options with lognormal returns (one constant sigma). Real index returns
are left-skewed with fat tails (crashes, leverage effect, demand for downside protection).
Out-of-the-money options pay off only in the tails, so their prices are set by tail
probabilities; a model with a fatter left tail prices OTM puts higher than lognormal, and
inverting Black-Scholes on those higher prices gives a higher implied vol at low strikes.

Uses the Heston parameters calibrated to SPX (run scripts/run_spx.py first) at one expiry.
Figure: (a) return densities, (b) same on a log scale to expose the tails, (c) market and
model implied vols against the flat Black-Scholes line.

Usage: python scripts/run_smile_intuition.py [days_to_expiry=90]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from src.black76 import black76_price
from src.calibration import build_slice
from src.heston import HestonParams, simulate_heston_paths
from src.market_data import clean_chain, fetch_chain
from src.smile import heston_smile_cf, heston_smile_mc

BLUE, RED = "#1f77b4", "#d62728"
PATHS = 400_000


def main(target_days: int = 90):
    params_file = ROOT / "outputs" / "spx_calibrated_params.csv"
    if not params_file.exists():
        sys.exit("Run scripts/run_spx.py first (it calibrates Heston to SPX).")
    P = HestonParams(**pd.read_csv(params_file).iloc[0].to_dict())
    raw, spot, asof = fetch_chain("^SPX")
    chain = clean_chain(raw, spot)
    counts = chain.groupby("days").size()
    avail = counts[counts >= 20].index.values
    days = int(avail[np.abs(avail - target_days).argmin()])
    sl = build_slice(chain, spot, days)
    F, D, T = sl["F"], sl["D"], sl["T"]
    r = -np.log(D) / T
    m_mkt, iv_mkt = np.log(sl["K"] / F), sl["iv"]
    sigma = float(np.interp(0.0, m_mkt, iv_mkt))  # Black-Scholes comparison uses the market ATM vol

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    q = r - np.log(F / spot) / T
    st, _ = simulate_heston_paths(spot, P, r, q, T, max(25, int(100 * T)), PATHS, seed=5, scheme="qe")
    x = np.log(st[:, -1] / F)  # log forward-return under the risk-neutral measure
    mgrid = np.linspace(m_mkt.min(), m_mkt.max(), 40)
    iv_model = heston_smile_cf(F, F * np.exp(mgrid), T, P, D)
    lo, hi = -0.45, 0.25
    grid = np.linspace(lo, hi, 400)
    heston_pdf = gaussian_kde(x[::4], bw_method=0.08)(grid)
    bs_pdf = norm.pdf(grid, -0.5 * sigma**2 * T, sigma * np.sqrt(T))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a, scale in zip(ax[:2], ("linear", "log")):
        a.plot(grid, bs_pdf, color=BLUE, lw=2, label=f"Black-Scholes (lognormal, vol={sigma:.0%})")
        a.plot(grid, heston_pdf, color=RED, lw=2, label="Heston calibrated to SPX")
        a.set(xlim=(lo, hi), xlabel="log-return ln($S_T$/F)", ylabel="Risk-neutral density", yscale=scale)
    ax[0].set_title("(a) Return distribution", fontweight="bold")
    ax[0].set_ylim(0, heston_pdf.max() * 1.1)
    ax[1].set_title("(b) Log scale: fat left tail", fontweight="bold")
    ax[1].set_ylim(1e-3, heston_pdf.max() * 2.5)
    ax[1].axvspan(lo, -0.15, color="0.85", alpha=0.6, zorder=0)
    ax[1].text(lo + 0.01, heston_pdf.max() * 1.8, "crash region:\nOTM puts pay here", fontsize=9, va="top")
    show = np.unique(np.linspace(0, len(m_mkt) - 1, min(45, len(m_mkt))).round().astype(int))
    ax[2].plot(m_mkt[show], iv_mkt[show] * 100, "o", ms=4, color="0.25", alpha=0.8, label="SPX market (mid)")
    ax[2].plot(mgrid, iv_model * 100, color=RED, lw=2, label="Heston")
    ax[2].axhline(sigma * 100, color=BLUE, lw=2, label="Black-Scholes: flat")
    ax[2].set(title="(c) Implied-vol curve", xlabel="log-moneyness ln(K/F)", ylabel="Implied volatility (%)")
    ax[2].title.set_fontweight("bold")
    ax[2].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Why SPX options have a smile ({days}-day expiry, {asof})", fontsize=14, fontweight="bold")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = ROOT / "outputs"
    fig.savefig(out / "smile_intuition.png", dpi=150)

    # Numbers behind the story: a put 10% below the forward priced under each model
    k = F * np.exp(-0.10)
    p_h = np.exp(-r * T) * np.mean(np.maximum(k - st[:, -1], 0))
    p_bs = black76_price(F, k, T, sigma, D, "put")
    print(f"{days}d, ATM vol {sigma:.1%}. Put 10% below forward: Heston {p_h:.2f} vs Black-Scholes {p_bs:.2f} "
          f"({p_h / p_bs - 1:+.0%}); P(return < -15%): Heston {np.mean(x < -0.15):.2%} vs "
          f"BS {norm.cdf(-0.15, -0.5 * sigma**2 * T, sigma * np.sqrt(T)):.2%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 90)
