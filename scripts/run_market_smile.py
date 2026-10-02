"""Part 2: the market implied-volatility curve of the S&P 500 index (default; any Yahoo ticker works).

Black-Scholes assumes one constant volatility, so every strike would imply the same sigma.
Inverting real SPX option prices instead gives a curve in strike -- the skew/smile.
SPX options are European and cash-settled, so Black-76 on the put-call-parity forward is exact
(no early-exercise approximation needed, unlike single stocks).

Usage: python scripts/run_market_smile.py [SPX|AAPL|...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import build_slice
from src.market_data import clean_chain, fetch_chain

OUT = Path(__file__).resolve().parents[1] / "outputs"


def pick_expiries(chain, targets=(30, 90, 250)):
    counts = chain.groupby("days").size()
    days = counts[counts >= 15].index.values
    return sorted({int(days[np.abs(days - t).argmin()]) for t in targets})


def main(name: str = "SPX"):
    OUT.mkdir(exist_ok=True)
    symbol = name if name.startswith("^") or name.upper() != "SPX" else "^SPX"
    raw, spot, asof = fetch_chain(symbol)
    chain = clean_chain(raw, spot)
    if len(chain) < 100:  # market closed: Yahoo gives zero bid/ask
        print("WARNING: too few two-sided quotes (market closed?); using last traded prices.")
        chain = clean_chain(raw, spot, use_last_price=True)
    print(f"{name} as of {asof}: spot {spot:.2f}, {len(chain)} clean quotes")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, 3))
    rows = []
    for days, c in zip(pick_expiries(chain), colors):
        sl = build_slice(chain, spot, days)
        m, v = np.log(sl["K"] / sl["F"]), sl["iv"]
        show = np.unique(np.linspace(0, len(m) - 1, min(60, len(m))).round().astype(int))
        atm = float(np.interp(0.0, m, v))
        ax.plot(m[show], v[show] * 100, "o", ms=4, color=c, alpha=0.85, label=f"{days} days")
        ax.hlines(atm * 100, m.min(), m.max(), color=c, ls="--", lw=1, alpha=0.8)
        rows += [dict(days=days, strike=k, log_moneyness=x, implied_vol=y) for k, x, y in zip(sl["K"], m, v)]
        print(f"  {days:3d}d: F={sl['F']:.1f}, ATM IV={atm:.1%}, IV range {v.min():.1%}-{v.max():.1%}")
    ax.plot([], [], "k--", lw=1, label="Black-Scholes: flat vol (= ATM)")
    ax.set(xlabel="log-moneyness ln(K/F)", ylabel="Implied volatility (%)",
           title=f"{name} implied-volatility curve ({asof}, spot {spot:.0f})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    tag = name.lstrip("^").lower()
    fig.savefig(OUT / f"{tag}_iv_curve.png", dpi=150)
    pd.DataFrame(rows).to_csv(OUT / f"{tag}_iv_curve.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SPX")
