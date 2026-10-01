"""Run the first Heston pricing and implied-volatility demonstration."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.black76 import implied_volatility
from src.heston import HestonParams, simulate_heston_paths
from src.monte_carlo import price_european_mc


def main() -> None:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    # Fixed, illustrative risk-neutral parameters; this is not a market calibration.
    p = HestonParams(kappa=2.0, theta=0.04, xi=0.60, rho=-0.70, v0=0.09)
    s0, r, q, maturity, steps = 100.0, 0.03, 0.0, 1.0, 252
    forward = s0 * np.exp((r - q) * maturity)
    discount = np.exp(-r * maturity)

    paths, variance = simulate_heston_paths(s0, p, r, q, maturity, steps, 100_000, seed=2026)
    times = np.linspace(0, maturity, steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(12):
        ax.plot(times, variance[i], lw=1, alpha=0.8)
    ax.set(title="Illustrative Heston variance paths", xlabel="Time (years)", ylabel="Variance")
    fig.tight_layout(); fig.savefig(out / "variance_paths.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(12):
        ax.plot(times, paths[i], lw=1, alpha=0.8)
    ax.set(title="Illustrative Heston asset-price paths", xlabel="Time (years)", ylabel="Spot price")
    fig.tight_layout(); fig.savefig(out / "asset_paths.png", dpi=180); plt.close(fig)

    strike = forward
    conv_rows = []
    for n in (25_000, 50_000, 100_000):
        ss, _ = simulate_heston_paths(s0, p, r, q, maturity, steps, n, seed=7000 + n)
        result = price_european_mc(ss[:, -1], strike, maturity, r, "call", antithetic=True)
        conv_rows.append({"paths": n, "price": result.price, "standard_error": result.standard_error})
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(out / "convergence.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(conv.paths, conv.price, yerr=1.96 * conv.standard_error, fmt="o-", capsize=4)
    ax.set(title="Monte Carlo convergence (95% normal intervals)", xlabel="Number of paths", ylabel="ATM call price")
    fig.tight_layout(); fig.savefig(out / "mc_convergence.png", dpi=180); plt.close(fig)

    strikes = np.linspace(0.75 * forward, 1.25 * forward, 21)
    rows = []
    for k in strikes:
        kind = "put" if k < forward else "call"
        res = price_european_mc(paths[:, -1], float(k), maturity, r, kind, antithetic=True)
        iv = implied_volatility(res.price, forward, float(k), maturity, discount, kind)
        rows.append({"strike": k, "log_moneyness": np.log(k / forward), "option_type": kind,
                     "price": res.price, "standard_error": res.standard_error, "implied_volatility": iv})
    smile = pd.DataFrame(rows)
    if smile.implied_volatility.isna().any() or not np.isfinite(smile.implied_volatility).all():
        raise FloatingPointError("Smile contains invalid implied volatilities.")
    smile.to_csv(out / "heston_smile.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smile.strike, smile.price, "o-", ms=4)
    ax.set(title="Heston Monte Carlo option prices", xlabel="Strike", ylabel="Option price")
    fig.tight_layout(); fig.savefig(out / "option_prices.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smile.log_moneyness, 100 * smile.implied_volatility, "o-", ms=4,
            label="Heston MC (illustrative parameters)")
    ax.set(title="Heston-generated implied-volatility smile (1Y)",
           xlabel="Log-forward-moneyness ln(K/F)", ylabel="Implied volatility (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "heston_smile.png", dpi=180); plt.close(fig)
    print("Illustrative parameters:", p)
    print("Forward:", forward, "Discount:", discount)
    print("Variance sample mean at T:", float(variance[:, -1].mean()))
    print("Convergence (ATM call):\n", conv.to_string(index=False))
    print("Smile IV range (%):", 100 * smile.implied_volatility.min(), "to", 100 * smile.implied_volatility.max())
    print("Wrote figures and CSV data to", out)


if __name__ == "__main__":
    main()
