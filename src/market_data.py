"""SPX option-chain download, cleaning, and forward/discount extraction."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_spx_chain(min_days: int = 20, max_days: int = 400, refresh: bool = False) -> tuple[pd.DataFrame, float, str]:
    """Return (chain, spot, asof). Snapshot is cached to data/ so runs are reproducible."""
    import yfinance as yf  # imported lazily; only needed when downloading
    DATA_DIR.mkdir(exist_ok=True)
    asof = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = DATA_DIR / f"spx_chain_{asof}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        return df, float(df["spot"].iloc[0]), asof
    tk = yf.Ticker("^SPX")
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    frames = []
    for exp in tk.options:
        days = (pd.Timestamp(exp) - pd.Timestamp(asof)).days
        if not min_days <= days <= max_days:
            continue
        ch = tk.option_chain(exp)
        for kind, d in (("call", ch.calls), ("put", ch.puts)):
            d = d[["strike", "bid", "ask", "volume", "openInterest"]].copy()
            d["type"], d["expiry"], d["days"] = kind, exp, days
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["spot"] = spot
    df.to_csv(path, index=False)
    return df, spot, asof


def clean_chain(df: pd.DataFrame, spot: float, max_rel_spread: float = 0.5) -> pd.DataFrame:
    """Keep two-sided quotes with sane spreads near the money."""
    d = df[(df.bid > 0) & (df.ask > df.bid)].copy()
    d["mid"] = 0.5 * (d.bid + d.ask)
    d = d[(d.ask - d.bid) / d.mid <= max_rel_spread]
    d = d[(d.strike > 0.7 * spot) & (d.strike < 1.3 * spot)]
    d["T"] = d.days / 365.0
    return d.reset_index(drop=True)


def forward_and_discount(chain: pd.DataFrame, spot: float) -> tuple[float, float]:
    """Implied forward and discount factor from put-call parity: C-P = D*F - D*K."""
    c = chain[chain.type == "call"].set_index("strike").mid
    p = chain[chain.type == "put"].set_index("strike").mid
    both = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
    both = both[(both.index > 0.95 * spot) & (both.index < 1.05 * spot)]
    if len(both) < 3:
        raise ValueError("Not enough matched call/put strikes to infer the forward.")
    slope, intercept = np.polyfit(both.index.values, (both.c - both.p).values, 1)
    disc = -slope
    return float(intercept / disc), float(disc)
