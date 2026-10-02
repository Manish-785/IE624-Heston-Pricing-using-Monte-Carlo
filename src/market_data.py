"""SPX option-chain download, cleaning, and forward/discount extraction."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_chain(symbol: str = "^SPX", min_days: int = 20, max_days: int = 400,
                refresh: bool = False) -> tuple[pd.DataFrame, float, str]:
    """Return (chain, spot, asof) for any Yahoo ticker. Snapshot is cached to data/ so runs are reproducible."""
    DATA_DIR.mkdir(exist_ok=True)
    tag = symbol.lstrip("^").lower()
    asof = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = DATA_DIR / f"{tag}_chain_{asof}.csv"
    if path.exists() and not refresh:
        snap = _read_snapshot(path, tag)
        if "lastPrice" in snap[0].columns:  # older snapshots lack it; re-download those
            return snap
    try:
        return _download_chain(symbol, path, asof, min_days, max_days)
    except Exception as exc:  # network down, Yahoo schema change, empty chain, ...
        cached = sorted(DATA_DIR.glob(f"{tag}_chain_*.csv"))
        if not cached:
            raise RuntimeError(f"{symbol} download failed ({exc}) and no cached snapshot exists.") from exc
        print(f"WARNING: {symbol} download failed ({exc}); using cached snapshot {cached[-1].name}")
        return _read_snapshot(cached[-1], tag)


def fetch_spx_chain(min_days: int = 20, max_days: int = 400, refresh: bool = False):
    return fetch_chain("^SPX", min_days, max_days, refresh)


def _read_snapshot(path: Path, tag: str) -> tuple[pd.DataFrame, float, str]:
    df = pd.read_csv(path)
    return df, float(df["spot"].iloc[0]), path.stem.removeprefix(f"{tag}_chain_")


def _download_chain(symbol: str, path: Path, asof: str, min_days: int, max_days: int
                    ) -> tuple[pd.DataFrame, float, str]:
    import yfinance as yf  # imported lazily; only needed when downloading
    tk = yf.Ticker(symbol)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    frames = []
    for exp in tk.options:
        days = (pd.Timestamp(exp) - pd.Timestamp(asof)).days
        if not min_days <= days <= max_days:
            continue
        ch = tk.option_chain(exp)
        for kind, d in (("call", ch.calls), ("put", ch.puts)):
            d = d[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
            d["type"], d["expiry"], d["days"] = kind, exp, days
            frames.append(d)
    if not frames:
        raise ValueError("Yahoo returned no option expiries in the requested range.")
    df = pd.concat(frames, ignore_index=True)
    df["spot"] = spot
    df.to_csv(path, index=False)
    return df, spot, asof


def clean_chain(df: pd.DataFrame, spot: float, max_rel_spread: float = 0.5,
                use_last_price: bool = False) -> pd.DataFrame:
    """Keep sane quotes near the money.

    Default: two-sided quotes priced at bid/ask mid. With use_last_price=True (for when the
    market is closed and Yahoo returns zero bid/ask) the last traded price is used instead;
    these are non-synchronous, so the resulting smile is noisier.
    """
    if use_last_price:
        d = df[df.lastPrice > 0].copy()
        d["mid"] = d["bid"] = d["ask"] = d.lastPrice
    else:
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
    k, y = both.index.values, (both.c - both.p).values
    # Theil-Sen is insensitive to stale/crossed quotes; refit by OLS on the inliers.
    slope = theilslopes(y, k)[0]
    intercept = np.median(y - slope * k)
    res = y - (intercept + slope * k)
    mad = max(np.median(np.abs(res - np.median(res))) * 1.4826, 1e-6 * np.max(np.abs(y)))
    keep = np.abs(res) <= 4 * mad
    if keep.sum() >= 3:
        slope, intercept = np.polyfit(k[keep], y[keep], 1)
    disc = -slope
    if not (0.5 < disc < 1.05) or intercept <= 0:
        raise ValueError(f"Implausible parity fit (D={disc:.3f}); check the quotes for this expiry.")
    return float(intercept / disc), float(disc)
