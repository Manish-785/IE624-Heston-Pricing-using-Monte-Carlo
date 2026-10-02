"""Model-implied smiles: Heston (semi-analytic or Monte Carlo) in Black-76 implied-vol terms."""
import numpy as np

from .black76 import implied_volatility
from .heston import HestonParams, simulate_heston_paths
from .heston_cf import heston_call_cf
from .monte_carlo import price_european_mc


def _invert(price: float, forward: float, strike: float, T: float, disc: float, kind: str) -> float:
    try:
        return implied_volatility(price, forward, strike, T, disc, kind)
    except (ValueError, RuntimeError):
        return np.nan


def heston_smile_cf(forward: float, strikes, T: float, params: HestonParams, disc: float = 1.0,
                    kinds=None) -> np.ndarray:
    """Heston implied vols at `strikes`. `kinds` (per-strike 'call'/'put') defaults to the
    OTM convention (put below the forward, call at/above); NaN where inversion is impossible."""
    strikes = np.asarray(strikes, dtype=float)
    if kinds is None:
        kinds = np.where(strikes < forward, "put", "call")
    calls = heston_call_cf(forward, strikes, T, params, disc)
    out = np.empty(len(strikes))
    for i, (k, c, kind) in enumerate(zip(strikes, calls, kinds)):
        price = c if kind == "call" else c - disc * (forward - k)  # put-call parity
        out[i] = _invert(max(price, 0.0), forward, k, T, disc, kind)
    return out


def heston_smile_mc(spot: float, forward: float, strikes, T: float, params: HestonParams,
                    disc: float, n_paths: int = 100_000, steps_per_year: int = 100,
                    seed: int = 11, kinds=None) -> np.ndarray:
    """Heston Monte Carlo (QE scheme) implied vols; the dividend yield is backed out so the
    simulated forward equals `forward`."""
    strikes = np.asarray(strikes, dtype=float)
    if kinds is None:
        kinds = np.where(strikes < forward, "put", "call")
    r = -np.log(disc) / T
    q = r - np.log(forward / spot) / T
    steps = max(25, int(steps_per_year * T))
    st, _ = simulate_heston_paths(spot, params, r, q, T, steps, n_paths, seed=seed, scheme="qe")
    return np.array([
        _invert(price_european_mc(st[:, -1], k, T, r, kind, antithetic=True).price, forward, k, T, disc, kind)
        for k, kind in zip(strikes, kinds)])
