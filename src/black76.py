"""Black-76 option pricing and implied-volatility inversion."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _validate(forward: float, strike: float, maturity: float, discount: float) -> None:
    if not np.all(np.isfinite([forward, strike, maturity, discount])):
        raise ValueError("Black-76 inputs must be finite.")
    if forward <= 0 or strike <= 0 or maturity <= 0 or discount <= 0:
        raise ValueError("Require forward, strike, maturity, and discount > 0.")


def black76_price(forward: float, strike: float, maturity: float, volatility: float,
                  discount: float = 1.0, option_type: str = "call") -> float:
    """Discounted Black-76 price from forward: D*[phi F N(phi d1)-phi K N(phi d2)]."""
    _validate(forward, strike, maturity, discount)
    if volatility < 0 or not np.isfinite(volatility):
        raise ValueError("volatility must be finite and nonnegative.")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if volatility == 0:
        return float(discount * max(forward - strike, 0) if kind == "call" else discount * max(strike - forward, 0))
    st = volatility * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * st * st) / st
    d2 = d1 - st
    phi = 1 if kind == "call" else -1
    return float(discount * phi * (forward * norm.cdf(phi * d1) - strike * norm.cdf(phi * d2)))


def implied_volatility(price: float, forward: float, strike: float, maturity: float,
                       discount: float = 1.0, option_type: str = "call",
                       tol: float = 1e-10) -> float:
    """Invert Black-76 using Brent's method; reject prices outside bounds."""
    _validate(forward, strike, maturity, discount)
    if not np.isfinite(price) or price < 0:
        raise ValueError("price must be finite and nonnegative.")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    intrinsic = discount * (max(forward - strike, 0) if kind == "call" else max(strike - forward, 0))
    upper = discount * (forward if kind == "call" else strike)
    eps = 1e-12 * max(1.0, upper)
    if price < intrinsic - eps or price >= upper:
        raise ValueError(f"Option price {price:g} violates bounds [{intrinsic:g}, {upper:g}).")
    if price <= intrinsic + eps:
        return 0.0
    fn = lambda vol: black76_price(forward, strike, maturity, vol, discount, kind) - price
    hi = 1.0
    while fn(hi) < 0 and hi < 16:
        hi *= 2
    if fn(hi) < 0:
        raise RuntimeError("Could not bracket implied volatility root.")
    try:
        return float(brentq(fn, 1e-12, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200))
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError("Black-76 implied-volatility inversion failed.") from exc
