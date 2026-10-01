"""Monte Carlo European option pricing and standard errors."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    standard_error: float
    n_paths: int


def price_european_mc(terminal_spot: np.ndarray, strike: float, maturity: float,
                      r: float, option_type: str = "call",
                      antithetic: bool = False) -> MCResult:
    """Price a European option; antithetic standard error uses pair averages."""
    st = np.asarray(terminal_spot, dtype=float)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if st.ndim != 1 or st.size < 2 or not np.all(np.isfinite(st)) or np.any(st < 0):
        raise ValueError("terminal_spot must be a finite nonnegative 1D array with >=2 values.")
    if strike <= 0 or maturity <= 0 or not np.isfinite(r):
        raise ValueError("Require strike > 0, maturity > 0, and finite r.")
    payoff = np.maximum(st - strike, 0) if kind == "call" else np.maximum(strike - st, 0)
    discounted = np.exp(-r * maturity) * payoff
    if antithetic:
        if st.size % 2:
            raise ValueError("Antithetic standard error requires an even number of paths.")
        samples = (discounted[:st.size // 2] + discounted[st.size // 2:]) / 2
        # Simulation orders positive then negative shocks; corresponding paths pair by index.
    else:
        samples = discounted
    price = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(samples.size))
    if not np.isfinite(price) or not np.isfinite(se):
        raise FloatingPointError("Non-finite Monte Carlo estimate.")
    return MCResult(price, se, int(st.size))
