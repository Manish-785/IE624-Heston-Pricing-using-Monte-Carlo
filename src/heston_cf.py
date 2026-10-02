"""Semi-analytic Heston European option prices via the characteristic function.

Used for calibration (fast, noise-free); the Monte Carlo engine is then used to
cross-check the calibrated parameters.
"""
from functools import lru_cache

import numpy as np

from .heston import HestonParams

# Composite Gauss-Legendre rule for the Fourier integral. The integrand decays like
# exp(-u^2 * vbar * T / 2), so the truncation point grows as maturity shrinks; a fixed
# [0, 200] grid is wrong for expiries under a few weeks.
_PANEL_WIDTH = 20.0
_NODES_PER_PANEL = 48
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_NODES_PER_PANEL)
_U_MIN_CUT, _U_MAX_CUT = 300.0, 8000.0


def _quadrature(maturity: float, p: HestonParams) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights on [0, u_max], with u_max chosen so the integrand has decayed."""
    decay = np.exp(-p.kappa * maturity)
    vbar = max(p.theta + (p.v0 - p.theta) * (1 - decay) / (p.kappa * maturity), 1e-4)
    u_max = float(np.clip(np.sqrt(2 * 50 / (0.5 * vbar * maturity)), _U_MIN_CUT, _U_MAX_CUT))
    return _panels(int(np.ceil(u_max / _PANEL_WIDTH)))


@lru_cache(maxsize=64)
def _panels(n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, n_panels * _PANEL_WIDTH, n_panels + 1)
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[:-1] + edges[1:])[:, None]
    return (mid + half * _GL_X).ravel(), (half * _GL_W).ravel()


def _log_cf(u: np.ndarray, maturity: float, p: HestonParams) -> np.ndarray:
    """E[exp(i u ln(F_T/F_0))] (Albrecher et al. 'little Heston trap' form)."""
    xi = max(p.xi, 1e-8)
    iu = 1j * u
    b = p.kappa - p.rho * xi * iu
    d = np.sqrt(b * b + xi**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * maturity)
    c = p.kappa * p.theta / xi**2 * ((b - d) * maturity - 2.0 * np.log((1 - g * e) / (1 - g)))
    dd = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(c + dd * p.v0)


def heston_call_cf(forward: float, strikes: np.ndarray, maturity: float,
                   p: HestonParams, discount: float = 1.0) -> np.ndarray:
    """Discounted Heston call prices: D*[F - sqrt(FK)/pi * int Re(e^{iuk} phi(u-i/2))/(u^2+1/4) du].

    Prices are clipped to the no-arbitrage band [D*(F-K)+, D*F] to remove quadrature noise
    (~1e-10) that would otherwise make implied-vol inversion fail deep in the wings.
    """
    p.validate()
    strikes = np.asarray(strikes, dtype=float)
    if forward <= 0 or maturity <= 0 or discount <= 0 or np.any(strikes <= 0):
        raise ValueError("Require forward, maturity, discount and strikes > 0.")
    u, w = _quadrature(maturity, p)
    k = np.log(forward / strikes)[:, None]
    phi = _log_cf(u - 0.5j, maturity, p)[None, :]
    integrand = np.real(np.exp(1j * u[None, :] * k) * phi) / (u[None, :] ** 2 + 0.25)
    call = discount * (forward - np.sqrt(forward * strikes) / np.pi * (integrand @ w))
    if not np.all(np.isfinite(call)):
        raise FloatingPointError("Heston characteristic-function pricer returned non-finite values.")
    return np.clip(call, discount * np.maximum(forward - strikes, 0.0), discount * forward)


def heston_put_cf(forward: float, strikes: np.ndarray, maturity: float,
                  p: HestonParams, discount: float = 1.0) -> np.ndarray:
    call = heston_call_cf(forward, strikes, maturity, p, discount)
    return call - discount * (forward - np.asarray(strikes))  # put-call parity
