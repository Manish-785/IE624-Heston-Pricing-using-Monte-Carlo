"""Semi-analytic Heston European option prices via the characteristic function.

Used for calibration (fast, noise-free); the Monte Carlo engine is then used to
cross-check the calibrated parameters.
"""
import numpy as np

from .heston import HestonParams

# Gauss-Legendre nodes on [0, U_MAX] for the Fourier integral.
_U_MAX = 200.0
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(600)
_U = 0.5 * _U_MAX * (_NODES + 1.0)
_W = 0.5 * _U_MAX * _WEIGHTS


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
    """Discounted Heston call prices: D*[F - sqrt(FK)/pi * int Re(e^{iuk} phi(u-i/2))/(u^2+1/4) du]."""
    k = np.log(forward / np.asarray(strikes, dtype=float))[:, None]
    phi = _log_cf(_U - 0.5j, maturity, p)[None, :]
    integrand = np.real(np.exp(1j * _U[None, :] * k) * phi) / (_U[None, :] ** 2 + 0.25)
    integral = integrand @ _W
    return discount * (forward - np.sqrt(forward * np.asarray(strikes)) / np.pi * integral)


def heston_put_cf(forward: float, strikes: np.ndarray, maturity: float,
                  p: HestonParams, discount: float = 1.0) -> np.ndarray:
    call = heston_call_cf(forward, strikes, maturity, p, discount)
    return call - discount * (forward - np.asarray(strikes))  # put-call parity
