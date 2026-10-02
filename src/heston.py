"""Heston stochastic-volatility simulation with full-truncation Euler."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Risk-neutral Heston parameters; variances are in annualized units."""
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def validate(self) -> None:
        if not np.all(np.isfinite([self.kappa, self.theta, self.xi, self.rho, self.v0])):
            raise ValueError("All Heston parameters must be finite.")
        if self.kappa <= 0 or self.theta < 0 or self.xi < 0 or self.v0 < 0:
            raise ValueError("Require kappa > 0 and theta, xi, v0 >= 0.")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must lie in [-1, 1].")


def simulate_heston_paths(s0: float, params: HestonParams, r: float, q: float,
                          maturity: float, steps: int, n_paths: int,
                          seed: int | None = 1234, antithetic: bool = True,
                          scheme: str = "euler") -> tuple[np.ndarray, np.ndarray]:
    """Simulate paths; scheme="euler" (log-Euler S, full-truncation v) or "qe".

    "qe" is Andersen's (2008) quadratic-exponential scheme with martingale-corrected
    log-spot; it is far less biased when the Feller condition fails (as for SPX fits).

    Returns arrays of shape (n_paths, steps + 1). Correlated shocks satisfy
    Z_v = rho Z_S + sqrt(1-rho**2) Z_perp. The state variance is floored at
    zero, and its drift/diffusion use max(v, 0), avoiding invalid square roots.
    """
    params.validate()
    if scheme not in {"euler", "qe"}:
        raise ValueError("scheme must be 'euler' or 'qe'.")
    if not np.isfinite(s0) or s0 <= 0 or not np.isfinite(r) or not np.isfinite(q):
        raise ValueError("s0 must be positive and rates must be finite.")
    if maturity <= 0 or steps < 1 or n_paths < 2:
        raise ValueError("Require maturity > 0, steps >= 1, and n_paths >= 2.")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic variates are enabled.")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    s = np.empty((n_paths, steps + 1), dtype=float)
    v = np.empty_like(s)
    s[:, 0] = s0
    v[:, 0] = params.v0
    batch = n_paths // 2 if antithetic else n_paths
    if scheme == "qe":
        return _simulate_qe(s, v, rng, batch, antithetic, r, q, dt, steps, params)
    rho_perp = np.sqrt(max(0.0, 1.0 - params.rho**2))
    for j in range(steps):
        zs = rng.standard_normal(batch)
        zv_perp = rng.standard_normal(batch)
        if antithetic:
            zs = np.concatenate((zs, -zs))
            zv_perp = np.concatenate((zv_perp, -zv_perp))
        zv = params.rho * zs + rho_perp * zv_perp
        vp = np.maximum(v[:, j], 0.0)
        s[:, j + 1] = s[:, j] * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * zs)
        v[:, j + 1] = v[:, j] + params.kappa * (params.theta - vp) * dt + params.xi * np.sqrt(vp * dt) * zv
        # Full truncation: retain raw Euler state for drift next step, but
        # floor output to keep the simulated variance nonnegative.
        v[:, j + 1] = np.maximum(v[:, j + 1], 0.0)
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(v)):
        raise FloatingPointError("Simulation produced non-finite paths.")
    return s, v


def _simulate_qe(s, v, rng, batch, antithetic, r, q, dt, steps, p):
    """Andersen QE variance step with martingale-corrected log-spot (gamma1=gamma2=0.5)."""
    psi_c, g1, g2 = 1.5, 0.5, 0.5
    kap, th, xi, rho = p.kappa, p.theta, max(p.xi, 1e-8), p.rho
    e = np.exp(-kap * dt)
    k1 = g1 * dt * (kap * rho / xi - 0.5) - rho / xi
    k2 = g2 * dt * (kap * rho / xi - 0.5) + rho / xi
    k3 = g1 * dt * (1 - rho**2)
    k4 = g2 * dt * (1 - rho**2)
    a_coef = k2 + 0.5 * k4
    logs = np.full(s.shape[0], np.log(s[0, 0]))
    for j in range(steps):
        vj = v[:, j]
        zv = rng.standard_normal(batch)
        u = rng.random(batch)
        zs = rng.standard_normal(batch)
        if antithetic:
            zv, u, zs = (np.concatenate((x, y)) for x, y in ((zv, -zv), (u, 1 - u), (zs, -zs)))
        m = th + (vj - th) * e
        s2 = vj * xi**2 * e / kap * (1 - e) + th * xi**2 / (2 * kap) * (1 - e) ** 2
        psi = s2 / np.maximum(m * m, 1e-300)
        quad = psi <= psi_c
        psi_q = np.maximum(psi, 1e-12)
        # b2 is only used where psi <= 1.5 (quadratic branch); clip so the masked-out
        # branch cannot produce negative square roots.
        b2 = np.maximum(2 / psi_q - 1 + np.sqrt(2 / psi_q) * np.sqrt(np.maximum(2 / psi_q - 1, 0)), 0.0)
        a = m / (1 + b2)
        pe = (psi - 1) / (psi + 1)
        beta = (1 - pe) / np.maximum(m, 1e-300)
        v_quad = a * (np.sqrt(b2) + zv) ** 2
        v_exp = np.where(u <= pe, 0.0, np.log((1 - pe) / np.maximum(1 - u, 1e-300)) / beta)
        vn = np.where(quad, v_quad, v_exp)
        # E[exp(A v')] for the martingale correction
        with np.errstate(over="ignore", invalid="ignore"):
            mq = np.exp(a_coef * a * b2 / (1 - 2 * a_coef * a)) / np.sqrt(1 - 2 * a_coef * a)
            me = pe + beta * (1 - pe) / (beta - a_coef)
        mgf = np.where(quad, mq, me)
        if not np.all(np.isfinite(mgf) & (mgf > 0)):
            raise FloatingPointError(
                "QE martingale correction is invalid (parameters too extreme for this step size); "
                "use more time steps.")
        k0 = -np.log(mgf) - (k1 + 0.5 * k3) * vj
        logs = logs + (r - q) * dt + k0 + k1 * vj + k2 * vn + np.sqrt(np.maximum(k3 * vj + k4 * vn, 0)) * zs
        s[:, j + 1] = np.exp(logs)
        v[:, j + 1] = vn
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(v)):
        raise FloatingPointError("Simulation produced non-finite paths.")
    return s, v
