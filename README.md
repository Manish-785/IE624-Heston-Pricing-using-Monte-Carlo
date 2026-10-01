# IE624: Heston Pricing using Monte Carlo

This repository implements the first computational stage of the project: a risk-neutral Heston simulator, Monte Carlo European option pricing, Black-76 implied-volatility inversion, and a model-generated smile. The example parameters are fixed and **illustrative, not calibrated**. SPX calibration is described in the next section.

## Run

From the repository root, install the dependencies and run the demo:

```powershell
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

The demo writes six figures and two CSV files to `outputs/`. The main result is `outputs/heston_smile.png`; plotted points are direct Black-76 inversions of Heston Monte Carlo option prices, without smoothing. Runtime depends on hardware because the demo simulates 100,000 paths over 252 time steps.

To run the validation suite:

```powershell
python -m pytest
```

## Equations and numerical choices

Under the risk-neutral measure,

\[
dS_t=(r-q)S_tdt+\sqrt{v_t}S_tdW_t^S,\qquad
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\qquad dW_t^S dW_t^v=\rho dt.
\]

The simulator uses log-Euler for spot and a projected full-truncation Euler variance step. With \(v^+=\max(v,0)\) and \(Z_v=\rho Z_S+\sqrt{1-\rho^2}Z_\perp\):

\[
S_{n+1}=S_n\exp((r-q-v_n^+/2)\Delta t+\sqrt{v_n^+\Delta t}Z_S),\quad
v_{n+1}=\max\{0,v_n+\kappa(\theta-v_n^+)\Delta t+\xi\sqrt{v_n^+\Delta t}Z_v\}.
\]

The explicit floor guarantees returned variance paths are nonnegative. It is a practical Euler discretization, not an exact transition sampler, so results retain time-step bias (especially for large \(\xi\) or coarse steps). Antithetic paths use paired opposite Gaussian shocks. Confidence intervals and standard errors account for pair averages.

Discounted terminal payoffs are \(e^{-rT}(S_T-K)^+\) and \(e^{-rT}(K-S_T)^+\). The forward is \(F=S_0e^{(r-q)T}\), discount factor is \(D=e^{-rT}\), and Black-76 prices use

\[
C=D[F N(d_1)-K N(d_2)],\quad P=D[K N(-d_2)-F N(-d_1)],\quad
d_{1,2}=\frac{\ln(F/K)\pm\tfrac12\sigma^2T}{\sigma\sqrt{T}}.
\]

Implied volatility is solved from the price using Brent's bracketed root finder. Prices outside discounted intrinsic/upper bounds raise an error. For the smile, OTM puts are used below the forward and calls at/above it; each is inverted using the same forward and maturity.

## Validation and artifacts

The tests check Black-76 put-call parity, strike monotonicity, inversion round-trips, invalid-price rejection, simulation positivity/finite values, invalid parameters, and pricing input checks. The demo reports antithetic MC standard errors at 25k, 50k, and 100k paths. Outputs include sampled spot/variance paths, convergence, option prices by strike, and the resulting implied-volatility smile.

The initial smile uses \(\kappa=2.0,\theta=0.04,\xi=0.60,\rho=-0.70,v_0=0.09\), with \(S_0=100,r=3\%,q=0,T=1\). These stylized values produce volatility skew through leverage correlation and stochastic variance; they do not represent an SPX fit.

## SPX calibration (real data)

```powershell
python scripts/run_spx.py
```

1. **Data**: SPX option chain from Yahoo Finance (`yfinance`), cached to `data/spx_chain_<date>.csv` so a run is reproducible. Quotes need a two-sided market and a relative spread <= 50%; strikes within 70-130% of spot.
2. **Forward / discount**: per expiry, from put-call parity (`C - P = D(F - K)`), so no dividend or rate assumption is needed.
3. **Market smile**: Black-76 implied vols of OTM options (puts below the forward, calls above) at bid/ask mid.
4. **Calibration**: one parameter set (`kappa, theta, xi, rho, v0`) fitted jointly to 4 maturities, using a semi-analytic characteristic-function pricer (`src/heston_cf.py`) and a vega-weighted price loss, which approximates an implied-vol fit. Monte Carlo is too noisy to calibrate with directly.
5. **Monte Carlo check**: the calibrated model is re-priced with the simulator and inverted back to implied vol; these points should sit on the fitted curve.

Calibrated SPX parameters violate the Feller condition (`2*kappa*theta < xi^2`). In that regime the default Euler scheme is badly biased: ATM 3M call prices were still ~10% too high with 4,000 steps. The simulator therefore has `scheme="qe"` (Andersen's quadratic-exponential scheme), which matches the analytic price within MC error using ~25 steps. `scripts/run_spx.py` uses it.

Outputs: `outputs/spx_smile_market_vs_heston.png` (the comparison), `spx_fit_summary.csv` (RMSE by maturity, in vol points), `spx_calibrated_params.csv`, and per-maturity `spx_smile_*d.csv`.

Caveats: the fit is to a single snapshot of mid quotes; Yahoo data can be delayed or stale (especially far wings and short expiries); and a one-factor Heston cannot fit very short-dated smiles as well as longer ones.

## Structure

```text
src/heston.py       Heston parameters and path simulation
src/monte_carlo.py  European option estimator and standard error
src/heston_cf.py    Semi-analytic Heston prices (calibration)
src/calibration.py  Least-squares Heston calibration to implied vols
src/market_data.py  SPX chain download, cleaning, forward extraction
src/black76.py     Black-76 prices and implied volatility
scripts/run_spx.py  SPX calibration and market-vs-model smile
scripts/run_demo.py Reproducible first demonstration and figures
tests/              Core numerical checks
outputs/            Generated plots and data (created by demo)
```
