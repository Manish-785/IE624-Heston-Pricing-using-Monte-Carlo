# Part 3: Why should there be a volatility curve?

**Black-Scholes says there should not be one.** It assumes a single constant volatility, so
inverting any option price (any strike, any maturity) would return the same sigma. In real
markets the implied volatility varies with strike: for equities and indices it slopes
downward (a "skew"), and it often turns up in the far wings (a "smile"). See
`outputs/spx_iv_curve.png` (S&P 500) and `outputs/spx_smile_market_vs_heston.png`.

## Intuition

1. **Implied vol is a price quoted in vol units.** A high implied vol at a strike simply means
   that option is expensive relative to what a lognormal, constant-vol model would charge.
2. **OTM options are bets on the tails.** An option far out of the money pays off only if the
   underlying makes a big move, so its price is almost entirely the probability of that move.
   Black-Scholes assumes lognormal returns, which have thin tails. Real returns have fatter tails.
   Pricing the tail correctly therefore requires a *higher* sigma than the ATM one.
3. **Why the left tail is the fat one (the skew):**
   - *Hedging demand / payoff asymmetry.* Investors who hold stocks buy OTM puts as portfolio
     insurance (a put pays exactly in the crash scenarios they fear), while few investors buy
     OTM calls; many sell them (covered-call writing). Persistent demand for puts bids up their
     prices, i.e. their implied vol. Dealers who sell puts must hedge the jump/gap risk of a
     crash that delta-hedging cannot cover, and charge for it.
   - *Leverage effect.* When prices fall, firm leverage rises and volatility rises with it:
     returns and volatility are negatively correlated. Down-moves are therefore larger and
     more clustered than up-moves, giving a left-skewed return distribution.
4. **Short expiries are more curved** than long ones: over a short horizon a jump or a vol
   spike cannot average out, so the tail matters more. (Visible in the 28-day AAPL curve.)

## What Heston adds

Heston makes volatility random (CIR variance) and correlated with the spot (rho < 0).
- `rho < 0` produces the downward skew: spot falls when variance rises, creating a fat left tail.
- `xi` (vol of vol) produces kurtosis, i.e. curvature in the wings.
- With `rho = 0` the model produces a symmetric smile only (see `outputs/heston_smile.png`).

`python scripts/run_smile_intuition.py` (run `scripts/run_spx.py` first) produces
`outputs/smile_intuition.png` for the 90-day SPX expiry: the risk-neutral return density of
Heston calibrated to SPX against the lognormal Black-Scholes density at the same ATM vol, the
left-tail comparison on a log scale, and the implied-vol curve with the actual SPX quotes on top.
In that run the probability of a return below -15% is 5.6% under Heston against 1.6% under
Black-Scholes, and a put 10% below the forward costs about 4x the flat-vol price, which is the
higher implied vol at low strikes.
