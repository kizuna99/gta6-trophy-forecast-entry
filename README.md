[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kizuna99/gta6-trophy-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kizuna99/gta6-trophy-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast,
for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on
Exophase at the case target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95)
scored by mean pinball loss on the logit scale.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic: the same
input always produces byte-identical output. With no environment variables
set it reads `/input` and writes `/output`, as the grader runs it.

## Model

Three layers, all deterministic:

1. **Hurdle-Poisson count core.** Each trophy's earner count is recovered
   from the last read (`rate * players_tracked / 100`). The tracked pool
   and the earner counts are projected to the target from the case
   history: log-growth extrapolation, damped (pool 0.55, growth 0.60) and
   capped (pool x5, growth x8), with per-trophy growth shrunk 50% toward
   the global earners growth. When the history is frozen (the page never
   refreshed in the window), the pool jumps to a prior of 4,200 tracked
   players and earners grow by a prior 15%, faded by saturation
   `(1 - rate/100)^0.5`. New earners are hurdle-Poisson: with probability
   0.18 a trophy gains none; otherwise the increment is Poisson. The
   core's median and spread come from that mixture.

2. **Two-driver logit component.** The same pool factor and growth factor
   drive a direct forecast of the rate (both driver weights 1.0, growth
   faded by saturation), with no hurdle.

3. **Quantile blend layer.** The final median logit is a blend of the two
   components (driver weight 0.28) in logit space. The spread combines
   base (0.07), growth (0.145), pool (0.145, scaled by a discrete pool
   fan of 0.94 / 1.00 / 1.10 at probabilities 0.25 / 0.50 / 0.25) and
   count (0.175) terms in quadrature, scaled by a damped horizon factor.
   (Spread scales were narrowed from the 0.12 / 0.25 / 0.25 / 0.28
   starting set after dev-suite measurement: wider spreads scored
   worse, narrower ones traded mean gains for worse worst cases; the
   values above are the robust middle.)
   Quantiles are `sigmoid(blended_logit + z_tau * sigma)`.

A trophy whose last rate is 100% is pinned to `[99.9, 100, 100, 100, 100]`.

## Files

- `main.py` — the entrypoint.
- `.makefun/owner.json` — payout wallet declaration.
- `LICENSE` — MIT.
