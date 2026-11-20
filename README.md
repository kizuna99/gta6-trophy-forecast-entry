[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kizuna99/gta6-trophy-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kizuna99/gta6-trophy-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — hybrid v4 entry

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at a case's
target time. Scored by mean pinball loss on the logit scale.

## How it works

The case history is first collapsed into **distinct snapshots**
(consecutive reads with identical `players_tracked` and identical
`rates` count once). Two regimes follow:

**Single snapshot (frozen page).** The history shows one unchanged
page, so the forecast is about the next Exophase refresh (assumed
cadence 280 minutes). If no refresh can land in the forecast window,
the last page is carried as-is with a narrow logit spread
(sigma 0.05). Otherwise earners are recovered as integers (rounding,
then reconciled against the page's trophies-earned total by fractional
parts), the tracked pool is projected one refresh ahead with an hourly
growth rate of `0.386 * exp(-age_hours/48)`, and each trophy's new
earners follow a **hurdle-Poisson** law: with probability 0.10 the
count is unchanged, otherwise it grows by `Poisson(0.016 * e**2.5)`
(capped by the projected pool). Quantiles are read off the exact
discrete CDF by interpolation in logit space, widened asymmetrically
(+0.16 sigma below the median, +0.08 above), and then **shrunk around
their own median** by the calibration factor `CAL` in `main.py` —
the median keeps the sharp single-snapshot centre while the spread is
narrowed to hold roughly 0.88–0.92 coverage instead of over-covering.

**Multiple snapshots.** Counts (pool, total earners, per-trophy
earners) are fit by least squares against `ln(1 + hours/8)`, damped by
`span/(span+dx)`, and extrapolated to the last refresh at or before
the target. Per-trophy slopes are shrunk toward the trophy's share of
the total slope. The spread is a symmetric logit sigma derived from
the fit residuals, capped at 0.35.

## Running

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` per case. Python 3 standard library only, no network,
fully deterministic (two runs are byte-identical).

## License

MIT — see `LICENSE`.
