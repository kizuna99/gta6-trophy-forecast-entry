# GTA VI Trophy Rarity Forecaster

Entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast, for every
PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy. Scored by mean pinball
loss on the logit scale; lower is better.

## Approach

The forecast page is a snapshot that only changes when Exophase republishes
its counts, so `main.py` first collapses the case history to its distinct
states. Two forces then move a trophy's rate, and the model fits them
separately per trophy on the logit scale:

- **Dilution** — newly tracked players have earned less, so growth in the
  tracked-player count drags rates down (`beta * ln(players)`).
- **Progression** — the tracked base keeps playing and earns more
  (`gamma * ln(1 + hours)`).

Both coefficients are ridge-shrunk toward the pooled fit across all
trophies; over long horizons an unsaturated trophy's progression term is
floored at zero, since dilution is carried by the player term alone. The
player count at the target comes from whichever of three growth curves
(log-time linear, power law, logistic in log-time) backtests best on the
case's own states. Forecasts are clamped count-consistently (implied earners
never fall below the earners already observed), averaged over the typical
refresh lag — the answer is the last refresh's state, not the smooth path —
and given a logit-normal spread widened by the horizon, the fit residuals,
and a typical refresh step.

A single-state history (the launch-night practice shape) has nothing to
fit, so a stated prior takes over: the tracked count is expected to
multiply several-fold at the first refresh while the incoming cohort earns
at a fraction of the frozen cohort's rates set by the ratio of their play
windows; the platinum's earners stay frozen.

## Running

Standard library only, Python 3.13, no network, deterministic.

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` beside it in
the output directory.
