# GTA VI Trophy Rarity Forecaster — merged entry

Entry for make.fun **gta6-trophy-rarity-forecast-01**: forecast every GTA VI
PS5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles
per trophy, from the case's history of house reads. Python 3.13, standard
library only, deterministic (repeat runs byte-identical), ~0.1 s per case.

## Architecture: two engines, one blend

**Engine 1 — hurdle-Poisson / log-time (v3).** Collapses the history to
distinct page states. For a single frozen state it projects the pending
refresh with a hurdle-Poisson earners law (a large share of trophies gain
no earners across a refresh; growth concentrates in already-common
trophies). For rich histories it fits earners growth in log-time with
shrinkage toward the aggregate slope. Its medians measured sharp (the
family's official practice loss was 0.173), but its intervals measured
far too narrow (official practice coverage_90 0.418), so every row is
recalibrated in logit space: deviations from the median are rescaled by
per-path asymmetric factors with a sigma floor.

**Engine 2 — two-driver logit (v6).** Per-trophy logit-rate fits against
dilution (ln players) and progression (ln hours), ridge-shrunk toward the
pooled fit, with a backtested player projection, a count-consistency
clamp (implied earners never below observed earners) and a spread built
from horizon, fit residuals and refresh-lag range. Its medians track
violent growth regimes where engine 1 lags.

**Blend.** Median = equal logit-average of the two engines' medians.
Spread = per-side maximum of engine 1's recalibrated deviation and
0.9 × engine 2's deviation, so each row inherits whichever engine is
more cautious on that side.

## Validation (simulations, not official scores)

On neutral synthetic worlds generated from the real opening state and
refresh anchors (hurdle and smooth structures × final and practice case
shapes, 6 seeds each), and on two independent violent-growth suites, this
blend is the only candidate of the three (engine 1, engine 2, blend) that
clears the house persistence baseline in **every** final-shaped cell:

- neutral hurdle-final 0.239 vs house 0.234 (statistical tie), smooth-final 0.251 vs 0.276
- neutral practice cells 0.331 / 0.337 vs house 1.216 / 0.921
- violent-growth suites: final cells 0.67–1.60 vs house ~1.36–1.93

Coverage of the 90% band on the neutral worlds: 0.93–1.00. The official
score is decided by the server on the final case; no simulation here is
an official result.

## Layout

- `main.py` — the whole entry (engine 2 is embedded and run in its own
  namespace to keep the two engines' helpers apart).
- `LICENSE` — MIT.
