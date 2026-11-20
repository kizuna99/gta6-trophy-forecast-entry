#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model: a hurdle-Poisson count core per trophy, blended in logit space with
a two-driver logit component, plus a quantile blend/spread layer.

Count core
----------
For each trophy we recover the earner count implied by the last read
(e = rate * players / 100) and project two quantities to the target:

* the tracked pool P, extrapolated from the pool trajectory in the case
  history (log-growth, damped, capped). When the history is frozen (the
  page never refreshed inside the window) the pool instead jumps to a
  prior pool size, because a stale page is refreshed by the target.
* the earner count E, extrapolated from the global earners trajectory
  shrunk toward each trophy's own trajectory. When the history is frozen
  the earners grow by a prior growth rate, faded by saturation as a
  trophy's rate approaches 100%.

The count distribution is hurdle-Poisson: with probability HURDLE_PI the
trophy gains no new earners in the window (the hurdle branch); otherwise
new earners arrive as a Poisson increment with mean lambda = e*(fE - 1).
The component median uses E_med = e + (1 - pi) * lambda and its spread
uses the hurdle-mixture variance.

Two-driver logit component
--------------------------
The same pool factor and earner factor drive a direct logit forecast:
logit(r_driver) is built from the last rate with the pool driver and the
growth driver applied at full weight (DRIVER_POOL = DRIVER_GROWTH = 1.0),
the growth driver faded by the same saturation term. This component has
no hurdle, so it sits slightly above the count core whenever growth is
positive.

Blend layer
-----------
The final median logit is (1 - BLEND_W) * logit(count core) +
BLEND_W * logit(driver component). The spread combines a base term, a
growth-uncertainty term, a pool-uncertainty term (its scale is set by a
discrete pool fan: factors 0.94 / 1.00 / 1.10 with probabilities
0.25 / 0.50 / 0.25) and a count term from the hurdle-Poisson variance,
scaled by a damped horizon factor. Quantiles are the blended median
logit plus z_tau * sigma, mapped back through the sigmoid.

A trophy whose last rate is 100% is pinned to [99.9, 100, 100, 100, 100].

Deterministic, standard library only. Reads every *.in file in
$MAKEFUN_INPUT_DIR (else /input) in name order and writes one *.out per
case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# --- Model parameters (initial set, tuned on the dev suite) ---
HURDLE_PI = 0.18        # hurdle probability of no new earners
GROWTH_MU = 0.15        # prior earner growth when history is frozen
POOL_PRIOR = 4200.0     # prior pool size after a refresh of a frozen page
BLEND_W = 0.28          # weight of the logit driver component
SPREAD_BASE = 0.07      # base logit spread
SPREAD_GROWTH = 0.145   # growth-uncertainty logit spread
SPREAD_POOL = 0.145     # pool-uncertainty logit spread
SPREAD_COUNT = 0.175    # count (hurdle-Poisson) logit spread scale
SAT_POW = 0.5           # saturation fade exponent
DRIVER_POOL = 1.0       # pool driver weight in the logit component
DRIVER_GROWTH = 1.0     # growth driver weight in the logit component
POOL_FAN = (0.94, 1.0, 1.10)        # discrete pool uncertainty fan
POOL_FAN_P = (0.25, 0.50, 0.25)     # fan probabilities
POOL_DAMP = 0.55        # damping of extrapolated pool log-growth
POOL_CAP = 5.0          # cap on the damped pool factor
GROWTH_DAMP = 0.60      # damping of extrapolated earner log-growth
GROWTH_CAP = 8.0        # cap on the damped earner factor
SHRINK = 0.5            # shrinkage of per-trophy growth to the global one

_NORM = NormalDist()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    """Logit of a percent rate, with the scoring clamp 0.01..99.99."""
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(l):
    p = 1.0 / (1.0 + math.exp(-clamp(l, -30.0, 30.0)))
    return clamp(100.0 * p, 0.0, 100.0)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def damped_factor(per_hour_log_growth, hours, damp, cap):
    """Extrapolate a log-growth rate over `hours`, damped and capped."""
    if per_hour_log_growth == 0.0 or hours <= 0.0:
        return 1.0
    f = math.exp(damp * per_hour_log_growth * hours)
    return clamp(f, 1.0 / cap, cap)


def pool_fan_sd():
    """SD of the log pool factor under the discrete pool fan."""
    logs = [math.log(f) for f in POOL_FAN]
    mean = sum(p * l for p, l in zip(POOL_FAN_P, logs))
    var = sum(p * (l - mean) ** 2 for p, l in zip(POOL_FAN_P, logs))
    return math.sqrt(var)


FAN_SD = pool_fan_sd()


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    levels = case["levels"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = horizon_h / 24.0

    last = history[-1]
    first = history[0]
    p_last = float(last["players_tracked"])
    rates_last = last["rates"]

    t_first = parse_time(first["at"])
    t_last = parse_time(last["at"])
    hist_h = max((t_last - t_first).total_seconds() / 3600.0, 0.0)

    # --- Pool trajectory ---
    p_first = float(first["players_tracked"])
    pool_moved = any(h["players_tracked"] != first["players_tracked"]
                     for h in history)
    if pool_moved and hist_h > 0.0 and p_first > 0.0:
        g_pool = math.log(p_last / p_first) / hist_h
    else:
        g_pool = 0.0
    pool_trend = damped_factor(g_pool, horizon_h, POOL_DAMP, POOL_CAP)
    if pool_moved:
        f_pool = pool_trend
    else:
        # Frozen page: a refresh by the target moves the pool to the prior.
        f_pool = max(1.0, POOL_PRIOR / p_last) if p_last > 0.0 else 1.0
    p_med = p_last * f_pool

    # --- Global earners trajectory (trophies_earned totals) ---
    e_first = float(first.get("trophies_earned") or 0.0)
    e_last_tot = float(last.get("trophies_earned") or 0.0)
    earners_moved = e_last_tot > e_first > 0.0
    if earners_moved and hist_h > 0.0:
        g_earn = math.log(e_last_tot / e_first) / hist_h
    else:
        g_earn = 0.0
    global_trend = damped_factor(g_earn, horizon_h, GROWTH_DAMP, GROWTH_CAP)

    # Prior growth scale: the prior mu is calibrated on a 90-minute
    # practice horizon; longer horizons scale it with sqrt(time).
    prior_scale = math.sqrt(max(horizon_h, 1.5) / 1.5)

    # Horizon factor for the spread: grows with the fourth root of the
    # horizon relative to the 90-minute practice horizon, capped.
    horizon_factor = clamp((max(horizon_d, 0.0625) / 0.0625) ** 0.25, 1.0, 3.0)

    rows = []
    for idx, t in enumerate(trophies):
        r_last = rates_last[idx]
        if r_last is None:
            r_last = 0.0
        # Pin the universal trophy exactly as specified.
        if r_last >= 99.995:
            rows.append({"id": t["id"], "q": [99.9, 100.0, 100.0, 100.0, 100.0]})
            continue

        e_now = r_last / 100.0 * p_last
        e_round = int(math.floor(e_now + 0.5))
        sat = (1.0 - clamp(r_last, 0.0, 100.0) / 100.0) ** SAT_POW

        # Per-trophy earner trend from its own first/last counts.
        r_first = first["rates"][idx]
        if r_first is None:
            r_first = r_last
        e_first_i = r_first / 100.0 * p_first
        if earners_moved and hist_h > 0.0 and e_first_i > 0.0 and e_now > 0.0:
            g_i = math.log(e_now / e_first_i) / hist_h
            trend_i = damped_factor(g_i, horizon_h, GROWTH_DAMP, GROWTH_CAP)
            f_earn_trend = math.exp(
                SHRINK * math.log(global_trend)
                + (1.0 - SHRINK) * math.log(trend_i))
        elif earners_moved:
            f_earn_trend = global_trend
        else:
            f_earn_trend = 1.0

        if earners_moved:
            f_earn = f_earn_trend
        else:
            # Frozen page: prior growth, faded by saturation.
            f_earn = 1.0 + GROWTH_MU * prior_scale * sat

        # --- Hurdle-Poisson count core ---
        e_eff = max(float(e_round), 0.35)
        lam = e_eff * max(f_earn - 1.0, 0.0)
        e_med = e_eff + (1.0 - HURDLE_PI) * lam
        # Hurdle-mixture variance of the target count.
        var_e = (1.0 - HURDLE_PI) * lam + HURDLE_PI * (1.0 - HURDLE_PI) * lam * lam
        r_count = clamp(100.0 * e_med / p_med, 0.0, 100.0) if p_med > 0 else 0.0

        # --- Two-driver logit component (no hurdle) ---
        e_driver = e_eff * (1.0 + DRIVER_GROWTH * max(f_earn - 1.0, 0.0) * sat)
        p_driver = p_last * (f_pool ** DRIVER_POOL)
        r_driver = clamp(100.0 * e_driver / p_driver, 0.0, 100.0) \
            if p_driver > 0 else 0.0

        # --- Blend layer in logit space ---
        l_med = ((1.0 - BLEND_W) * logit_pct(r_count)
                 + BLEND_W * logit_pct(r_driver))

        # Spread: base + growth + pool (fan-scaled) + count terms.
        g_u = clamp(abs(math.log(f_earn)) / 0.35 + 0.25, 0.0, 1.0)
        p_u = clamp(abs(math.log(f_pool)) / 1.0 + 0.25, 0.0, 1.0)
        p_med_frac = clamp(r_count / 100.0, 0.001, 0.999)
        count_sd_logit = (math.sqrt(var_e) / max(e_med, 0.35)) / (1.0 - p_med_frac)
        c_u = clamp(count_sd_logit / 0.35, 0.0, 1.5)
        fan_u = FAN_SD / 0.064  # ~1.0 for the specified fan
        sigma = math.sqrt(
            SPREAD_BASE ** 2
            + (SPREAD_GROWTH * g_u) ** 2
            + (SPREAD_POOL * p_u * fan_u) ** 2
            + (SPREAD_COUNT * c_u) ** 2
        ) * horizon_factor

        qs = []
        for tau in levels:
            z = _NORM.inv_cdf(tau)
            qs.append(round(pct_from_logit(l_med + z * sigma), 4))
        # Enforce non-decreasing quantiles.
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
