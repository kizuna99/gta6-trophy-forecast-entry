#!/usr/bin/env python3
"""Forecast Exophase earn rates for GTA VI (PS5) trophies at a case target.

Model: an earn rate is a ratio, so forecast its two parts separately.
  rate_i = earners_i / players_tracked
Earners are integers hiding inside the printed rates: a rate printed to 2
decimals at P tracked players recovers its earner count by rounding, and
the case's trophies_earned total confirms the recovery.

Two regimes, one model (growth of counts against time, then a ratio):

  * Several distinct page snapshots: fit pool and earners counts as
    straight lines against log-time (a launch curve is close to linear
    in ln(1 + t/tau); fitting log-counts instead bends the extrapolation
    the wrong way and was v2's measured bias). Each trophy's count slope
    is shrunk toward its share of the total-earns slope, the projected
    change is damped by how far beyond the observed span it reaches,
    and the spread comes from the fits' own residuals.

  * One frozen snapshot (the launch-night practice): the page is stale
    and a refresh lands before the target. The pool grows at the
    early-launch rate evidenced by this game's first refresh (638 ->
    3,868 tracked players over one refresh span). Earners grow by a
    hurdle-Poisson law fitted to that same refresh: with probability
    pi a trophy's count does not move at all across the refresh, and
    otherwise it gains Poisson(mu) earners with mu growing superlinearly
    in its current count (mu = A * e^gamma) -- across the observed
    refresh, growth concentrated on the trophies players had already
    earned most, and a large share did not move. The forecast's five
    quantiles are read off that predictive distribution trophy by
    trophy, widened slightly for pool and law uncertainty, so the
    spread is the model's own rather than a symmetric band.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Launch-refresh prior (single frozen snapshot): minutes between Exophase
# refreshes of a hot launch page, and the hourly growth of the tracked
# pool in the first hours after release, decaying with the game's age.
# 0.386/h over the 280-minute span reproduces the one observed refresh
# of this page (638 -> ~3,868 players).
PRIOR_CADENCE_MIN = 280.0
PRIOR_GROWTH_PER_HOUR = 0.386
PRIOR_GROWTH_DECAY_HOURS = 48.0

# Earners growth law across one refresh of a frozen page, fitted to the
# game's observed first refresh (see the module docstring): a trophy's
# count stays put with probability HURDLE_PI, else gains Poisson(mu)
# with mu = GROWTH_A * earners**GROWTH_GAMMA over LAW_SPAN_HOURS,
# scaling linearly with the span to the answering refresh.
GROWTH_A = 0.016
GROWTH_GAMMA = 2.5
HURDLE_PI = 0.10
LAW_SPAN_HOURS = PRIOR_CADENCE_MIN / 60.0

# Extra logit spread on the stale path's distribution quantiles, for
# pool and law uncertainty the fitted distribution does not carry. The
# lower side is wider: after a refresh the dominant risk is dilution
# (the pool jumping further than the prior, pushing every rate down),
# while earners growth is bounded by how far players can progress.
SIGMA_EXTRA_STALE_LO = 0.16
SIGMA_EXTRA_STALE_HI = 0.08

SIGMA_FRESH = 0.05

# General path: log-time basis (hours) for the count fits, the
# pseudo-count at which a trophy's own slope is trusted as much as its
# share of the total-earns slope, and the spread model: a floor plus
# the pooled fit residual, growing slowly with the projection span.
LOGTIME_TAU_HOURS = 8.0
SHRINK_COUNTS = 25.0
SIGMA_GENERAL_FLOOR = 0.04
SIGMA_GENERAL_PER_SQRT_DAY = 0.03


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit_of_rate(x):
    p = clamp_rate(x) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def recover_earners(rates, players, total_earned):
    """Undo the page's rounding: earners_i ~= rate_i/100 * players."""
    raw = [None if r is None else r / 100.0 * players for r in rates]
    est = [None if v is None else int(math.floor(v + 0.5)) for v in raw]
    if total_earned is not None:
        known = [i for i, v in enumerate(est) if v is not None]
        diff = total_earned - sum(est[i] for i in known)
        if known and 0 < abs(diff) <= len(known):
            frac = sorted(known, key=lambda i: raw[i] - math.floor(raw[i]),
                          reverse=diff > 0)
            step = 1 if diff > 0 else -1
            for i in frac[:abs(diff)]:
                if est[i] + step >= 0:
                    est[i] += step
    return est


class Snapshot:
    """One distinct state of the page, seen over a run of reads."""

    def __init__(self, read):
        self.first = parse_time(read["at"])
        self.last = self.first
        self.players = read["players_tracked"]
        self.earned_total = read["trophies_earned"]
        self.rates = list(read["rates"])
        self.earners = recover_earners(self.rates, self.players,
                                       self.earned_total)

    def same_page(self, read):
        return (read["players_tracked"] == self.players
                and list(read["rates"]) == self.rates)


def snapshots_of(history):
    snaps = []
    for read in history:
        if snaps and snaps[-1].same_page(read):
            snaps[-1].last = parse_time(read["at"])
        else:
            snaps.append(Snapshot(read))
    return snaps


def quantile_row(median_rate, sigma):
    centre = logit_of_rate(median_rate)
    row = []
    for z in Z:
        q = rate_of_logit(centre + z * sigma)
        row.append(round(min(100.0, max(0.0, q)), 4))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def _poisson_pmf(mu):
    """Poisson pmf over its effective support, as a {k: p} dict.

    Mode-centred so large mu does not underflow.
    """
    if mu <= 0:
        return {0: 1.0}
    m = int(mu)
    logp = -mu + m * math.log(mu) - math.lgamma(m + 1)
    pmf = {m: math.exp(logp)}
    p = pmf[m]
    k = m
    while k > 0:
        p = p * k / mu
        k -= 1
        pmf[k] = p
    p = pmf[m]
    k = m
    while p > 1e-13:
        p = p * mu / (k + 1)
        k += 1
        pmf[k] = p
    s = sum(pmf.values())
    return {k: v / s for k, v in pmf.items()}


def _growth_mu(earners, span_hours, players_hat, players0):
    """Mean earners gain of one trophy across a frozen-page refresh.

    The fitted law, with two extrapolation guards for counts far above
    the fitted range: the gain cannot push the count past the projected
    pool, and a trophy's count cannot outgrow the pool's own growth by
    more than half again.
    """
    if earners <= 0:
        return 0.0
    mu = GROWTH_A * (earners ** GROWTH_GAMMA) * (span_hours / LAW_SPAN_HOURS)
    pool_factor = players_hat / max(1, players0)
    mu = min(mu, earners * (1.5 * pool_factor - 1.0))
    mu = min(mu, max(0.0, players_hat - earners))
    return max(0.0, mu)


def _distribution_row(earners, span_hours, players_hat, players0):
    """Five quantiles of one trophy's rate at the answering refresh,
    read off its hurdle-Poisson predictive distribution."""
    if earners >= players0:
        return [100.0] * 5
    mu = _growth_mu(earners, span_hours, players_hat, players0)
    dist = {}
    for d, p in _poisson_pmf(mu).items():
        E = min(earners + d, players_hat)
        dist[E] = dist.get(E, 0.0) + (1.0 - HURDLE_PI) * p
    y0 = 100.0 * earners / players_hat
    dist[earners] = dist.get(earners, 0.0) + HURDLE_PI
    # CDF over rates, interpolated linearly in logit space.
    pts = sorted((100.0 * E / players_hat, p) for E, p in dist.items())
    cdf = []
    run = 0.0
    for y, p in pts:
        run += p
        cdf.append(run)
    row = []
    for tau, z in zip(LEVELS, Z):
        if tau <= cdf[0]:
            q = pts[0][0]
        elif tau >= cdf[-1]:
            q = pts[-1][0]
        else:
            q = pts[-1][0]
            for i in range(1, len(pts)):
                if cdf[i] >= tau:
                    y_lo, y_hi = pts[i - 1][0], pts[i][0]
                    c_lo, c_hi = cdf[i - 1], cdf[i]
                    if y_hi <= y_lo or c_hi <= c_lo:
                        q = y_hi
                    else:
                        f = (tau - c_lo) / (c_hi - c_lo)
                        q = rate_of_logit(logit_of_rate(y_lo)
                                          + f * (logit_of_rate(y_hi)
                                                 - logit_of_rate(y_lo)))
                    break
        sig = SIGMA_EXTRA_STALE_LO if z < 0 else SIGMA_EXTRA_STALE_HI
        q = rate_of_logit(logit_of_rate(q) + z * sig)
        row.append(round(min(100.0, max(0.0, q)), 4))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def forecast_stale(snaps, case, at, target, release):
    """One frozen snapshot: project the refresh the target's page shows.

    Returns one quantile row per trophy.
    """
    snap = snaps[0]
    anchor = snap.first
    cadence = timedelta(minutes=PRIOR_CADENCE_MIN)
    refresh_at = anchor + cadence
    if refresh_at <= at or refresh_at > target + timedelta(minutes=20.0):
        # A refresh was already due before the checkpoint and the page
        # did not change, or none lands before the target: the answer
        # is this very page, carried as it stands.
        return [quantile_row(r if r is not None else 0.01, SIGMA_FRESH)
                for r in snap.rates]
    age_hours = max(0.0, (anchor - release).total_seconds() / 3600.0)
    growth = PRIOR_GROWTH_PER_HOUR * math.exp(-age_hours
                                              / PRIOR_GROWTH_DECAY_HOURS)
    hours = (refresh_at - anchor).total_seconds() / 3600.0
    players_hat = snap.players * min(50.0, math.exp(growth * hours))
    rows = []
    for earners, rate in zip(snap.earners, snap.rates):
        if rate is None or earners is None:
            rows.append(quantile_row(0.01, SIGMA_FRESH))
        else:
            rows.append(_distribution_row(earners, hours, players_hat,
                                          snap.players))
    return rows


def _logtime(hours):
    return math.log(1.0 + hours / LOGTIME_TAU_HOURS)


def _linfit(xs, ys):
    """Least squares y ~ a + b x; returns (a, b, residual sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def forecast_general(snaps, case, at, target, release):
    """Several snapshots: fit count growth, project to the refresh.

    Pool and total earns are fitted as counts linear in log-time. Each
    trophy's count slope is shrunk toward its share of the total slope,
    and the projected change is damped by the share of the total span
    the data covers: a projection as far beyond the data as the data
    itself spans is trusted half, a far one far less. When the observed
    counts barely move, every slope is near zero and the forecast
    collapses onto the last page on its own.
    """
    snaps = snaps[-12:]
    last = snaps[-1]
    hours = [(s.first - snaps[0].first).total_seconds() / 3600.0
             for s in snaps]
    xs = [_logtime(h) for h in hours]
    gaps = [(snaps[i + 1].first - snaps[i].first).total_seconds() / 60.0
            for i in range(len(snaps) - 1)]
    gaps.sort()
    cadence = timedelta(minutes=gaps[len(gaps) // 2]) if gaps \
        else timedelta(minutes=PRIOR_CADENCE_MIN)
    refresh_at = last.first
    while refresh_at + cadence <= target:
        refresh_at += cadence
    dx = _logtime((refresh_at - snaps[0].first).total_seconds() / 3600.0) \
        - xs[-1]
    span = xs[-1] - xs[0]
    damp = span / (span + dx) if span + dx > 0 else 1.0
    # Pool and total-earns count fits.
    _, pool_slope, pool_resid = _linfit(xs, [float(s.players)
                                             for s in snaps])
    pool_slope = max(0.0, pool_slope)
    players_hat = max(float(last.players),
                      last.players + pool_slope * dx * damp)
    totals = [float(s.earned_total or 0) for s in snaps]
    _, total_slope, total_resid = _linfit(xs, totals)
    total_slope = max(0.0, total_slope)
    mean_total = sum(totals) / len(totals)
    # Spread: the fits' relative residuals, pooled, on the logit scale,
    # growing slowly with the projection span.
    rel = 0.0
    if mean_total > 0:
        rel = total_resid / mean_total
    if last.players > 0:
        rel = max(rel, pool_resid / last.players)
    days = max(0.0, (refresh_at - last.first).total_seconds() / 86400.0)
    sigma = math.sqrt(SIGMA_GENERAL_FLOOR ** 2 + rel ** 2
                      + (SIGMA_GENERAL_PER_SQRT_DAY * math.sqrt(days)) ** 2)
    sigma = min(0.35, sigma)
    rows = []
    for j, (earners, rate) in enumerate(zip(last.earners, last.rates)):
        if rate is None or earners is None:
            rows.append(quantile_row(0.01, sigma))
        elif earners >= last.players:
            rows.append(quantile_row(100.0, sigma))
        else:
            pts = [(x, float(s.earners[j])) for x, s in zip(xs, snaps)
                   if s.earners[j] is not None]
            if len(pts) >= 2:
                _, slope_i, _ = _linfit([p[0] for p in pts],
                                        [p[1] for p in pts])
                mean_e = sum(p[1] for p in pts) / len(pts)
            else:
                slope_i, mean_e = None, float(earners)
            share = (mean_e / mean_total) if mean_total > 0 else 0.0
            target_slope = total_slope * share
            if slope_i is None:
                slope = target_slope
            else:
                w = mean_e / (mean_e + SHRINK_COUNTS)
                slope = w * slope_i + (1.0 - w) * target_slope
            slope = max(0.0, slope)
            e_hat = earners + slope * dx * damp
            e_hat = min(e_hat, players_hat)
            median = round(100.0 * e_hat / players_hat, 2)
            rows.append(quantile_row(median, sigma))
    return rows


def _base_forecast_case(case):
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    snaps = snapshots_of(case["history"])
    if len(snaps) <= 1:
        rows_q = forecast_stale(snaps, case, at, target, release)
    else:
        rows_q = forecast_general(snaps, case, at, target, release)
    return {"trophies": [{"id": t["id"], "q": q}
                          for t, q in zip(case["trophies"], rows_q)]}


# ---- merged calibration layer -------------------------------------------
# The v3 median engine above (hurdle-Poisson stale path, log-time general
# path) measured sharp on medians but far too narrow on spread against
# every independent check available: the family's official practice
# coverage_90 was 0.418, and 0.51-0.64 on neutral hurdle/smooth worlds,
# where a calibrated 90% band should hold ~0.90. This layer keeps v3's
# medians and each row's asymmetry, and rescales the row's logit-space
# deviations from its median: multiplied by a per-path factor and floored
# at a per-path sigma. Constants tuned on neutral worlds generated from
# the real opening state and refresh anchors (both structures, both case
# shapes), targeting ~0.90 coverage in all four cells while staying ahead
# of the house persistence baseline in each.
MERGE_W_STALE_LO = 2.2
MERGE_W_STALE_HI = 4.4
MERGE_S_STALE = 0.45
MERGE_W_GEN_LO = 1.9
MERGE_W_GEN_HI = 1.9
MERGE_S_GEN = 0.30
_Z95 = 1.6448536269514722


def _row_sigma(row):
    lo = logit_of_rate(row[0])
    hi = logit_of_rate(row[4])
    return max(0.02, (hi - lo) / (2.0 * _Z95))


def _widen_row(row, w_lo, w_hi, s_floor):
    centre = logit_of_rate(row[2])
    sigma = _row_sigma(row)
    floor_scale = s_floor / sigma
    out = []
    for q in row:
        d = logit_of_rate(q) - centre
        scale = max(w_hi if d > 0 else w_lo, floor_scale)
        out.append(round(min(100.0, max(0.0, rate_of_logit(centre + d * scale))), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _v3w_forecast_case(case):
    base = _base_forecast_case(case)
    snaps = snapshots_of(case["history"])
    if len(snaps) <= 1:
        w_lo, w_hi, s = MERGE_W_STALE_LO, MERGE_W_STALE_HI, MERGE_S_STALE
    else:
        w_lo, w_hi, s = MERGE_W_GEN_LO, MERGE_W_GEN_HI, MERGE_S_GEN
    return {"trophies": [{"id": t["id"], "q": _widen_row(t["q"], w_lo, w_hi, s)}
                         for t in base["trophies"]]}


# ---- second engine: two-driver logit forecaster (v6) --------------------
# Embedded verbatim and executed in its own namespace: its medians track
# earners/players dynamics through violent growth regimes where the v3
# engine's medians lag, and its rows carry the refresh-lag spread model.
_V6_SOURCE = r'''"""gta6-trophy-rarity-forecast-01 entry: a refresh-aware, count-consistent forecaster.

How the page behaves drives the model. Exophase republishes its counts on its
own schedule, so between refreshes every read shows the same frozen state; the
history is therefore collapsed to its distinct states before anything is fit.
Two forces move a trophy's earn rate between states: dilution, as newly
tracked players (who have earned less) swell the denominator, and progression,
as the tracked base plays on and earns more. Each trophy's logit rate is fit
against both drivers at once,

    logit(rate) ~ beta * ln(players) + gamma * ln(1 + hours),

with per-trophy coefficients shrunk toward the pooled fit across all trophies,
so a quiet trophy borrows the field's behaviour and a loud one keeps its own.
The player count at the target is projected by whichever of three growth
curves (log-time linear, power law, logistic in log-time) backtests best on
the case's own states. The forecast is then clamped count-consistently: a
trophy's implied earners never fall below the earners already observed.
Quantiles are a logit-normal spread around the projection, widened by the
horizon, by each trophy's fit residuals, and by the size of a typical refresh
step, because the answer is one refresh's state, not the smooth path.

A history with a single distinct state (the launch-night practice shape) has
nothing to fit: the fallback is an explicit prior. The case is then hours
from release, the frozen page still shows the pre-release cohort, and the
first refresh is expected to multiply the tracked count while the new cohort
replicates the old one's earning pattern over a longer play window; earners
are projected with that replication, the platinum's earners stay frozen (it
is gated by every other trophy), and the spread is honestly wide.

Runs on Python 3.13, standard library only, no network. Reads every <id>.in
in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output), byte-identical across runs.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Ridge pull toward the pooled coefficients, and the base logit spread.
RIDGE = 2.5
SIGMA_DAY = 0.48


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def round4(x: float):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b


def distinct_states(case):
    """(hours_since_first_read, players, total_earned, [rates]) per page change."""
    history = case["history"]
    t0 = parse_time(history[0]["at"])
    states = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        hours = (parse_time(read["at"]) - t0) / 3600.0
        states.append((hours, float(read["players_tracked"]),
                       float(read.get("trophies_earned") or 0), list(read["rates"])))
    return states, t0


# ---------------------------------------------------------------- players ---

def _fit_logistic(ts, ps, k_mult):
    """Fit P = p0 + (K - p0) * sigmoid(a + b*ln(1+t)) for K = k_mult * p_last."""
    p0, p_last = ps[0], ps[-1]
    K = k_mult * p_last
    if K <= p0 * 1.05:
        return None
    xs, ys, ws = [], [], []
    for t, p in zip(ts, ps):
        frac = (p - p0) / (K - p0)
        frac = min(0.995, max(0.005, frac))
        xs.append(math.log1p(t))
        ys.append(math.log(frac / (1.0 - frac)))
        ws.append(1.0)
    a, b = wls(xs, ys, ws)
    if b <= 0:
        return None
    return (p0, K, a, b)


def _logistic_at(model, t):
    p0, K, a, b = model
    s = 1.0 / (1.0 + math.exp(-(a + b * math.log1p(t))))
    return p0 + (K - p0) * s


def project_players(states, t_target):
    ts = [s[0] for s in states]
    ps = [s[1] for s in states]
    p_last = ps[-1]
    if len(states) < 4:
        # Too few states to select a curve: log-time linear, floored.
        xs = [math.log1p(t) for t in ts]
        a, b = wls(xs, ps, [1.0] * len(xs))
        proj = a + b * math.log1p(t_target)
        return max(p_last, min(proj, p_last * 12.0))
    k = max(1, len(states) // 4)
    tr_t, tr_p = ts[:-k], ps[:-k]
    te_t, te_p = ts[-k:], ps[-k:]

    def backtest(fit_fn, at_fn):
        model = fit_fn(tr_t, tr_p)
        if model is None:
            return None, float("inf")
        err = 0.0
        for t, p in zip(te_t, te_p):
            pred = at_fn(model, t)
            if pred <= 0:
                return model, float("inf")
            err += (math.log(pred) - math.log(p)) ** 2
        return model, err / len(te_t)

    def fit_lin(tt, pp):
        xs = [math.log1p(t) for t in tt]
        return wls(xs, pp, [1.0] * len(xs))

    def at_lin(model, t):
        return model[0] + model[1] * math.log1p(t)

    def fit_pow(tt, pp):
        xs = [math.log1p(t) for t in tt]
        return wls(xs, [math.log(max(p, 1.0)) for p in pp], [1.0] * len(xs))

    def at_pow(model, t):
        return math.exp(model[0] + model[1] * math.log1p(t))

    def fit_log(tt, pp):
        best = None
        for mult in (1.5, 2.0, 3.0, 5.0, 8.0, 15.0):
            m = _fit_logistic(tt, pp, mult)
            if m is not None:
                best = m
        return best

    cands = []
    for fit_fn, at_fn in ((fit_lin, at_lin), (fit_pow, at_pow), (fit_log, _logistic_at)):
        _, err = backtest(fit_fn, at_fn)
        cands.append((err, fit_fn, at_fn))
    cands.sort(key=lambda c: c[0])
    model = cands[0][1](ts, ps)
    proj = cands[0][2](model, t_target) if model is not None else p_last
    return max(p_last, min(proj, p_last * 15.0))


# ----------------------------------------------------------------- fitting ---

def ridge_two_driver(dus, dvs, dys, ws, prior_b, prior_g, ridge):
    """Solve min sum w (dy - b du - g dv)^2 + ridge ((b-pb)^2 + (g-pg)^2)."""
    s11 = sum(w * u * u for u, w in zip(dus, ws)) + ridge
    s22 = sum(w * v * v for v, w in zip(dvs, ws)) + ridge
    s12 = sum(w * u * v for u, v, w in zip(dus, dvs, ws))
    r1 = sum(w * u * y for u, y, w in zip(dus, dys, ws)) + ridge * prior_b
    r2 = sum(w * v * y for v, y, w in zip(dvs, dys, ws)) + ridge * prior_g
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return prior_b, prior_g
    b = (r1 * s22 - r2 * s12) / det
    g = (s11 * r2 - s12 * r1) / det
    return b, g


def fit_drivers(states, series):
    """Pooled + per-series (beta, gamma) for logit-rate series over states.

    series: list of per-trophy lists of logit rates (or None), aligned with
    states. Returns (pooled_beta, pooled_gamma, [(beta_i, gamma_i, resid_i)]).
    """
    ts = [s[0] for s in states]
    ps = [s[1] for s in states]
    t_last, p_last = ts[-1], ps[-1]
    dus = [math.log(p / p_last) for p in ps]
    dvs = [math.log1p(t) - math.log1p(t_last) for t in ts]
    ws = [0.5 ** ((t_last - t) / 48.0) for t in ts]

    def pooled():
        # Mean logit series across trophies with full data.
        means = []
        for j in range(len(states)):
            vals = [s[j] for s in series if s[j] is not None]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        dys = [m - means[-1] for m in means]
        return ridge_two_driver(dus, dvs, dys, ws, 0.0, 0.0, RIDGE)

    pb, pg = pooled()
    out = []
    for s in series:
        pts = [(dus[j], dvs[j], s[j] - s[-1], ws[j]) for j in range(len(states)) if s[j] is not None]
        if len(pts) < 3 or s[-1] is None:
            out.append((pb, pg, 0.35))
            continue
        b, g = ridge_two_driver([p[0] for p in pts], [p[1] for p in pts],
                                [p[2] for p in pts], [p[3] for p in pts], pb, pg, RIDGE)
        resid = 0.0
        for u, v, y, w in pts:
            resid += w * (y - (b * u + g * v)) ** 2
        resid = math.sqrt(resid / max(sum(p[3] for p in pts), 1e-9))
        out.append((b, g, min(1.0, max(0.08, resid))))
    return pb, pg, out


# ---------------------------------------------------------------- fallback ---

def fallback_rows(case, states, horizon_hours):
    """Single-state history: launch-window prior, honestly wide spreads.

    The frozen page shows a cohort with only a few hours of play (its rates
    are launch-night low), and the first refresh after release brings the
    launch crowd: the kit's own rules describe the tracked count going from
    638 to 3,868 at the first refresh. The prior therefore expects the count
    to multiply several-fold when the case is young, and the incoming cohort
    to have earned each trophy at a fraction of the frozen cohort's rate set
    by the ratio of their play windows (rho < 1: they have played less).
    """
    _, players, _, rates = states[-1]
    checkpoint_age_h = 3.0
    try:
        rel = case["game"].get("release_date")
        if rel:
            rel_t = parse_time(rel + "T00:00:00Z")
            checkpoint_age_h = max(0.5, (parse_time(case["at"]) - rel_t) / 3600.0)
    except Exception:
        pass
    growth = 1.0 + 4.5 * math.exp(-checkpoint_age_h / 10.0) * min(1.0, horizon_hours)
    p_star = players * growth
    rho = min(1.0, max(0.15, (0.5 * horizon_hours + 0.5) / (checkpoint_age_h + 1.0)))
    spread = 0.85 + 0.3 * math.sqrt(horizon_hours / 24.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i]
        r = (rate if rate is not None else 0.0) / 100.0
        earners = r * players
        if trophy.get("tier") == "platinum":
            earners_star = earners
        else:
            f = 1.0 - (1.0 - r) ** rho
            earners_star = earners + (p_star - players) * f
        rate_star = clamp_pct(100.0 * min(earners_star, p_star) / p_star) if p_star > 0 else 0.01
        lg = logit(rate_star)
        qs = [round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


# -------------------------------------------------------------------- main ---

def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    states, t0 = distinct_states(case)
    horizon_hours = case["horizon_minutes"] / 60.0
    if len(states) == 1:
        return fallback_rows(case, states, horizon_hours)
    last_read_hours = (parse_time(case["history"][-1]["at"]) - t0) / 3600.0
    t_target = last_read_hours + horizon_hours
    t_last, p_last = states[-1][0], states[-1][1]
    # The answer is the state of the last refresh before the target read, so
    # it lags the smooth path by an unknown part of the typical refresh gap.
    # The projection therefore averages the path at the target with the path
    # one full gap earlier, and the spread carries that whole range.
    gaps = [states[j][0] - states[j - 1][0] for j in range(1, len(states))]
    gaps.sort()
    gap_med = gaps[len(gaps) // 2] if gaps else 0.0
    t_lag = max(t_last, t_target - gap_med)
    p_star = project_players(states, t_target)
    p_lag = project_players(states, t_lag)

    series = []
    for i in range(len(case["trophies"])):
        col = []
        for _, players, _, rates in states:
            rate = rates[i]
            col.append(logit(min(99.98, max(0.02, rate))) if rate is not None else None)
        series.append(col)
    _, _, fits = fit_drivers(states, series)

    # Typical refresh step in logit space, from the last few state changes.
    steps = []
    for j in range(max(1, len(states) - 6), len(states)):
        for i in range(len(case["trophies"])):
            a, b = states[j - 1][3][i], states[j][3][i]
            if a is not None and b is not None:
                steps.append(abs(logit(b) - logit(a)))
    steps.sort()
    refresh_step = steps[len(steps) // 2] if steps else 0.2

    du = math.log(p_star / p_last)
    dv = math.log1p(t_target) - math.log1p(t_last)
    du_lag = math.log(p_lag / p_last)
    dv_lag = math.log1p(t_lag) - math.log1p(t_last)
    base_spread = SIGMA_DAY * math.sqrt(max(horizon_hours, 0.0) / 24.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        beta, gamma, resid = fits[i]
        # Over a long horizon an unsaturated trophy's progression does not
        # run backwards: dilution is carried by beta alone, so a negative
        # gamma (an artefact of the two drivers moving together in the
        # observed window) is floored at zero.
        if horizon_hours > 48.0 and gamma < 0.0:
            _lr = states[-1][3][i]
            if _lr is not None and _lr < 90.0:
                gamma = 0.0
        last_rate = states[-1][3][i]
        earners_last = (last_rate or 0.0) / 100.0 * p_last
        y = logit(min(99.98, max(0.02, last_rate))) if last_rate is not None else logit(0.02)
        y_full = y + beta * du + gamma * dv
        y_lag = y + beta * du_lag + gamma * dv_lag
        y_star = 0.5 * (y_full + y_lag)
        rate_star = inv_logit(y_star)
        # Count-consistency: implied earners may not fall below observed.
        if rate_star / 100.0 * p_star < earners_last:
            rate_star = 100.0 * earners_last / p_star
            y_star = logit(rate_star)
            y_full = y_lag = y_star
        rate_star = clamp_pct(rate_star)
        sigma = math.sqrt(base_spread ** 2 + resid ** 2 + (0.35 * refresh_step) ** 2
                          + (0.5 * (y_full - y_lag)) ** 2)
        lg = logit(rate_star)
        qs = [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in Z]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
'''

_v6_ns = {"__name__": "v6_engine", "__file__": __file__}
exec(compile(_V6_SOURCE, "v6_engine", "exec"), _v6_ns)
_v6_forecast_case = _v6_ns["forecast_case"]


# ---- blend ---------------------------------------------------------------
# Median: equal logit-average of the two engines' medians (per-path tests
# at 0.65/0.35 changed nothing material; 0.5 is the robust split).
# Spread: per-side maximum of the widened v3 deviation and 0.9x the v6
# deviation, so each row inherits whichever engine is more cautious on
# that side. Across neutral hurdle/smooth worlds and two independent
# violent-growth suites, this blend is the only candidate that clears
# the house persistence baseline in every final-shaped cell.
def _blogit(p):
    p = min(99.99, max(0.01, p)) / 100.0
    return math.log(p / (1.0 - p))


def _binv_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def forecast_case(case):
    f3 = _v3w_forecast_case(case)
    f6 = _v6_forecast_case(case, "final")
    rows3 = {t["id"]: t["q"] for t in f3["trophies"]}
    rows6 = {t["id"]: t["q"] for t in f6["trophies"]}
    out = []
    for t in case["trophies"]:
        tid = t["id"]
        q3, q6 = rows3[tid], rows6[tid]
        med = _binv_logit(0.5 * (_blogit(q3[2]) + _blogit(q6[2])))
        lm = _blogit(med)
        row = []
        for i in range(5):
            d3 = _blogit(q3[i]) - _blogit(q3[2])
            d6 = _blogit(q6[i]) - _blogit(q6[2])
            if i < 2:
                d = -max(-d3, -0.9 * d6)
            elif i > 2:
                d = max(d3, 0.9 * d6)
            else:
                d = 0.0
            row.append(round(min(100.0, max(0.0, _binv_logit(lm + d))), 4))
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        out.append({"id": tid, "q": row})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
