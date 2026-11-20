#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster -- hybrid v4.

Two regimes, chosen by how many DISTINCT snapshots a case history holds
(a snapshot collapses consecutive reads with identical players_tracked
and identical rates):

  Path A (one distinct snapshot): the page is frozen, so the forecast is
  about the next Exophase refresh.  Earners are recovered as integers,
  the pool is projected one refresh ahead with a decaying growth rate,
  and each trophy's new earners follow a hurdle-Poisson law whose
  quantiles are read off in logit space, widened asymmetrically, and
  finally shrunk around their own median by the CAL factor (a narrower
  spread than the uncalibrated law, tuned so the 90% band keeps
  roughly 0.88-0.92 coverage instead of over-covering).

  Path B (several distinct snapshots): damped log-time least-squares
  trends of counts (pool, total earners, and per-trophy earners shrunk
  toward their share of the total trend) are extrapolated to the last
  refresh at or before the target, with a symmetric logit spread whose
  width comes from the fit residuals.

Deterministic, Python 3 standard library only, no network.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Spread calibration for Path A: logit deltas around the row median are
# multiplied by CAL after the hurdle-Poisson quantiles are widened.
# CAL was chosen on a local truth-region sweep (four hurdle-Poisson
# answer laws at the observed refresh pool, enumerated exactly): it is
# the only swept value whose mean 90% coverage across those laws lands
# in [0.88, 0.92] (0.911), and it also gives the lowest mean expected
# pinball loss of the sweep (0.09169, vs 0.09371 uncalibrated at 1.00).
# The median itself is never touched by the shrink, so the sharp
# single-snapshot centre is preserved; only the spread is narrowed
# from the uncalibrated law's over-coverage (0.977) toward nominal.
CAL = 0.67

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZVALS = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)

REFRESH_MINUTES = 280.0          # assumed Exophase refresh cadence
POOL_GROWTH0 = 0.386             # hourly pool growth at release
POOL_DECAY_HOURS = 48.0          # pool growth decay timescale
HURDLE_A = 0.016                 # earners growth law: mu = A * e**GAMMA
HURDLE_GAMMA = 2.5
HURDLE_PI = 0.10                 # probability a trophy gains nobody
SIGMA_EXTRA_LO = 0.16            # extra logit width below the median
SIGMA_EXTRA_HI = 0.08            # extra logit width above the median
FLAT_SIGMA = 0.05                # spread when the page is taken as-is


# ---------------------------------------------------------------- helpers

def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if text is None:
        return None
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def tidy_row(values):
    """Round to 4 decimals, clamp to [0, 100], enforce non-decreasing."""
    out = []
    prev = 0.0
    for v in values:
        v = round(min(100.0, max(0.0, v)), 4)
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def quantile_row(centre, sigma):
    """Symmetric logit-normal quantile row around a centre rate."""
    if centre is None:
        centre = 0.01
    base = logit_pct(centre)
    return tidy_row([inv_logit_pct(base + z * sigma) for z in ZVALS])


def read_time(entry):
    return parse_time(entry.get("at") or entry.get("slot"))


def snapshots_of(history):
    """Collapse consecutive identical reads into distinct snapshots.

    Each snapshot keeps its players count, rates, trophies-earned total
    (from its latest read) and its first/last read times.
    """
    snaps = []
    for entry in history or []:
        players = entry.get("players_tracked")
        rates = list(entry.get("rates") or [])
        key = (players, tuple(rates))
        when = read_time(entry)
        if snaps and snaps[-1]["key"] == key:
            snaps[-1]["last"] = when
            snaps[-1]["earned"] = entry.get("trophies_earned")
        else:
            snaps.append({
                "key": key,
                "players": players,
                "rates": rates,
                "earned": entry.get("trophies_earned"),
                "first": when,
                "last": when,
            })
    return snaps


def recover_earners(rates, players, total):
    """Integer earners per trophy from printed rates.

    Starts from rounding rate*players/100, then distributes the gap to
    the page's trophies-earned total one unit at a time, largest
    fractional parts first when the gap is positive and smallest when
    it is negative, never driving a count below zero.
    """
    earners = []
    fracs = []
    for rate in rates:
        if rate is None or players is None:
            earners.append(None)
            fracs.append(0.0)
            continue
        raw = rate / 100.0 * players
        base = int(math.floor(raw + 0.5))
        earners.append(base)
        fracs.append(raw - math.floor(raw))
    known = [i for i, e in enumerate(earners) if e is not None]
    if total is not None and known:
        diff = int(total) - sum(earners[i] for i in known)
        if 0 < abs(diff) <= len(known):
            if diff > 0:
                order = sorted(known, key=lambda i: (-fracs[i], i))
                step = 1
            else:
                order = sorted(known, key=lambda i: (fracs[i], i))
                step = -1
            left = abs(diff)
            for i in order:
                if left <= 0:
                    break
                if step < 0 and earners[i] <= 0:
                    continue
                earners[i] += step
                left -= 1
    return earners


def poisson_pmf(mu):
    """Poisson pmf as {k: p}, built outwards from the mode.

    Starting at the mode with its log-pmf avoids underflow for large
    mu; the upward recursion stops once the tail drops below 1e-13
    and the result is renormalised.
    """
    if mu <= 0:
        return {0: 1.0}
    mode = int(mu)
    logp = -mu + mode * math.log(mu) - math.lgamma(mode + 1)
    pmf = {mode: math.exp(logp)}
    p = pmf[mode]
    k = mode
    while k > 0:
        p = p * k / mu
        k -= 1
        pmf[k] = p
    p = pmf[mode]
    k = mode
    while p >= 1e-13:
        p = p * mu / (k + 1)
        k += 1
        pmf[k] = p
        if k > mode + 100000:  # safety net, never expected
            break
    total = sum(pmf.values())
    return {k: v / total for k, v in pmf.items()}


def cdf_quantile(points, tau):
    """Quantile of a discrete CDF given as (cdf, rate) points.

    Interpolates linearly in logit space between consecutive points;
    outside the CDF range the end rates are returned.
    """
    if tau <= points[0][0]:
        return points[0][1]
    if tau >= points[-1][0]:
        return points[-1][1]
    for j in range(1, len(points)):
        f1, r1 = points[j]
        if tau <= f1:
            f0, r0 = points[j - 1]
            if f1 <= f0:
                return r1
            w = (tau - f0) / (f1 - f0)
            return inv_logit_pct(logit_pct(r0) + w * (logit_pct(r1) - logit_pct(r0)))
    return points[-1][1]


# ------------------------------------------------------------------ path A

def forecast_single(case, snap, trophies):
    players = snap["players"]
    rates = snap["rates"]
    earners = recover_earners(rates, players, snap["earned"])
    at_time = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    first_read = snap["first"]
    refresh_at = None
    if first_read is not None:
        refresh_at = first_read + timedelta(minutes=REFRESH_MINUTES)

    stale = (
        refresh_at is None
        or at_time is None
        or target is None
        or refresh_at <= at_time
        or refresh_at > target + timedelta(minutes=20)
    )
    if stale:
        return [quantile_row(rates[i] if i < len(rates) else None, FLAT_SIGMA)
                for i in range(len(trophies))]

    # One refresh ahead.  The frozen page's content dates from when it
    # was first read, so pool growth is aged from that anchor.
    anchor = first_read
    release = case.get("game", {}).get("release_date")
    rel_dt = parse_time(release) if release else None
    if rel_dt is not None and rel_dt.hour == 0 and "T" not in str(release):
        rel_dt = rel_dt.replace(hour=0, minute=0, second=0)
    if anchor is not None and rel_dt is not None:
        age_hours = max(0.0, (anchor - rel_dt).total_seconds() / 3600.0)
    else:
        age_hours = 0.0
    growth = POOL_GROWTH0 * math.exp(-age_hours / POOL_DECAY_HOURS)
    span_hours = REFRESH_MINUTES / 60.0
    if players and players > 0:
        players_hat = players * min(50.0, math.exp(growth * span_hours))
    else:
        players_hat = float(players or 0)
    cap_e = int(math.floor(players_hat)) if players_hat > 0 else 0

    rows = []
    for i in range(len(trophies)):
        rate = rates[i] if i < len(rates) else None
        e = earners[i] if i < len(earners) else None
        if rate is None or e is None or not players:
            rows.append(quantile_row(0.01, FLAT_SIGMA))
            continue
        if e >= players:
            rows.append([100.0, 100.0, 100.0, 100.0, 100.0])
            continue
        mu = HURDLE_A * (e ** HURDLE_GAMMA) * (span_hours / (REFRESH_MINUTES / 60.0)) if e > 0 else 0.0
        mu = min(mu, e * (1.5 * players_hat / players - 1.0))
        mu = min(mu, max(0.0, players_hat - e))
        mu = max(0.0, mu)
        # Hurdle-Poisson distribution over the earners count at refresh.
        dist = {}
        for k, p in poisson_pmf(mu).items():
            big_e = min(e + k, cap_e)
            dist[big_e] = dist.get(big_e, 0.0) + (1.0 - HURDLE_PI) * p
        dist[e] = dist.get(e, 0.0) + HURDLE_PI
        points = []
        cdf = 0.0
        for big_e in sorted(dist):
            cdf += dist[big_e]
            points.append((cdf, 100.0 * big_e / players_hat if players_hat > 0 else 0.0))
        logits = []
        for tau, z in zip(LEVELS, ZVALS):
            q = cdf_quantile(points, tau)
            extra = SIGMA_EXTRA_LO if z < 0 else SIGMA_EXTRA_HI
            logits.append(logit_pct(q) + z * extra)
        median_logit = logits[2]
        shrunk = [inv_logit_pct(median_logit + CAL * (lg - median_logit))
                  for lg in logits]
        rows.append(tidy_row(shrunk))
    return rows


# ------------------------------------------------------------------ path B

def lin_fit(xs, ys):
    """Least-squares slope/intercept and residual sd of ys on xs."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return slope, intercept, resid


def forecast_multi(case, snaps, trophies):
    snaps = snaps[-12:]
    first_time = snaps[0]["first"]
    last = snaps[-1]
    target = parse_time(case.get("target"))

    # Cadence: median gap between consecutive distinct snapshots.
    gaps = []
    for a, b in zip(snaps, snaps[1:]):
        if a["first"] is not None and b["first"] is not None:
            gaps.append((b["first"] - a["first"]).total_seconds() / 60.0)
    if gaps:
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        cadence = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        if cadence <= 0:
            cadence = REFRESH_MINUTES
    else:
        cadence = REFRESH_MINUTES

    # Last refresh at or before the target, on the cadence grid that
    # starts at the last snapshot's first read.
    refresh_at = last["first"]
    if refresh_at is not None and target is not None:
        step = timedelta(minutes=cadence)
        while refresh_at + step <= target:
            refresh_at = refresh_at + step

    def x_of(when):
        if when is None or first_time is None:
            return 0.0
        hours = (when - first_time).total_seconds() / 3600.0
        return math.log(1.0 + hours / 8.0)

    xs = [x_of(s["first"]) for s in snaps]
    pool_counts = [float(s["players"] or 0) for s in snaps]
    totals = []
    for s in snaps:
        if s["earned"] is not None:
            totals.append(float(s["earned"]))
        else:
            es = recover_earners(s["rates"], s["players"], None)
            totals.append(float(sum(e for e in es if e is not None)))

    pool_slope, _, pool_resid = lin_fit(xs, pool_counts)
    total_slope, _, total_resid = lin_fit(xs, totals)
    pool_slope = max(0.0, pool_slope)
    total_slope = max(0.0, total_slope)
    mean_total = sum(totals) / len(totals) if totals else 0.0

    x_last = xs[-1]
    x_first = xs[0]
    x_ref = x_of(refresh_at)
    dx = x_ref - x_last
    span = x_last - x_first
    damp = span / (span + dx) if (span + dx) > 0 else 1.0

    last_players = float(last["players"] or 0)
    players_hat = max(last_players, last_players + pool_slope * dx * damp)

    rel_pool = pool_resid / last_players if last_players > 0 else 0.0
    rel_total = total_resid / mean_total if mean_total > 0 else 0.0
    rel = max(rel_total, rel_pool)
    if refresh_at is not None and last["first"] is not None:
        days = max(0.0, (refresh_at - last["first"]).total_seconds() / 86400.0)
    else:
        days = 0.0
    sigma = min(0.35, math.sqrt(0.04 ** 2 + rel ** 2 + (0.03 * math.sqrt(days)) ** 2))

    # Per-snapshot earners for every trophy (plain rounding recovery).
    per_snap_earners = [recover_earners(s["rates"], s["players"], None)
                        for s in snaps]
    last_earners = per_snap_earners[-1]

    rows = []
    for i in range(len(trophies)):
        e_last = last_earners[i] if i < len(last_earners) else None
        if e_last is None:
            rows.append(quantile_row(0.01, sigma))
            continue
        if last_players > 0 and e_last >= last_players:
            rows.append(quantile_row(100.0, sigma))
            continue
        pts = [(xs[j], per_snap_earners[j][i])
               for j in range(len(snaps))
               if i < len(per_snap_earners[j]) and per_snap_earners[j][i] is not None]
        if pts:
            mean_e = sum(p[1] for p in pts) / len(pts)
        else:
            mean_e = 0.0
        share = mean_e / mean_total if mean_total > 0 else 0.0
        target_slope = total_slope * share
        if len(pts) >= 2:
            own_slope, _, _ = lin_fit([p[0] for p in pts], [float(p[1]) for p in pts])
            w = mean_e / (mean_e + 25.0)
            slope = w * own_slope + (1.0 - w) * target_slope
        else:
            slope = target_slope
        slope = max(0.0, slope)
        e_hat = min(e_last + slope * dx * damp, players_hat)
        if players_hat > 0:
            median = round(100.0 * e_hat / players_hat, 2)
        else:
            median = 0.01
        rows.append(quantile_row(median, sigma))
    return rows


# -------------------------------------------------------------------- main

def forecast_case(case):
    trophies = case.get("trophies") or []
    snaps = snapshots_of(case.get("history"))
    if not snaps:
        rows = [quantile_row(0.01, FLAT_SIGMA) for _ in trophies]
    elif len(snaps) == 1:
        rows = forecast_single(case, snaps[0], trophies)
    else:
        rows = forecast_multi(case, snaps, trophies)
    return {"trophies": [{"id": t["id"], "q": rows[i]}
                         for i, t in enumerate(trophies)]}


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
