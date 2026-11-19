"""gta6-trophy-rarity-forecast-01 entry: a refresh-aware, count-consistent forecaster.

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
