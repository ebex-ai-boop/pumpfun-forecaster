"""Predict pump.fun's top 5 new coins by market cap -- Monte-Carlo cap-diffusion forecaster.

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json
(or $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR when set).

Method
------
Each coin's log10 market cap (in SOL) is a latent path. It starts at the
last observed log-cap and follows an Ornstein-Uhlenbeck process: it
diffuses with a per-coin sigma (residual std, shrunk toward the global
median sigma, floored) and reverts toward the chart's median log-cap
(the pack level) at speed theta. Trend-following drift is ZERO: 10-minute
trends are pure noise, and any trend extrapolation hurts at both
horizons. The reversion ramps up over the first hours (no reversion is
visible at 10-minute scale; full reversion is measured at ~10h), and the
OU pull is integrated in small substeps so a long first read interval
cannot destabilise the Euler step. In every simulated path the coins are
re-ranked by simulated cap at each of the task's seven answer reads; the
fraction of (path, read) pairs in which a coin holds place k is its
forecast probability for that place, and the simulated caps at each place
give that place's size-bin probabilities. The same simulation prices both
the order and the sizes. The simulation is blended with the house
persistence baseline, with the persistence weight decaying over ~2h.

Calibration (v2: 5 sweep rounds on the PC; v3: mean-reversion round on
the VM, 3 short-horizon splits of the kit history scored with the
competition's log-score rule, plus THREE long-horizon pseudo-reads
~10.4h / ~11.4h / ~12.2h after the kit cut built from live DexScreener
caps of all 454 kit mints; multi-seed spread 0.04 short / 0.38 long
over seeds 12345/999/2026):
  * drift must be ZERO (0.0): 10-minute trends are pure noise;
  * mean reversion is the dominant long-horizon effect (theta 1.0,
    ramp 3h, level = chart median): over ~10.4h the cohort's median
    log-cap shifted only +0.005 dex while the top-5 crashed from
    log-cap 3.28-3.56 to 1.64-2.12 and the new top-5 came from ranks
    14-31. v2's long-horizon size score was pure floor (-30.2); v3's
    is -3.5. A common downward drift would be wrong (it moves the
    median too); only OU pulls the extremes while leaving the pack;
  * with reversion on, the vol floor drops to 0.30 (the stationary
    spread is sigma/sqrt(2*theta)) and the persistence size sd back
    to 0.35 (tighter fits short horizons once the MC handles the
    long-horizon spread);
  * tempering helps (alpha 0.6), persistence must decay fast
    (tau 2.0h), and only 15 coins per place are named: at long horizon
    the cohort churns completely, so the unnamed share prices the
    churn better than naming a 16th-20th coin at a tiny p;
  * tested and REJECTED: a lower reversion level for pool (graduated)
    coins (+0.18 order, -0.18 size on both snapshots, net ~0 --
    mean reversion already captures it), and persist_tau_h 3.0
    (hurts the short score);
  * shipped mean -37.89 short-horizon (vs house baseline -37.77, vs
    v2 -37.71) and -27.3 / -27.1 / -26.7 on the three long
    pseudo-reads (vs baseline -72.14, vs v2 -54.30 / -54.32 /
    -54.27): the simulation now beats persistence by ~+27 points
    at the final's 7-13h horizon, confirmed on a held-out third
    snapshot taken after tuning.

Standard library only, no network, deterministic: a fixed seed, ties
broken by mint address, so two runs on the same history are byte-identical.
"""
import bisect
import json
import math
import os
import random
from datetime import datetime

# Tuned on the PC backtest (see docstring); do not retune per run.
PARAMS = {
    "n_paths": 3000,
    "drift_scale": 0.0,      # no trend-following: 10-min trends are noise
    "drift_halflife_h": 6.0,  # unused at drift_scale 0.0, kept for shape
    "drift_cap": 1.0,        # max |drift|, log10 SOL per hour
    "vol_floor": 0.30,       # min hourly log10-cap sigma (0.40 in v2;
                             # with mean reversion the stationary spread
                             # is sigma/sqrt(2*theta), so 0.30 suffices)
    "vol_mult": 1.0,
    "alpha": 0.6,            # tempering exponent on occupancy
    "floor": 0.0005,         # added to every candidate's occupancy before tempering
    "named_sum": 0.5,        # mass named per place; the rest stays unnamed
    "max_named": 15,         # name only the top-N coins per place; the rest
                             # take the unnamed share, which prices them better
                             # than a tiny named p would
    "persist_tau_h": 2.0,    # horizon half-life of the persistence blend
    "persist_sd": 0.35,      # sd of the persistence size normal (log10 SOL);
                             # 0.35 (v2 used 0.5): tighter fits short
                             # horizons better once the MC handles the
                             # long-horizon spread via mean reversion
    "size_floor": 0.003,     # per-bin floor for sizes (fat tails: caps crash)
    "size_named": 0.99,      # mass named per place's sizes
    "cand_top": 60,          # last chart's top N are candidates
    "cand_recent": 30,       # plus any coin top-N in the last 3 marks
    "seed": 12345,
    # v3: Ornstein-Uhlenbeck mean reversion toward the chart's median
    # log-cap. Measured, not guessed: over ~10.4h the kit cohort's
    # same-mint median log-cap shifted only +0.005 dex, while the top-5
    # (log-cap 3.28-3.56) crashed to 1.64-2.12 and the new top-5 came
    # from ranks 14-31. Leaders revert to the pack; the pack stays put.
    # A common downward drift would be wrong (it moves the median too);
    # OU pulls only the extremes.
    "mr_theta": 1.0,         # mean-reversion speed, per hour
    "mr_level": None,        # None = median log-cap of the last chart
    # v3b: reversion ramps up over the first mr_ramp_h hours. The kit
    # history shows ~no reversion at 10-min scale, while the 10h
    # snapshots show full reversion; the shape between is unobserved.
    # A ramp keeps short horizons (practice) intact and costs nothing
    # on the final, whose reads are all 7-13h out (fully reverted by
    # then either way). 0.0 = no ramp (constant theta from t=0).
    "mr_ramp_h": 3.0,
    # Numerical safety: the OU pull is integrated in substeps of at most
    # mr_max_dt hours. Euler with theta*dt > 1 OSCILLATES and reverses
    # the ranking (the final's first read step is 7h; theta=1.0 would
    # give theta*dt = 7). 0.25h keeps theta*dt <= 0.25 for theta <= 1.
    "mr_max_dt": 0.25,
    # v3c (TESTED AND REJECTED): pool (graduated) coins revert toward a
    # lower level than curve coins. Measured: 14 of the last mark's 15
    # pool coins fell OUT of the snapshot top-100, while 78/185 curve
    # coins stayed in. But the delta bought +0.18 order and lost -0.18
    # size on both snapshots (net ~0): mean reversion already captures
    # the effect. Kept at 0.0; do not re-enable without fresh evidence.
    "mr_pool_delta": 0.0,
}


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def _logcap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _load_marks(input_dir):
    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    return marks


def forecast(input_dir, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    marks = _load_marks(input_dir)
    if not marks:
        raise ValueError("the history holds no ok mark")
    task = json.load(open(os.path.join(input_dir, "task.json"), encoding="utf-8"))
    places = task["places"]
    edges = task["size"]["edges"]
    reads = sorted(_parse_ts(r) for r in task["reads"])
    t_cut = _parse_ts(marks[-1]["at"])
    offsets = [max(0.0, r - t_cut) for r in reads]

    # Per-mint log-cap series.
    series = {}
    for m in marks:
        t = _parse_ts(m["at"])
        for row in m["rows"]:
            try:
                y = _logcap(row)
            except (KeyError, ValueError):
                continue
            series.setdefault(row["mint"], []).append((t, y, row["rank"]))

    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise ValueError("the last mark ranks fewer coins than places")
    last_rank = {r["mint"]: r["rank"] for r in last_rows}
    last_cap = {}
    for r in last_rows:
        try:
            last_cap[r["mint"]] = _logcap(r)
        except (KeyError, ValueError):
            pass

    # Candidates: last chart's top N, plus recent top-N climbers.
    cand = {r["mint"] for r in last_rows[: p["cand_top"]]}
    for m in marks[-3:]:
        for r in m["rows"]:
            if r["rank"] <= p["cand_recent"]:
                cand.add(r["mint"])
    cand = sorted(cand, key=lambda u: (last_rank.get(u, 999), u))
    start = {}
    for uid in cand:
        if uid in last_cap:
            start[uid] = last_cap[uid]
        elif series.get(uid):
            start[uid] = series[uid][-1][1]
        else:
            start[uid] = 1.0  # unseen fallback; should not happen

    # Per-coin drift (log10 SOL/hour) and vol from weighted linear fits.
    slopes, vols = {}, {}
    all_slopes, all_vols = [], []
    for uid in cand:
        s = series.get(uid, [])
        if len(s) >= 3:
            ws = [1.0 + (t - s[0][0]) / max(1e-9, s[-1][0] - s[0][0])
                  for t, _, _ in s]
            sw = sum(ws)
            mt = sum(w * t for w, (t, _, _) in zip(ws, s)) / sw
            my = sum(w * y for w, (_, y, _) in zip(ws, s)) / sw
            num = sum(w * (t - mt) * (y - my) for w, (t, y, _) in zip(ws, s))
            den = sum(w * (t - mt) ** 2 for w, (t, _, _) in zip(ws, s))
            slope = num / den if den > 0 else 0.0
            resid = [y - (my + slope * (t - mt)) for t, y, _ in s]
            vol = math.sqrt(sum(r * r for r in resid) / max(1, len(s) - 2))
            span = max(1e-9, s[-1][0] - s[0][0])
            vol = vol / math.sqrt(span)  # per-hour sigma
            slopes[uid] = slope
            vols[uid] = vol
            all_slopes.append(slope)
            all_vols.append(vol)
        elif len(s) == 2:
            span = max(1e-9, s[1][0] - s[0][0])
            slopes[uid] = (s[1][1] - s[0][1]) / span
            all_slopes.append(slopes[uid])
    g_drift = _median(all_slopes)
    g_vol = _median(all_vols) if all_vols else 0.3
    drift, sigma = {}, {}
    for uid in cand:
        n = len(series.get(uid, []))
        w = n / (n + 3.0)
        d = w * slopes.get(uid, g_drift) + (1.0 - w) * g_drift
        drift[uid] = max(-p["drift_cap"],
                         min(p["drift_cap"], p["drift_scale"] * d))
        v = w * vols.get(uid, g_vol) + (1.0 - w) * g_vol
        sigma[uid] = max(p["vol_floor"], p["vol_mult"] * v)

    # Mean-reversion level: the chart's median log-cap (the pack level
    # leaders revert toward). Computed from the last mark, deterministic.
    # Pool coins revert toward a lower level (mr_pool_delta < 0).
    mr_level = p["mr_level"]
    if mr_level is None:
        mr_level = _median(list(last_cap.values()))
    venue = {r["mint"]: r.get("venue", "curve") for r in last_rows}
    level = {uid: mr_level + (p["mr_pool_delta"] if venue.get(uid) == "pool" else 0.0)
             for uid in cand}

    # Monte-Carlo: one path is a full trajectory; record at each read.
    # The OU pull is integrated in substeps (<= mr_max_dt) so a long
    # first read interval cannot destabilise the Euler step.
    rng = random.Random(p["seed"])
    n_reads = len(offsets)
    occ = {k: {uid: 0 for uid in cand} for k in range(1, places + 1)}
    size_hits = {k: [0] * (len(edges) + 1) for k in range(1, places + 1)}
    total_slots = p["n_paths"] * n_reads
    ramp = p["mr_ramp_h"]
    max_dt = p["mr_max_dt"]
    for _ in range(p["n_paths"]):
        x = {uid: start[uid] + rng.gauss(0, 0.02) for uid in cand}
        t_prev = 0.0
        for off in offsets:
            dt = off - t_prev
            t_prev = off
            if dt > 0:
                n_sub = max(1, math.ceil(dt / max_dt))
                h = dt / n_sub
                sq = math.sqrt(h)
                t0 = off - dt
                for j in range(n_sub):
                    t_mid = t0 + (j + 0.5) * h
                    decay = 0.5 ** (t_mid / p["drift_halflife_h"])
                    th_eff = (p["mr_theta"] * min(1.0, t_mid / ramp)
                              if ramp > 0 else p["mr_theta"])
                    for uid in cand:
                        ou = -th_eff * (x[uid] - level[uid])
                        x[uid] += (drift[uid] * decay + ou) * h + rng.gauss(0, sigma[uid] * sq)
            order = sorted(cand, key=lambda u: (-x[u], u))
            for k in range(1, places + 1):
                uid = order[k - 1]
                occ[k][uid] += 1
                size_hits[k][bisect.bisect_right(edges, x[uid])] += 1

    # Blend weight: short horizon -> persistence, long horizon -> simulation.
    mean_offset = sum(offsets) / max(1, len(offsets))
    w_persist = math.exp(-mean_offset / p["persist_tau_h"])

    def _norm_above(x, mean, sd):
        if x == float("-inf"):
            return 1.0
        if x == float("inf"):
            return 0.0
        return 0.5 * (1.0 - math.erf((x - mean) / (sd * math.sqrt(2.0))))

    micro = 1_000_000
    probabilities = []
    blended = {}  # (mint, place) -> blended p, for the top5 score
    for k in range(1, places + 1):
        raw = {}
        for uid in cand:
            q = occ[k][uid] / total_slots
            raw[uid] = (q + p["floor"]) ** p["alpha"]
        s = sum(raw.values())
        mc = {uid: p["named_sum"] * q / s for uid, q in raw.items()}
        # Persistence rows, same as the house baseline, scaled to named_sum.
        pr = {last_rows[k - 1]["mint"]: 0.30}
        if k > 1:
            pr[last_rows[k - 2]["mint"]] = pr.get(last_rows[k - 2]["mint"], 0.0) + 0.10
        if len(last_rows) > k:
            pr[last_rows[k]["mint"]] = pr.get(last_rows[k]["mint"], 0.0) + 0.10
        ps = sum(pr.values())
        persist = {uid: p["named_sum"] * v / ps for uid, v in pr.items()}
        for uid in cand:
            if uid not in mc and uid not in persist:
                continue
            b = w_persist * persist.get(uid, 0.0) + (1.0 - w_persist) * mc.get(uid, 0.0)
            blended[(uid, k)] = b
        # Name only the top-N by blended p: a coin named at a tiny p scores
        # worse than the unnamed share would give it, so the tail is left
        # unnamed (its mass returns to the unnamed share).
        ranked = sorted(
            (uid for uid in cand if (uid, k) in blended),
            key=lambda u: (-blended[(u, k)], last_rank.get(u, 999), u))
        for uid in ranked[: p["max_named"]]:
            u = math.floor(blended[(uid, k)] * micro)
            if u > 0:
                probabilities.append({"mint": uid, "place": k, "p": u / micro})

    score = {uid: sum(blended.get((uid, k), 0.0) * (places + 1 - k)
                      for k in range(1, places + 1)) for uid in cand}
    top5 = sorted(cand, key=lambda u: (-score[u], last_rank.get(u, 999), u))[:places]

    sizes = []
    nbins = len(edges) + 1
    for k in range(1, places + 1):
        raw = [h / total_slots + p["size_floor"] for h in size_hits[k]]
        s = sum(raw)
        mc = [p["size_named"] * q / s for q in raw]
        mean = last_cap[last_rows[k - 1]["mint"]]
        persist = []
        for b in range(nbins):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            persist.append(_norm_above(lo, mean, p["persist_sd"])
                           - _norm_above(hi, mean, p["persist_sd"]))
        ps = sum(persist)
        persist = [p["size_named"] * v / ps for v in persist]
        for b in range(nbins):
            val = w_persist * persist[b] + (1.0 - w_persist) * mc[b]
            u = math.floor(val * micro)
            if u > 0:
                sizes.append({"place": k, "bin": b, "p": u / micro})

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task = json.load(open(os.path.join(input_dir, "task.json"), encoding="utf-8"))
    fc = forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task["output"]), "w", encoding="utf-8") as f:
        f.write(json.dumps(fc) + "\n")


if __name__ == "__main__":
    main()
