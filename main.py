"""Predict pump.fun's top 5 new coins by market cap -- Monte-Carlo cap-diffusion forecaster.

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json
(or $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR when set).

Method
------
Each coin's log10 market cap (in SOL) is a latent path. It starts at the
last observed log-cap and follows an Ornstein-Uhlenbeck process: it
diffuses with a per-coin sigma (residual std, shrunk toward the global
median sigma, floored) and reverts toward the chart's median log-cap
(the pack level) at speed theta. A small trend-following drift (0.2)
is kept: the focused search found it helps slightly at short horizons,
but the dominant effects are mean reversion and the horizon-adaptive
persistence size sd. The reversion ramps up over the first hours (no
reversion is visible at 10-minute scale; full reversion is measured at
~10h), and the OU pull is integrated in small substeps so a long first
read interval cannot destabilise the Euler step. In every simulated
path the coins are re-ranked by simulated cap at each of the task's
seven answer reads; the fraction of (path, read) pairs in which a coin
holds place k is its forecast probability for that place, and the
simulated caps at each place give that place's size-bin probabilities.
The same simulation prices both the order and the sizes. The simulation
is blended with the house persistence baseline, with the persistence
weight decaying over ~1.5h.

Calibration (v4 "D": 2 focused random-search rounds on the PC, 602
configs total, scored on 3 short-horizon splits of the kit history plus
TWO long-horizon pseudo-reads ~10.4h / ~11.4h after the kit cut built
from live DexScreener caps of all 454 kit mints; multi-seed spread
0.04 short / 0.38 long over seeds 12345/999/2026; validated on TWO
held-out snapshots taken after tuning, ~12.7h / ~13.1h after the cut):
  * mean reversion is the dominant long-horizon effect (theta 1.2,
    ramp 6h, level = chart median): over ~10.4h the cohort's median
    log-cap shifted only +0.005 dex while the top-5 crashed from
    log-cap 3.28-3.56 to 1.64-2.12 and the new top-5 came from ranks
    14-31. v2's long-horizon size score was pure floor (-30.2); v4's
    is -3.2. A common downward drift would be wrong (it moves the
    median too); only OU pulls the extremes while leaving the pack;
  * horizon-adaptive persistence size sd (0.15 at offsets <= 0.25h,
    ramping to 0.70 at offsets >= 0.30h): measured best sd is 0.20
    at 10-20 min horizons but 0.35+ at 30+ min (the place caps move
    far at 30+ min: median |dlogcap| 0.527). Fixed sd 0.20 helps
    splits 4-5 but hurts split 3 (-6.3); the ramp keeps both;
  * with reversion on, the vol floor is 0.35 (the stationary spread
    is sigma/sqrt(2*theta)); tempering helps (alpha 0.6); persistence
    must decay fast (tau 1.5h); only 12 coins per place are named
    (at long horizon the cohort churns completely, so the unnamed
    share prices the churn better than naming a 13th coin at a tiny
    p); the MC component names more mass (named_sum_mc 0.8) than the
    persistence component (0.5);
  * tested and REJECTED: a lower reversion level for pool (graduated)
    coins (net ~0 -- mean reversion already captures it), persist_tau
    3.0h (hurts short), p=0 exclusion of crashing leaders (its EV
    rests on an unmeasurable assumption), and a separate longer size
    persistence half-life (hurts long);
  * shipped mean -34.82 short-horizon (multi-seed; vs house baseline
    -37.77, vs v3 -37.89, vs v2 -37.71) and -26.46 long (multi-seed
    avg; vs baseline -72.14, vs v2 -54.30): the simulation now beats
    persistence by ~+27 points at the final's 7-13h horizon,
    confirmed on two held-out snapshots (-26.46 and -26.29, both
    beating v3's -26.80 and -26.39).

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
    "drift_scale": 0.2,      # small trend-following: 0.2 won in the focused search
    "drift_halflife_h": 6.0,  # drift decays with this half-life
    "drift_cap": 1.0,        # max |drift|, log10 SOL per hour
    "vol_floor": 0.35,       # min hourly log10-cap sigma (0.40 in v2;
                             # with mean reversion the stationary spread
                             # is sigma/sqrt(2*theta), so 0.35 suffices)
    "vol_mult": 1.0,
    "alpha": 0.6,            # tempering exponent on occupancy
    "floor": 0.0002,         # added to every candidate's occupancy before tempering
    "named_sum": 0.5,        # mass named per place; the rest stays unnamed
    # v4: separate named mass for the persistence and MC components.
    # At SHORT horizons persistence is strong (the top coins mostly
    # stay), so the persistence rows deserve MORE than 0.5 total mass;
    # at LONG horizons the MC component keeps 0.5 (churn is priced by
    # the unnamed share). The blend weight already encodes the horizon,
    # so the blended total named mass adapts automatically: ~S_persist
    # at short horizons, ~S_mc at long ones. None = fall back to
    # named_sum (v3 behaviour).
    "named_sum_persist": None,
    "named_sum_mc": 0.8,
    "max_named": 12,         # name only the top-N coins per place; the rest
                             # take the unnamed share, which prices them better
                             # than a tiny named p would
    "persist_tau_h": 1.5,    # horizon half-life of the persistence blend
    # v4: separate blend half-life for SIZES. The place-k cap is far
    # more stable than any single coin's cap (new launches refill the
    # top-5 at ~the current level), so sizes should stay persistent
    # much longer than the order does; the MC size component (mean
    # reverted toward the pack median) is actively wrong at short
    # horizons. None = same as persist_tau_h (v3 behaviour).
    "persist_tau_size_h": None,
    "persist_sd": 0.35,      # sd of the persistence size normal (log10 SOL);
                             # 0.35 (v2 used 0.5): tighter fits short
                             # horizons better once the MC handles the
                             # long-horizon spread via mean reversion
    # v4: horizon-adaptive persistence size sd. Measured (kit history):
    # best sd is 0.20 at 10-20 min horizons but 0.35 at 30+ min (the
    # place caps move far at 30+ min: median |dlogcap| 0.527). Fixed
    # sd 0.20 helps splits 4-5 (+1.4/+2.0) but hurts split 3 (-6.3).
    # Piecewise-linear ramp from sd_short (offset <= t0) to sd_long
    # (offset >= t1). None values = fixed persist_sd (v3 behaviour).
    "persist_sd_short": 0.15,
    "persist_sd_long": 0.7,
    "persist_sd_t0_h": 0.25,
    "persist_sd_t1_h": 0.3,
    "size_floor": 0.002,     # per-bin floor for sizes (fat tails: caps crash)
    "size_named": 0.97,      # mass named per place's sizes
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
    "mr_theta": 1.2,         # mean-reversion speed, per hour
    "mr_level": None,        # None = median log-cap of the last chart
    # v3b: reversion ramps up over the first mr_ramp_h hours. The kit
    # history shows ~no reversion at 10-min scale, while the 10h
    # snapshots show full reversion; the shape between is unobserved.
    # A ramp keeps short horizons (practice) intact and costs nothing
    # on the final, whose reads are all 7-13h out (fully reverted by
    # then either way). 0.0 = no ramp (constant theta from t=0).
    "mr_ramp_h": 6.0,
    # Numerical safety: the OU pull is integrated in substeps of at most
    # mr_max_dt hours. Euler with theta*dt > 1 OSCILLATES and reverses
    # the ranking (the final's first read step is 7h; theta=1.0 would
    # give theta*dt = 7). 0.25h keeps theta*dt <= 0.3 for theta <= 1.2.
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
    tau_size = p["persist_tau_size_h"] if p["persist_tau_size_h"] is not None else p["persist_tau_h"]
    w_persist_size = math.exp(-mean_offset / tau_size)

    def _norm_above(x, mean, sd):
        if x == float("-inf"):
            return 1.0
        if x == float("inf"):
            return 0.0
        return 0.5 * (1.0 - math.erf((x - mean) / (sd * math.sqrt(2.0))))

    micro = 1_000_000
    probabilities = []
    blended = {}  # (mint, place) -> blended p, for the top5 score
    ns_persist = p["named_sum_persist"] if p["named_sum_persist"] is not None else p["named_sum"]
    ns_mc = p["named_sum_mc"] if p["named_sum_mc"] is not None else p["named_sum"]
    for k in range(1, places + 1):
        raw = {}
        for uid in cand:
            q = occ[k][uid] / total_slots
            raw[uid] = (q + p["floor"]) ** p["alpha"]
        s = sum(raw.values())
        mc = {uid: ns_mc * q / s for uid, q in raw.items()}
        # Persistence rows, same as the house baseline, scaled to named_sum.
        pr = {last_rows[k - 1]["mint"]: 0.30}
        if k > 1:
            pr[last_rows[k - 2]["mint"]] = pr.get(last_rows[k - 2]["mint"], 0.0) + 0.10
        if len(last_rows) > k:
            pr[last_rows[k]["mint"]] = pr.get(last_rows[k]["mint"], 0.0) + 0.10
        ps = sum(pr.values())
        persist = {uid: ns_persist * v / ps for uid, v in pr.items()}
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
    # Horizon-adaptive persistence size sd (see PARAMS).
    if p["persist_sd_short"] is not None and p["persist_sd_long"] is not None:
        t0, t1 = p["persist_sd_t0_h"], p["persist_sd_t1_h"]
        if mean_offset <= t0:
            sd_persist = p["persist_sd_short"]
        elif mean_offset >= t1:
            sd_persist = p["persist_sd_long"]
        else:
            f = (mean_offset - t0) / (t1 - t0)
            sd_persist = p["persist_sd_short"] + f * (p["persist_sd_long"] - p["persist_sd_short"])
    else:
        sd_persist = p["persist_sd"]
    for k in range(1, places + 1):
        raw = [h / total_slots + p["size_floor"] for h in size_hits[k]]
        s = sum(raw)
        mc = [p["size_named"] * q / s for q in raw]
        mean = last_cap[last_rows[k - 1]["mint"]]
        persist = []
        for b in range(nbins):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            persist.append(_norm_above(lo, mean, sd_persist)
                           - _norm_above(hi, mean, sd_persist))
        ps = sum(persist)
        persist = [p["size_named"] * v / ps for v in persist]
        for b in range(nbins):
            val = w_persist_size * persist[b] + (1.0 - w_persist_size) * mc[b]
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
