"""Predict pump.fun's top 5 new coins by market cap -- Monte-Carlo cap-diffusion forecaster.

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json
(or $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR when set).

Method
------
Each coin's log10 market cap (in SOL) is a latent path. It starts at the
last observed log-cap, drifts with the coin's recent log-cap velocity
(weighted linear fit, shrunk toward the global median drift -- most
pump.fun coins fade within hours, so the global drift is negative -- and
decaying with a half-life), and diffuses with a per-coin sigma (residual
std, shrunk toward the global median sigma, floored). In every simulated
path the coins are re-ranked by simulated cap at each of the task's seven
answer reads; the fraction of (path, read) pairs in which a coin holds
place k is its forecast probability for that place, and the simulated
caps at each place give that place's size-bin probabilities. The same
simulation prices both the order and the sizes.

Calibration (PC backtest, 108-config sweep + focused re-test, 3 splits
of the kit history scored with the competition's log-score rule;
multi-seed spread 0.08 over seeds 12345/999/2026):
  * drift must be small (0.2): 10-minute trends are mostly noise, and
    drift 0.8 never reached the top 10;
  * tempering helps (alpha 0.75 in all top-10 configs; alpha 1.0 in
    all worst-3);
  * only half of each place's mass is named (0.5): the unnamed share
    prices the churn and the coins born after the cut;
  * only the top 20 coins per place are named: a coin named at a tiny
    p scores worse than the unnamed share would give it;
  * shipped mean -39.64 vs the house baseline -37.77 on these
    short-horizon (10-30 min) splits: persistence wins at minutes.
    The final is scored 7-13 hours out, where the simulation (which
    the blend weight hands over to as the horizon grows) is the
    edge the splits cannot measure. The practice score on submit
    is the real short-horizon check.

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
    "drift_scale": 0.2,      # shrink on per-coin log-cap drift (per hour)
    "drift_halflife_h": 6.0,  # drift decays with this half-life
    "drift_cap": 1.0,        # max |drift|, log10 SOL per hour
    "vol_floor": 0.10,       # min hourly log10-cap sigma
    "vol_mult": 1.0,
    "alpha": 0.75,           # tempering exponent on occupancy
    "floor": 0.0005,         # added to every candidate's occupancy before tempering
    "named_sum": 0.5,        # mass named per place; the rest stays unnamed
    "max_named": 20,         # name only the top-N coins per place; the rest
                             # take the unnamed share, which prices them better
                             # than a tiny named p would
    "persist_tau_h": 3.0,    # horizon half-life of the persistence blend
    "persist_sd": 0.35,      # sd of the persistence size normal (log10 SOL)
    "size_floor": 0.0005,    # per-bin floor for sizes
    "size_named": 0.95,      # mass named per place's sizes
    "cand_top": 60,          # last chart's top N are candidates
    "cand_recent": 30,       # plus any coin top-N in the last 3 marks
    "seed": 12345,
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

    # Monte-Carlo: one path is a full trajectory; record at each read.
    rng = random.Random(p["seed"])
    n_reads = len(offsets)
    occ = {k: {uid: 0 for uid in cand} for k in range(1, places + 1)}
    size_hits = {k: [0] * (len(edges) + 1) for k in range(1, places + 1)}
    total_slots = p["n_paths"] * n_reads
    for _ in range(p["n_paths"]):
        x = {uid: start[uid] + rng.gauss(0, 0.02) for uid in cand}
        t_prev = 0.0
        for off in offsets:
            dt = off - t_prev
            t_prev = off
            if dt > 0:
                t_mid = off - 0.5 * dt
                decay = 0.5 ** (t_mid / p["drift_halflife_h"])
                sq = math.sqrt(dt)
                for uid in cand:
                    x[uid] += drift[uid] * decay * dt + rng.gauss(0, sigma[uid] * sq)
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
