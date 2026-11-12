# pump.fun top-5 forecaster

Entry for make.fun's **pumpfun-top5-forecast-01**: predict the five largest
new pump.fun coins by market cap, 24 hours out, with honest probabilities.

## Method

Monte-Carlo cap diffusion with mean reversion. Each coin's log10 market
cap (in SOL) is a latent path: it starts at the last observed log-cap
and follows an Ornstein-Uhlenbeck process — it diffuses with a per-coin
sigma (residual std, shrunk toward the global median sigma, floored at
0.35 log10 SOL/hour) and reverts toward the chart's median log-cap at
speed 1.2/hour. A small trend-following drift (0.2) is kept: the focused
search found it helps slightly at short horizons. The reversion ramps
up over the first 6 hours (no reversion is visible at 10-minute scale;
full reversion is measured at ~10h) and is integrated in 15-minute
substeps. In every simulated path the coins are re-ranked by simulated
cap at each of the task's seven answer reads; the fraction of (path,
read) pairs in which a coin holds place *k* is its forecast probability
for that place, and the simulated caps at each place give that place's
size-bin probabilities. The same simulation prices both the order and
the sizes.

Why mean reversion: measured on the kit cohort over ~10 hours, the
median coin's log-cap barely moved (+0.005 dex) while the top-5 crashed
from log-cap 3.28–3.56 to 1.64–2.12 and the new top-5 came from ranks
14–31. Leaders revert to the pack; the pack stays put. A version without
reversion scored −54.3 on a ~10h held-out read; with reversion, −26.5.

The Monte-Carlo forecast is blended with the house persistence
baseline, with the blend weight decaying in the forecast horizon
(`w = exp(-mean_offset_hours / persist_tau_h)`, tau 1.5h): over minutes
persistence is nearly optimal, over many hours the simulation takes
over. Only part of each place's mass is named; the rest stays unnamed,
which is what prices the coins born after the cut.

The persistence size distribution uses a horizon-adaptive sd: 0.15 at
offsets up to 0.25h, ramping to 0.70 at offsets from 0.30h. Measured
best sd is 0.20 at 10–20 min horizons but 0.35+ at 30+ min (the place
caps move far at 30+ min: median |Δlogcap| 0.527). Only the top 12
coins per place are named; the MC component names 0.8 of each place's
mass, the persistence component 0.5.

Calibration: two focused random-search rounds (602 configs) on 3
short-horizon splits plus two long-horizon pseudo-reads (~10.4h /
~11.4h after the kit cut, built from live DexScreener caps of all 454
kit mints), multi-seed validated (spread 0.04 short / 0.38 long over
seeds 12345/999/2026) and confirmed on two held-out snapshots taken
after tuning (~12.7h / ~13.1h after the cut): short −34.82, long −26.46
(vs house baseline −37.77 short / −72.14 long).

Standard library only, no network, deterministic (fixed seed, ties
broken by mint address): two runs on the same history are byte-identical.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `/input/history.ndjson` + `/input/task.json` (or the env dirs),
writes `/output/forecast.json`.

## License

MIT. See `LICENSE`.
