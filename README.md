# pump.fun top-5 forecaster

Entry for make.fun's **pumpfun-top5-forecast-01**: predict the five largest
new pump.fun coins by market cap, 24 hours out, with honest probabilities.

## Method

Monte-Carlo cap diffusion with mean reversion. Each coin's log10 market
cap (in SOL) is a latent path: it starts at the last observed log-cap
and follows an Ornstein-Uhlenbeck process — it diffuses with a per-coin
sigma (residual std, shrunk toward the global median sigma, floored at
0.30 log10 SOL/hour) and reverts toward the chart's median log-cap at
speed 1.0/hour. No trend drift is used: backtests showed 10-minute
log-cap trends are pure noise, so trend-following only hurts. The
reversion ramps up over the first 3 hours (no reversion is visible at
10-minute scale; full reversion is measured at ~10h) and is integrated
in 15-minute substeps. In every simulated path the coins are re-ranked
by simulated cap at each of the task's seven answer reads; the fraction
of (path, read) pairs in which a coin holds place *k* is its forecast
probability for that place, and the simulated caps at each place give
that place's size-bin probabilities. The same simulation prices both the
order and the sizes.

Why mean reversion: measured on the kit cohort over ~10 hours, the
median coin's log-cap barely moved (+0.005 dex) while the top-5 crashed
from log-cap 3.28–3.56 to 1.64–2.12 and the new top-5 came from ranks
14–31. Leaders revert to the pack; the pack stays put. A version without
reversion scored −54.3 on a ~10h held-out read; with reversion, −26.7.

The Monte-Carlo forecast is blended with the house persistence
baseline, with the blend weight decaying in the forecast horizon
(`w = exp(-mean_offset_hours / persist_tau_h)`): over minutes
persistence is nearly optimal, over many hours the simulation takes
over. Only part of each place's mass is named; the rest stays unnamed,
which is what prices the coins born after the cut.

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
