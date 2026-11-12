# pump.fun top-5 forecaster

Entry for make.fun's **pumpfun-top5-forecast-01**: predict the five largest
new pump.fun coins by market cap, 24 hours out, with honest probabilities.

## Method

Monte-Carlo cap diffusion. Each coin's log10 market cap (in SOL) is a
latent path: it starts at the last observed log-cap and diffuses with a
per-coin sigma (residual std, shrunk toward the global median sigma,
floored at 0.40 log10 SOL/hour — vol from six 10-minute marks badly
understates true vol). No drift is used: backtests showed 10-minute
log-cap trends are pure noise, so trend-following only hurts. In every simulated path the coins are re-ranked by simulated
cap at each of the task's seven answer reads; the fraction of (path,
read) pairs in which a coin holds place *k* is its forecast probability
for that place, and the simulated caps at each place give that place's
size-bin probabilities. The same simulation prices both the order and
the sizes.

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
