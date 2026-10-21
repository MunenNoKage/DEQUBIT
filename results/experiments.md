# Experiment log

One entry per experiment, newest first. Numbers come from
`results/metrics_validation.json` and `results/metrics_test.json`.

| Date | Experiment | Hypothesis | Median L1 relative change | Conclusion |
|---|---|---|---|---|
| 2026-10-21 | Baselines, MLP, shared scorer and per-qubit flip model on the validation circuits, 5 seeds, sampled corrections | Per-bitstring corrections from local counts beat corrections of the whole distribution | Shared scorer -0.719, MLP -0.694, MIX -0.693, linear map -0.693, per-qubit flip 0.001 | The shared scorer is best on every metric; the MLP matches MIX and the linear map. Without inputs (per-qubit flip) nothing is learned |
