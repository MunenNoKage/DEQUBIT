# Experiment log

One entry per experiment, newest first. Numbers come from
`results/metrics_validation.json` and `results/metrics_test.json`.

| Date | Experiment | Hypothesis | Median L1 relative change | Conclusion |
|---|---|---|---|---|
| 2026-10-21 | Ablations on the validation circuits, 3 seeds: MLP without local counts, without context, radius 1 for both models | The local counts carry the result; pair flips and the context add a little | MLP: no local counts 0.000, no context -0.692, radius 1 -0.662; shared scorer radius 1 -0.669 | Without local counts nothing is learned. The context adds nothing at one noise level; pair flips add 0.03 to 0.05 |
| 2026-10-21 | Baselines, MLP, shared scorer and per-qubit flip model on the validation circuits, 5 seeds, sampled corrections | Per-bitstring corrections from local counts beat corrections of the whole distribution | Shared scorer -0.719, MLP -0.694, MIX -0.693, linear map -0.693, per-qubit flip 0.001 | The shared scorer is best on every metric; the MLP matches MIX and the linear map. Without inputs (per-qubit flip) nothing is learned |
