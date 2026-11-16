# Experiment log

One entry per experiment, newest first. Numbers come from
`results/metrics_validation.json` and `results/metrics_test.json`.

| Date | Experiment | Hypothesis | Median L1 relative change | Conclusion |
|---|---|---|---|---|
| 2026-11-16 | Dataset v4 (FakeSherbrooke, 3 to 13 qubits, five circuit families), circuit context as model input, hyperparameter search, then MLP per size and one shared scorer for 3 to 11 qubits, 5 seeds, validation circuits, sampled corrections | The decoded circuit tells the model which bits are likely wrong; a shared scorer trained on all sizes transfers to larger circuits | Shared scorer at 3 / 5 / 8 / 11 / 12 / 13 qubits: -0.716 / -0.668 / -0.596 / -0.411 / -0.302 / -0.185; best baseline -0.735 / -0.660 / -0.507 / -0.318 / -0.251 / -0.124 | Circuit context adds 0.02 to 0.07. The shared scorer beats every baseline from 5 qubits on, by 0.09 to 0.10 at 8 to 11, and on unseen 12 and 13 qubit circuits; at 3 qubits MIX with the fitted physical rate is best |
| 2026-10-21 | Ablations on the validation circuits, 3 seeds: MLP without local counts, without context, radius 1 for both models | The local counts carry the result; pair flips and the context add a little | MLP: no local counts 0.000, no context -0.692, radius 1 -0.662; shared scorer radius 1 -0.669 | Without local counts nothing is learned. The context adds nothing at one noise level; pair flips add 0.03 to 0.05 |
| 2026-10-21 | Baselines, MLP, shared scorer and per-qubit flip model on the validation circuits, 5 seeds, sampled corrections | Per-bitstring corrections from local counts beat corrections of the whole distribution | Shared scorer -0.719, MLP -0.694, MIX -0.693, linear map -0.693, per-qubit flip 0.001 | The shared scorer is best on every metric; the MLP matches MIX and the linear map. Without inputs (per-qubit flip) nothing is learned |
