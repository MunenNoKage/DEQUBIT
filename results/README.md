# results/

Generated output: metrics, the experiment log, and figures. Committed, unlike `data/`
and `models/`, because these are small and they are what makes the numbers in the root
README checkable without anyone rerunning training.

Nothing here should be edited by hand except the written interpretation of the results,
which is the one part that cannot be computed.

| Path | Written by | Content |
|---|---|---|
| `metrics_validation.json`, `metrics_test.json` | `python -m dequbit evaluate` | Metrics of every model and baseline on the validation or test circuits |
| `figures/` | `python -m dequbit evaluate` | Figures for the README, slides and poster |
| `experiments.md` | by hand | One entry per experiment: hypothesis, setup, result, conclusion |
