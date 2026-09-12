# models/

`models/*` is gitignored. Trained weights do not belong in git.

| Threshold | What happens |
|---|---|
| > 1 MB | `.pre-commit-config.yaml`'s `check-added-large-files` hook fails at 1000 KB |
| > 100 MB | GitHub rejects the push outright |
| Any size, after the fact | `git rm` does **not** remove it from history. The object stays in every clone unless someone rewrites history (`git filter-repo`, BFG), which is painful to coordinate. Decide *before* the first commit. |

## Where to put them instead

| Option | Good for |
|---|---|
| Hugging Face Hub | Models and datasets, free hosting, versioning |
| GitHub Releases | A one-off artifact up to ~2 GB, no extra tooling |
| Git LFS | Keeping the file feeling like part of the repo |
| DVC + cloud storage | Many large artifacts over a project's lifetime |
| Zenodo | You need a DOI for a specific version |

## Model card checklist

Whichever you use, the published card should cover: the task, the architecture, the
training data (link to `data/README.md`, do not restate its terms), the metrics (pulled
from `results/metrics.json`, not hand-typed), the limitations, the license, and how to
cite it.
