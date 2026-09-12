# Notebook conventions

Naming: `NN-kebab-case-name.ipynb`, numbered in the order you would read them.

1. **A notebook is a narrative, not the source of truth.** Code used more than once
   belongs in `src/dequbit/`, and the notebook imports it.
2. **Every notebook opens with Goal / Input / Conclusion.** The Conclusion is written
   **after** the analysis, once you know what the results say.
3. **`Restart & Run All` before committing.** A notebook that does not run cleanly top
   to bottom does not get committed.
4. **Outputs are committed**, so plots and tables are visible on GitHub without
   re-running anything. `.pre-commit-config.yaml` ships an `nbstripout` hook, disabled
   by default to match this convention; enable it if merge conflicts outweigh the
   benefit.
5. **Notebooks and git merges do not mix.** One notebook, one author, at a time.
6. **Colab.** Each notebook carries an "Open in Colab" badge and a guarded first cell
   that clones the repo and installs dependencies; it is a no-op locally. Replace
   `<org-or-user>/<repo>` with your repository once it exists.

The four notebooks here are empty placeholders.
