# Historical experiments

This directory preserves earlier evaluation scripts, diagnostics, visualization scripts, and datasets while keeping the repository root focused on the supported training, evaluation, and demonstration entry points.

The archived files retain their original contents and relative paths beneath this directory. Former root-level experiments appear directly here; earlier simulator and plotting scripts appear in `src/`, and older lap data remains in `results/`. The original files are preserved for reference and provenance.

These scripts have **not been adapted to run from this archive location**. They may assume the original working directory, import other historical scripts, use earlier environment observations, or refer to paths outside this directory. Use the current root-level commands in the [project README](../../README.md) for supported workflows.

The complete original layout is available at published revision [`ef3f0718cc3adfc572a25a6aa839e9ce3281819c`](https://github.com/preetagam205-tech/rl-racing-line-optimization/tree/ef3f0718cc3adfc572a25a6aa839e9ce3281819c). To inspect that revision separately from the current checkout:

```bash
# Run from the current repository root.
git worktree add --detach ../rl-racing-line-original ef3f0718cc3adfc572a25a6aa839e9ce3281819c
```

Restoring the original layout does not remove the historical checkpoint-compatibility and training-provenance limitations documented in [CHECKPOINTS.md](../CHECKPOINTS.md) and [RESULTS.md](../RESULTS.md).

Active PPO learning-snapshot CSVs are in [`docs/benchmarks/`](../benchmarks/), and the original Pure Pursuit comparison is in [`docs/benchmarks/baseline_comparison/`](../benchmarks/baseline_comparison/). Current tuned-baseline, shared-physics evolution, and trajectory-analysis data remain in their named directories at the repository root. Interactive summary JSONs are in `assets/`.
