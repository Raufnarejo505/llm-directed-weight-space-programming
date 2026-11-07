# Contributing

Thanks for looking. This repository is a *record of an experiment* first and a toolkit
second, so the most useful contributions are reproducibility fixes and extensions of the
analysis, not refactors of the results.

## Ground rules

1. `checkpoints/original/` is immutable. Never patch it in place; `tools/patch.py` writes
   to `checkpoints/patched/<patch_id>/` by design.
2. No gradients. Any new intervention must be closed-form (means, covariances, projections)
   or explicitly labelled as a different class of experiment.
3. Claims must match evidence. The corpus is separable by a stack fingerprint, so
   classification gains here are not evidence of detection capability.

## Running the tests

    pip install numpy safetensors pytest
    pytest -q

The suite deliberately avoids torch and the checkpoint: the patch engine is tested against
synthetic safetensors, and the result tables are checked for internal consistency.

## Adding a patch

1. add the JSON to `patches/`, derived only from the *support* split;
2. run `python tools/eval_patch.py <patch_id>` (writes a manifest with delta L2);
3. add the row to `results/tables/` via the aggregation scripts;
4. report macro F1, FPR, AUC, between-class distance, separation ratio and drift —
   including the case where nothing changes.

## Style

`ruff` with line length 110 (see `pyproject.toml`). Type hints appreciated, not required.
