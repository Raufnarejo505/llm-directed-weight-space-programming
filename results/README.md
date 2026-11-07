# Results

Everything here was produced by `tools/eval_patch.py`, `tools/compare.py`,
`tools/geometry_bootstrap.py`, `tools/mechanism.py` and `tools/run_regressions.py`.

| path | contents |
|---|---|
| `original.json` | baseline condition: metrics, geometry, bootstrap CIs |
| `p1a-k*.json`, `p2-*.json`, `p3-*.json` | one record per patched condition |
| `geometry_ci.json` | 1,000-sample bootstrap CIs for between-class distance and separation ratio |
| `regression_*.json` | MLM / direction / metadata / swapped-burst heads on a fixed batch |
| `summary.json` | merged view of all conditions |
| `tables/metrics.csv` | macro F1, CIs, precision, recall, FPR, AUC, confusion matrix |
| `tables/geometry.csv` | between/within distances, separation ratio, drift, cosine |
| `tables/regression.csv` | pretraining-head losses per condition |
| `tables/edits.csv` | tensor, shape, operation, parameters changed, relative L2, checkpoint hash |
| `predictions/*.csv` | per-flow label, prediction, decision score and embedding drift |

Embedding caches (`*.npz`) are not committed; regenerate with `tools/eval_patch.py`.

## Reading the numbers

* All seven conditions predict the **same** labels: 2,001 true positives, 14 false
  positives, 367 true negatives, 0 false negatives.
* Geometry is where the intervention shows up: between-class distance 17.28 -> 34.54 and
  separation ratio 3.81 -> 6.76 across the kappa ladder, with drift 6% -> 59%.
* `p3-control-pooling` is the harness check: drift 1.7e-8, geometry identical to four
  decimals, pretraining losses bit-identical.
