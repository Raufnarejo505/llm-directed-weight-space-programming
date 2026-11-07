# Task E - Comparative report: closed-form weight-space interventions

Six patches, each applied to its own checkpoint copy. Original checkpoint never modified.
No gradients, no backpropagation, no optimiser anywhere in the pipeline.

## 0. Headline

1. A closed-form rank-1 edit of one tensor (393k values, 0.74% of the model) produces a
   **linear, monotone geometry dose-response**: between-class distance 17.28 -> 34.54
   (+100%), separation ratio 3.81 -> 6.76 (+77%) at kappa=10, from a 0.37% relative
   perturbation of that tensor.
2. **Zero decision flips at every strength.** Macro F1, precision, recall, FPR and the
   confusion matrix are identical for all six patches and the original. The intervention
   moves the representation but not the nearest-centroid decision.
3. The mechanism is measurable: displacement is proportional to the discriminative
   projection, and the 14 ambiguous benign flows move ~48% as far as scans, so their
   margin is preserved.
4. Negative control is exact (drift 1.7e-8, zero flips, identical pretraining losses).
5. The 512-value single-unit edit produces a contraction of comparable magnitude
   (-11% between-class distance) with the same zero-flip outcome: control is bidirectional.

## 1. Provenance

Base checkpoint: snlucsb/netFound-small, sha256 9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70.

| patch | tensor | shape | edit | values changed | % of 53,383,495 | weight rel. L2 | checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| p1a-k1-rank1-wow | encoder.layer.3.burst_encoder.mlp.Wo.weight | 512x768 | dW = 0.1098 u v^T | 393,149 | 0.7364 | 3.7e-4 | e52c138d507e... |
| p1a-k2-rank1-wow | same | 512x768 | dW = 0.2196 u v^T | 393,186 | 0.7365 | 7.5e-4 | aaa12b7cc906... |
| p1a-k5-rank1-wow | same | 512x768 | dW = 0.5491 u v^T | 393,205 | 0.7366 | 1.9e-3 | 78b3ab5fc4c4... |
| p1a-k10-rank1-wow | same | 512x768 | dW = 1.0982 u v^T | 393,212 | 0.7366 | 3.7e-3 | 7cb65c69a368... |
| p2-single-unit | same | 512x768 | column 656 += 0.565 u | 512 | 0.0010 | 1.9e-3 | 42c2b699daba... |
| p3-control-pooling | attentivePooling.lin_proj.weight | 512x512 | x1.05 | 262,144 | 0.4911 | 2.1e-3 | bb3666f104f9... |

u (512-d) and v (768-d) are unit vectors from support-set activation means; the rank-1
patch has 1,279 intrinsic degrees of freedom despite rewriting 393k values.

## 2. Method (identical for every condition)

* flow embedding = attention-mask-weighted mean of final hidden states (512-d);
* centroids from 1,999 scan / 380 benign support flows, recomputed *under each patch*;
* nearest centroid (Euclidean) on 2,001 scan / 381 benign held-out flows;
* 2,000-sample stratified bootstrap for metric CIs, 1,000-sample for geometry CIs;
* paired bootstrap of macro-F1 difference against the original on the same resamples.

## 3. Downstream classification

| condition | macro F1 | 95% CI | precision | recall | FPR | AUC | TP | FP | TN | FN | flips vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 0.99998 | 2001 | 14 | 367 | 0 | - |
| p1a-k1 | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |
| p1a-k2 | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |
| p1a-k5 | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |
| p1a-k10 | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |
| p2-single-unit | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |
| p3-control | 0.9889 | [0.9824, 0.9945] | 0.9931 | 1.0000 | 0.0367 | 1.00000 | 2001 | 14 | 367 | 0 | 0 |

Paired bootstrap difference vs original: 0.0000, 95% CI [0.0000, 0.0000], for every patch.
The decision rule is invariant to all six interventions on this corpus.

## 4. Geometry and drift

| condition | between (support) | within (pooled) | separation ratio | between (test) | 95% CI | sep (test) | 95% CI | drift | cosine | centroid shift |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 17.28 | 4.54 | 3.81 | 16.82 | [16.57, 17.04] | 3.43 | [3.24, 3.62] | 0 | 1.00000 | 0 |
| p1a-k1 | 19.02 | 4.58 | 4.15 | 18.51 | [18.22, 18.76] | 3.73 | [3.53, 3.95] | 5.96% | 0.99924 | 1.60 |
| p1a-k2 | 20.76 | 4.63 | 4.48 | 20.19 | [19.89, 20.45] | 4.02 | [3.79, 4.27] | 11.91% | 0.99723 | 3.19 |
| p1a-k5 | 25.95 | 4.80 | 5.41 | 25.24 | [24.92, 25.56] | 4.83 | [4.54, 5.15] | 29.67% | 0.98685 | 7.94 |
| p1a-k10 | 34.54 | 5.11 | 6.76 | 33.59 | [33.15, 33.95] | 5.99 | [5.58, 6.42] | 59.02% | 0.96488 | 15.80 |
| p2-single-unit | 15.40 | 4.50 | 3.42 | 15.09 | [14.87, 15.31] | 3.11 | [2.95, 3.29] | 6.23% | 0.99897 | 1.67 |
| p3-control | 17.28 | 4.54 | 3.81 | 16.82 | [16.58, 17.07] | 3.43 | [3.24, 3.62] | 1.7e-8 | 1.00000 | 1.0e-6 |

All bootstrap intervals are disjoint along the kappa ladder, so the geometric changes are
significant, not sampling noise.

## 5. Edit-strength response curve (rank-1)

| kappa | alpha | between (support) | delta | separation ratio | drift | decision flips |
|---|---|---|---|---|---|---|
| 0 | 0 | 17.28 | - | 3.81 | 0% | 0 |
| 1 | 0.1098 | 19.02 | +1.74 | 4.15 | 5.96% | 0 |
| 2 | 0.2196 | 20.76 | +3.48 | 4.48 | 11.91% | 0 |
| 5 | 0.5491 | 25.95 | +8.67 | 5.41 | 29.67% | 0 |
| 10 | 1.0982 | 34.54 | +17.26 | 6.76 | 59.02% | 0 |

Both the geometry gain and the drift are linear in kappa to within 1%:
between = 17.28 + 1.727 kappa (R^2 = 0.9999) and drift = 5.90 kappa %. The separation
ratio grows sub-linearly because within-class dispersion also rises (4.54 -> 5.11).

## 6. Single-unit vs rank-1

The single-unit edit rewrites 512 values (0.001% of the model) and moves 6.23% of the
embedding mass, comparable to kappa=1 (5.96%) with 768x fewer changed values.
Because the selected unit has a negative mean activation, the same construction pushes
in the -u direction and **contracts** the classes: between 15.40 instead of 19.02,
separation ratio 3.42 instead of 4.15. Decisions still do not change.

So the geometry is controllable in both directions, and the sign of the effect is
predictable from the sign of the edited unit's class-conditional mean activation.

## 7. Negative control (p3)

Scaling attentivePooling.lin_proj.weight by 1.05 - a tensor that is not on the
masked-mean embedding path - leaves the readout untouched: drift 1.7e-8 (float rounding),
cosine 1.00000, identical geometry to four decimals, 0 decision flips, and regression
losses bit-identical to the original. The harness does not fabricate effects.

## 8. Mechanism

Per-group displacement along the readout, kappa = 1 / 5 / 10:

| group | n | displacement | margin change (d_scan - d_benign) |
|---|---|---|---|
| scan true positives | 2,001 | 1.87 / 9.30 / 18.50 | -1.67 / -8.35 / -16.65 |
| benign true negatives | 367 | 0.14 / 0.70 / 1.40 | +1.64 / +8.23 / +16.46 |
| benign false positives | 14 | 0.88 / 4.41 / 8.82 | -0.02 / +0.15 / +0.70 |

The 14 false positives are exactly the flows whose MLP activations project like scans:
they move 47-48% as far as scans at every strength, tracking the scan centroid. Their
decision margin stays negative (-1.98 -> -1.28 at kappa=10), so they never cross the
boundary. The intervention amplifies the very feature that made them ambiguous.

Extrapolating the measured margin trajectory, the first flips would require
kappa ~ 20-25, i.e. roughly twice the strongest tested edit, with embedding drift
above 120% of the original norm.

## 9. Regression on pretraining capabilities

Fixed deterministic batch: 120 flows, 959 masked tokens, 446 bursts.

| condition | MLM loss | MLM top-1 | direction acc | metadata MAE | swapped-burst loss |
|---|---|---|---|---|---|
| original | 2.768039 | 0.68717 | 0.84081 | 1,352,269.652 | 4.3581 |
| p1a-k1 | 2.768123 | 0.68717 | 0.84081 | 1,352,269.653 | 4.4534 (+2.2%) |
| p1a-k2 | 2.768198 | 0.68717 | 0.84305 | 1,352,269.653 | 4.5495 (+4.4%) |
| p1a-k5 | 2.768365 | 0.68717 | 0.84529 | 1,352,269.653 | 4.8375 (+11.0%) |
| p1a-k10 | 2.768422 | 0.68613 | 0.84081 | 1,352,269.648 | 5.2994 (+21.6%) |
| p2-single-unit | 2.769590 | 0.68822 | 0.83857 | 1,352,269.650 | 4.2280 (-3.0%) |
| p3-control | 2.768039 | 0.68717 | 0.84081 | 1,352,269.652 | 4.3581 (0.0%) |

MLM top-1 is unchanged to 5 significant digits up to kappa=5 and drops 0.10 pp at
kappa=10. Direction accuracy is unchanged or slightly better. Metadata regression is
insensitive (the head reads burst representatives, which are dominated by the flow
encoder). The only clear regression is the swapped-burst classifier, which loses
2-22% of its margin as the residual stream grows.

## 10. Interpretation (four levels, kept separate)

* **Weight-space intervention**: real and recorded - six JSON patches, per-tensor delta
  manifests, checkpoint hashes. Not a fine-tune: 1,279 intrinsic degrees of freedom,
  derived from closed-form activation means.
* **Representation-geometry modification**: demonstrated, large, monotone and
  statistically significant (between-class distance 2x, separation ratio +77%).
* **Downstream classification performance**: unchanged - zero flips, identical macro F1
  and FPR on this corpus.
* **Genuine capability acquisition**: not claimed and not observed. The corpus is
  separable by IP TTL / TCP window / IP length alone (single-feature AUC 1.000), so
  nothing here measures port-scan detection.

The result is therefore a controlled geometry intervention, not a capability edit. Its
interest is the dissociation: geometry moved by 100% without moving a single decision,
because the edit is proportional to the discriminative projection and therefore scales
the class means and the ambiguous members together.

## 11. Retain / reject decisions

| patch | retain? | reason |
|---|---|---|
| p1a-k1-rank1-wow | retain as control instrument | cleanest dose point; regression within tolerance; no classification effect |
| p1a-k2-rank1-wow | retain as control instrument | linear dose point; regression within tolerance |
| p1a-k5-rank1-wow | retain with caveat | large geometric effect, swapped-burst loss +11% |
| p1a-k10-rank1-wow | reject | MLM top-1 -0.10 pp and swapped-burst loss +21.6% for no decision benefit |
| p2-single-unit | retain as minimal-footprint datapoint | 512 values, contraction direction, small regressions |
| p3-control-pooling | retain as negative control | exact zero effect, validates harness |

None of the six is retained as a classification improvement: all produce identical
predictions. Any future work that claims a decision-level effect from this family of
edits would need a readout that is not invariant to class-homogeneous translation
(e.g. a margin-based or temperature-sensitive rule).

## 12. Reproduction

    .py312/python.exe tools/eval_patch.py <patch_id>     # apply + embed + metrics + CIs
    .py312/python.exe tools/run_regressions.py <ids>     # pretraining-head suite
    .py312/python.exe tools/compare.py                   # aggregate + paired bootstrap
    .py312/python.exe tools/geometry_bootstrap.py        # geometry CIs
    .py312/python.exe tools/mechanism.py <ids>           # per-group displacement

Raw records: results/*.json, results/emb_*.npz; patches: patches/*.json;
patched checkpoints: checkpoints/patched/<patch_id>/.
