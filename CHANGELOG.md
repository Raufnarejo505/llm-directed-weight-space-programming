# Changelog

## 1.0.0 — 2026-11-07

First public record of the experiment.

### Added

* inspection of `snlucsb/netFound-small`: 132 tensors, 53,383,495 parameters, weight
  statistics, embedding spectrum, parameter budget;
* a Python port of the upstream netFound flow tokenizer (12 header tokens per packet,
  10 ms burst splitting, relative seq/ack, payload stripped);
* dataset validity analysis of CTU-IDSEVAL-6 showing single-feature class separation
  (IP TTL, TCP window, IP length: AUC 1.000 each);
* baseline nearest-centroid evaluation: macro F1 0.9889, FPR 0.0367, AUC 0.99998;
* six reversible JSON patches and a validated patch engine with per-tensor manifests;
* evaluation of all six patches: identical decisions, monotone geometry dose-response
  (between-class distance 17.28 -> 34.54, linear in kappa, R^2 = 0.9999);
* mechanism analysis: the 14 ambiguous flows move ~48% as far as scans, preserving margins;
* pretraining-head regression suite and negative control (drift 1.7e-8).

### Known limitations

* the corpus cannot support port-scan-detection claims (fingerprint shortcut);
* CPU-only timings (~11 min per patch evaluation on 4 cores);
* embedding caches are regenerated rather than committed.
