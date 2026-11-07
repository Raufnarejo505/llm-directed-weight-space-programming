# Reproduction guide

Everything below is CPU-only and was executed on a 4-core Windows machine. Wall-clock
numbers are from that machine.

## 1. Environment

Python 3.10-3.12 with:

    pip install torch transformers==4.57.3 datasets numpy safetensors scapy pandas \
                tensorboard psutil matplotlib

`transformers` must be 4.57.x: the model code imports `ModernBertMLP` and
`RoFormerSinusoidalPositionalEmbedding` from specific module paths.

Note on Python 3.15: at the time of the experiment the host's system Python was 3.15, for
which `pyyaml` (a `datasets` dependency) had no wheel and no C++ compiler was available.
A self-contained CPython 3.12 was installed and used instead. Any 3.10-3.12 works.

## 2. Model code and checkpoint

    git clone --depth 1 https://github.com/SNL-UCSB/netFound vendor/netFound
    huggingface-cli download snlucsb/netFound-small --local-dir checkpoints/original

Verify the checkpoint:

    sha256 9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70
    size   213,551,820 bytes

`tools/nfcommon.py` puts `vendor/netFound/src` on `sys.path`; set `NETFOUND_SRC` to override.

## 3. Dataset

Download CTU-IDSEVAL-6 (Zenodo 21027042), unpack `pcap.zip` and `zeek.zip` so that:

    data/ctu/pcap/pcap/ctu-idseval-6-*.pcap
    data/ctu/zeek/zeek/*.conn-labeled.log

Then convert the four captures used here:

    python tools/build_dataset.py "data/ctu/pcap/pcap/*.pcap" data/flows_tcp.jsonl

Expected: 761 benign and 133,347 scan TCP flows (~5 min on 4 cores; the 9.7 MB scan-3
capture produces 131,148 of them).

## 4. Baseline

    python tools/embed_all.py          # ~8 min, writes data/emb_orig.npz and prints metrics
    python tools/make_original_result.py

Expected original metrics: macro F1 0.9889, precision 0.9931, recall 1.0000, FPR 0.0367,
AUC 0.99998, TP 2001 / FP 14 / TN 367 / FN 0.

## 5. Patch construction (optional - the patches are committed)

    python tools/steering.py           # support-set activation means -> data/steering.npz (~3 min)
    python tools/make_patches.py       # writes patches/*.json and dry-run validates them

`make_patches.py` prints the unit selection, the kappa -> alpha mapping and the dry-run
report for each patch.

## 6. Apply and evaluate each patch

    python tools/eval_patch.py p1a-k1-rank1-wow
    python tools/eval_patch.py p1a-k2-rank1-wow
    python tools/eval_patch.py p1a-k5-rank1-wow
    python tools/eval_patch.py p1a-k10-rank1-wow
    python tools/eval_patch.py p2-single-unit
    python tools/eval_patch.py p3-control-pooling

Each run applies the patch to `checkpoints/patched/<patch_id>/`, re-embeds the fixed
support and test sets, and writes `results/<patch_id>.json` (metrics, geometry, bootstrap).
~11 min per patch on 4 cores.

## 7. Aggregation and analysis

    python tools/compare.py             # metrics, paired bootstrap, decision flips
    python tools/geometry_bootstrap.py  # bootstrap CIs for between-class distance / separation ratio
    python tools/mechanism.py p1a-k1-rank1-wow p1a-k5-rank1-wow p1a-k10-rank1-wow p2-single-unit

## 8. Regression suite

    python tools/run_regressions.py orig p1a-k1-rank1-wow p1a-k2-rank1-wow p1a-k5-rank1-wow \
        p1a-k10-rank1-wow p2-single-unit p3-control-pooling

Fixed deterministic masking (seed 7) over a fixed 120-flow batch, 959 masked tokens,
446 bursts. ~1-2 min per checkpoint.

## 9. Invariants to check

1. `checkpoints/original/model.safetensors` must still hash to 9e0080bb... after every run;
   `tools/patch.py` never writes to it.
2. `p3-control-pooling` must produce drift < 1e-6, identical metrics and zero flips.
3. All six patches must produce byte-identical predictions to the original
   (`results/predictions/*.csv`).

## 10. Expected numbers

| quantity | value |
|---|---|
| between-class distance, original | 17.28 |
| between-class distance, kappa=10 | 34.54 |
| slope of between-class distance in kappa | 1.727 (R^2 = 0.9999) |
| drift, kappa=10 | 59.0% |
| decision flips, all patches | 0 |
| paired bootstrap d(macro F1) | 0.0000 [0.0000, 0.0000] |
