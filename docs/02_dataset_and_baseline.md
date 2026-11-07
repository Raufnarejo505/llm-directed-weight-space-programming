# Phase 2 - Port-scan detection: dataset, baseline, analysis, proposed patches

Status: **Tasks A-D complete. No weight has been modified.** Task E awaits approval
(see the decision section at the end).

---

## Task A - Dataset verification

**Source.** CTU-IDSEVAL-6 (Zenodo 21027042), 'A Labeled Network Dataset for the Evaluation
of Intrusion Detection Systems': raw pcaps + Zeek conn logs + per-capture label configs.

| capture | packets | TCP flows (ours) | label |
|---|---|---|---|
| ctu-idseval-6-benign-user-traffic-1 | 75,356 | 761 | Benign / Background |
| ctu-idseval-6-malicious-portscan-1 | 2,009 | 1,999 | Malicious (src 147.32.80.37) |
| ctu-idseval-6-malicious-portscan-2 | 206 | 200 | Malicious |
| ctu-idseval-6-malicious-portscan-3 | 131,237 | 131,148 | Malicious |

**Token compatibility.** The released checkpoint expects strip_payload=True and
max_burst_length = 73 = 6x12+1. Our converter is a direct port of
pre_process_src/Tokenize.py + 3_field_extraction.cpp:

* bidirectional conversation per 5-tuple; direction +1 for the initiator (first packet source);
* bursts split on a >10 ms inter-packet gap, per direction, re-merged by start time;
* max 12 bursts, max 6 packets per burst, 12 header tokens per TCP packet;
* seq/ack made relative to the flow start, as the C++ extractor does.

Header token layout confirmed against the checkpoint's own training distribution
(the Zenodo netFound sampler streams the same 12-value shape, e.g.
[20, 0, 40, 0, 239, 2, 1024, 0, 0, 0, 0, 0] = IP_hl, tos, IP_len, IP_flags, TTL,
TCP_flags, window, seq_hi, seq_lo, ack_hi, ack_lo, urg):

    [IP_hl, IP_tos, IP_len, IP_flags, IP_ttl, TCP_flags, TCP_win, seq(2), ack(2), urg]

Layout, vocabulary and metadata semantics match; the checkpoint encodes these flows
without error.

**Split** (chronological inside every capture, so no near-duplicate leakage):

| | scan | benign |
|---|---|---|
| support (centroid construction) | 1,999 | 380 |
| held-out test | 2,001 | 381 |

---

## Validity finding: the corpus is separated by a stack fingerprint, not by scanning

This is the most important Phase 2 result and it conditions everything else.

| single feature | AUC | scan | benign | ranges overlap? |
|---|---|---|---|---|
| IP TTL | **1.000** | 48.0 (37-59) | 128.0 (128) | **none** |
| TCP window | **1.000** | 1024 (1024) | 8192 (8192) | **none** |
| IP total length | **1.000** | 44 (40-44) | 49.3 (48-52) | **none** |
| max IAT | **1.000** | 0 ms | 12,748 ms | no |
| packets per flow | 0.973 | 1 | 14.8 | yes |

A one-line rule (ttl < 64) scores **macro F1 = 1.0000, FPR = 0.000** on the full corpus.
Of 761 benign flows only **26** are single-packet, and **0** of them share an identical
12-token signature with any of the 133,256 single-packet scan flows: the benign capture
(one host, 192.168.1.114, TTL 128 / window 8192 / DF set) and the scan captures
(147.32.80.37, TTL 37-59 / window 1024) never overlap on host or path fingerprint.

**Consequence.** The two classes differ on nearly every channel the model can see, for
reasons unrelated to scanning: different hosts, different capture vantage points, and a
scan generator with randomised TTL. Any measured improvement on this corpus is at best
amplification of that shortcut. It is not evidence of port-scan detection.

A second, structural limit: the tokenizer **discards source/destination ports and IP
addresses**. Within a single flow the model therefore cannot see the fields that define
a scan (one source hitting many ports/destinations). After matching TTL, window, IP length
and packet count, a one-packet scan and a one-packet benign SYN produce **identical model
inputs**. Flow-level scan evidence in this input space is limited to fingerprint, flow
shape and timing.

---

## Task B - Baseline (original checkpoint)

Readout: flow embedding = attention-mask-weighted mean of the final hidden states (512-d);
class centroids from the support split; nearest centroid (Euclidean), no training.

| metric | value |
|---|---|
| macro F1 | **0.9889** |
| precision | 0.9931 |
| recall | 1.0000 |
| false-positive rate | 0.0367 (14 / 381) |
| false-negative rate | 0.0000 |
| accuracy | 0.9941 |
| ROC AUC | 1.0000 |

Confusion matrix (test): TP 2001, FP 14, TN 367, FN 0.
Per capture: benign 96.33% correct; portscan-1/2/3 100% each.

**Ceiling.** macro F1 cannot exceed 1.0, so the maximum attainable improvement is
**+1.11 pp**, not +5 pp. The stated acceptance threshold is unreachable on this corpus
by construction, independently of any weight edit.

---

## Task C - Where the separability lives

Per-layer pooled representations (400 scan / 160 benign subsample, nearest centroid):

| representation | macro F1 | FPR | AUC | mean-difference norm | Cohen d (projection) |
|---|---|---|---|---|---|
| embedding output | 0.9685 | 0.0875 | 1.000 | 9.87 | 6.16 |
| layer 0 | 0.9685 | 0.0875 | 1.000 | 39.83 | 6.70 |
| layer 1 | 0.9685 | 0.0875 | 1.000 | 41.86 | 7.11 |
| layer 2 | 0.9685 | 0.0875 | 1.000 | 75.78 | 7.33 |
| layer 3 (final) | 0.9685 | 0.0875 | 1.000 | 16.85 | 7.31 |

Separability is already maximal **before the first transformer layer** and is preserved,
not constructed, by the encoder. The model needs no scan-specific knowledge for this task.

Channel controls (final layer):

| manipulation | macro F1 | FPR | AUC |
|---|---|---|---|
| none | 0.9685 | 0.0875 | 1.0000 |
| TTL + window + IP-length permuted across flows | 0.9685 | 0.0875 | 0.9834 |
| 1-packet subset only | 0.9510 | 0.1400 | 1.0000 |
| 1-packet subset, metadata permuted | 0.9510 | 0.1400 | 1.0000 |

Even with the three obvious shortcuts scrambled, separation survives because packet count,
burst count and timing are also class-correlated in this corpus.

**Component assessment.**

| component | params | plausible influence on flow-level separability | evidence |
|---|---|---|---|
| token embedding 65600x512 | 33.6 M | carries the fingerprint tokens (TTL/win/len) | separation present at output |
| embedding fusion 1536->512 | 786 k | first mixing point; geometry of all tokens | layer-0 d = 39.8 |
| burst encoder (x4) | 8.9 M | per-packet features -> burst summary | d grows to 75.8 |
| flow encoder (x4) | 8.9 M | burst summary -> flow summary | final d shrinks to 16.9 |
| attentive pooling | 263 k | **not on the readout path** (masked mean used) | structural |
| last-layer MLP Wo | 393 k | last write to every token before pooling | chosen edit site |

The last burst-encoder MLP output projection is the last place where a single tensor
writes to *every* token that is later mean-pooled, so a rank-1 update there is the most
direct gradient-free lever on the readout.

---

## Task D - Proposed patches (built and dry-run validated, NOT applied)

Statistics are computed on the **support split only**, closed-form (means of activations,
no gradients, no optimiser):

* u = unit(mean final hidden | scan - mean final hidden | benign), 512-d, norm(dh) = 17.48
* v = unit(mean layer-3 MLP hidden | scan - mean layer-3 MLP hidden | benign), 768-d, norm(dphi) = 16.72
* projection of the MLP hidden on v: scan 17.47, benign 0.75

| patch | tensor | shape | operation | values changed | % of model | purpose |
|---|---|---|---|---|---|---|
| p1a-k1-rank1-wow | ...layer.3.burst_encoder.mlp.Wo.weight | [512,768] | dW = alpha*u v^T, alpha = 0.1098 | 393,216 | 0.737 | steer pooled embeddings |
| p1a-k2-... | same | [512,768] | alpha = 0.2196 | 393,216 | 0.737 | stronger steer |
| p1a-k5-... | same | [512,768] | alpha = 0.5491 | 393,216 | 0.737 | stronger steer |
| p1a-k10-... | same | [512,768] | alpha = 1.0982 | 393,216 | 0.737 | strongest steer |
| p2-single-unit | same | [512,768] | add one column (unit 656), scale 0.565 | **512** | 0.001 | minimal-footprint variant |
| p3-control-pooling | attentivePooling.lin_proj.weight | [512,512] | scale x1.05 | 262,144 | 0.491 | negative control: must have zero effect |

All six pass patch.dry_run (name, shape, dtype, finiteness, dimension checks). Each is a
standalone JSON in patches/; applying writes a new directory under
checkpoints/patched/<patch_id>/ and records per-tensor delta L2 and relative L2.

**Intrinsic dimensionality.** A rank-1 update of a 512x768 matrix has 1,279 degrees of
freedom even though it rewrites 393,216 values; the single-unit variant rewrites 512.

**Expected effect.** With a perfect ROC AUC but 14 false positives, the only headroom is
decision geometry: a boost along u should push benign flows that sit near the scan
centroid across the boundary. Best case macro F1 0.9889 -> 1.0000 (+1.11 pp); the 5 pp
criterion cannot be met.

**Risks.**
* The steer is derived from the same shortcut (TTL/window) the corpus rewards, so a gain
  would not generalise to scans that share the benign fingerprint.
* Over-steering (k5/k10) can collapse scan embeddings into a tight ball, raising FNR.
* The edit sits before the LM head, so MLM behaviour may shift; the regression suite
  (MLM top-1, direction accuracy, metadata MAE) must be re-measured.
* Editing Wo changes every token's residual stream, including the flow-encoder inputs.

---

## Pre-registered evaluation protocol (Task E, on approval)

1. Apply each patch to checkpoints/patched/<patch_id>/; base checkpoint untouched.
2. Re-embed exactly the same support and test flows; identical centroid rule.
3. Report macro F1, precision, recall, FPR, FNR, confusion matrix, AUC; per-capture recall.
4. Regression suite: MLM loss/top-1, direction-head accuracy, metadata L1 on a fixed
   deterministic masked batch; plus channel-control repeats.
5. Retain a patch only if macro F1 improves AND FPR does not rise by more than 2 pp
   AND MLM top-1 falls by less than 1 pp.

**Amended acceptance criterion (proposal).** Because the ceiling is 1.0, I propose
macro F1 >= 0.995 with FPR <= 0.0367 and no meaningful pretraining regression, instead of
+5 pp.

---

## Decision required before Task E

1. **Proceed as specified** on this corpus (expect <= +1.11 pp; interpret any gain as
   shortcut amplification, not scan detection).
2. **Change the target** to a capability representable in this input space
   (e.g. 'single-packet, unanswered SYN flow'), which the model can genuinely learn.
3. **Change the corpus** to one where benign and scan flows share host/stack fingerprints.
   Note: the released tokenizer drops ports and IPs, so no corpus can lift the single-flow
   information limit; a corpus can only remove the shortcut.
