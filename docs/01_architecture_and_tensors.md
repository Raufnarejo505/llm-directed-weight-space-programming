# Phase 1 — Inspection of snlucsb/netFound-small

Status: inspection complete. No weights have been modified.

## 1. Provenance and integrity

| field | value |
|---|---|
| repo | snlucsb/netFound-small (MIT) |
| file | model.safetensors |
| bytes | 213,551,820 |
| sha256 | 9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70 |
| HF LFS oid | 9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70  (match) |
| tensors | 132 |
| parameters | 53,383,495 (all F32) |
| trained with | transformers 4.57.3, 128 GPUs, 45k steps, ~9B tokens |

The local copy is bit-identical to the published checkpoint. It lives in
`checkpoints/original/` and is treated as read-only by every tool in this repo.

## 2. Runtime

The system Python is 3.15; `pyyaml` and `datasets` have no cp315 wheels and no C++
compiler is installed, so the stack runs in a self-contained CPython 3.12:
`D:\wsp-netfound\.py312\python.exe` with torch 2.15.0+cpu and transformers 4.57.3.

Verified by execution:
- `netFoundLanguageModelling.from_pretrained` loads all 132 tensors;
- `lm_head.decoder.weight` is tied to `word_embeddings.weight` (same storage);
- the four `position_embeddings` are one shared tensor (only layer 3 is serialised);
- a forward pass through `base_transformer` and through the full pretraining head runs.

## 3. Architecture (read from source, not inferred)

Model type `netFound` — a hierarchical *encoder* for network traffic, not a generative LM.
Input is a **flow** = up to 12 **bursts** = up to 6 packets x 12 header field-tokens (+1 CLS).

    token ids [B, L]  L = num_bursts * batch_max_burst_length   (max 12*73 = 876)
        |
        +-- word_embeddings 65600 x 512            (field-value id -> vector)
        +-- meta MLP 4 -> 1024 -> 512              (direction, bytes, pkt_count, iat)
        +-- protocol_embedding 18 x 512            (IP protocol, broadcast over the flow)
        |
        concat(512 + 512 + 512 = 1536) -> compressEmbeddings 1536->512 -> LayerNorm
        |
        +-- x4 [ netFoundLayer ]
        |      burst_encoder : RoFormerAttention(4 heads, RoPE) + ModernBERT GLU-MLP,
        |                      applied within each burst (length <= 73)
        |      position_embeddings 13x512 added to each burst's CLS/representative token
        |      flow_encoder  : same block applied across bursts (length <= 12)
        |
        +-- lm_head (dense 512->512, LN, decoder 512->65600 tied to word_embeddings)
        +-- attentivePooling (512->512, v:512->1) -> flow vector
        +-- swappedClassifierHiddenLayer 512->2, linearMetadataPred 512->3, dirPred 512->2

Pretraining heads present in the checkpoint: masked field-token modelling (MLM, p=0.3),
swapped-burst detection, burst metadata regression (iat/bytes/packet-count masked L1),
burst direction classification. Downstream classification in the repo uses attentive
pooling -> 2-layer MLP -> linear classifier.

## 4. Parameter budget

| component | params | % |
|---|---|---|
| token embedding (65600x512) | 33,587,200 | 62.92 |
| burst_encoder x4: attention | 4,206,592 | 7.88 |
| burst_encoder x4: MLP+norm | 4,720,640 | 8.84 |
| flow_encoder x4: attention | 4,206,592 | 7.88 |
| flow_encoder x4: MLP+norm | 4,720,640 | 8.84 |
| embedding fusion (1536->512) | 786,944 | 1.47 |
| metadata MLP | 529,920 | 0.99 |
| attentive pooling | 263,168 | 0.49 |
| LM head dense+norm | 263,680 | 0.49 |
| LM head decoder bias | 65,600 | 0.12 |
| protocol embedding | 9,216 | 0.02 |
| RoPE tables | 11,008 | 0.02 |
| flow position embeddings | 6,656 | 0.01 |
| pretraining heads (dir/meta/swap) | 3,591 | 0.01 |
| embedding LayerNorm | 1,024 | 0.00 |
| token_type embeddings — **UNUSED** | 1,024 | 0.00 |

`token_type_embeddings` is constructed by `RoFormerEmbeddings.__init__` but
`netFoundRoformerEmbeddings.forward` never reads it: 1,024 dead parameters.

## 5. Weight statistics (numpy, full checkpoint)

No NaN/Inf anywhere. No all-zero rows in the embedding matrix.

| group | std | notes |
|---|---|---|
| word_embeddings | 0.0337 | mean -0.0091 (global negative offset) |
| attention qkv | 0.039-0.093 | L0 burst qkv is the widest (0.093) |
| attention out dense | 0.021-0.030 | |
| MLP Wi / Wo | 0.028-0.035 | |
| attention LayerNorm (w+b pooled) | ~0.5 | i.e. weight ~1.0, bias ~0.0 |
| mlp_norm | ~0.99-1.06 | L2/L3 drift above 1 |
| lm_head.layer_norm | mean 1.107, std 0.413, **min -0.63, max 3.70** | strongly adapted |
| lm_head.bias | mean -0.0739, **min -3.31** | rare-token suppression |
| flow position_embeddings | 0.019 | row 0 is zero (padding_idx=0) |

Two facts matter for edit design: (a) the LM-head LayerNorm is the only LayerNorm with
negative gains and a large spread — it is doing real work at the output; (b) the decoder
bias is strongly negative on average, so LM-head logits are not centred.

## 6. Embedding-space structure

- row norms: mean 0.787, std 0.075; outliers: token id 1 (field value 0x00) = 2.646,
  MASK 65538 = 1.731, CLS 65537 = 0.474 (the *smallest* row).
- centred spectrum: top singular value explains 4.4% of variance, top-10 explain 21%,
  effective rank (exp of spectral entropy) = 304 of 512.
  => the embedding space is high-rank and distributed; there is no single "master
  direction" that a rank-1 edit could exploit. Edits must be targeted per token set.
- token identity is not ordinally encoded (id regressed on embeddings: R^2 < 0), i.e.
  embeddings are categorical, so "increase the value of byte X" is not a linear operation.

## 7. Candidate edit sites

| site | controls | edit shape | blast radius |
|---|---|---|---|
| word_embeddings (62.9% of params) | meaning of every field value | row edits, few rows | high: touches everything |
| compressEmbeddings | mixing of token / metadata / protocol | 512x1536 rows | medium-high |
| metaEmbeddingLayer1/2 | sensitivity to iat / bytes / pkt-count / direction | 1024x4, 512x1024 | medium |
| protocol embedding | per-IP-protocol prior | 18x512 rows | low |
| burst_encoder / flow_encoder | feature transforms, 4 layers each | full matrices or subspaces | medium-high |
| flow position_embeddings | burst order/position | 13x512 rows | low |
| attentivePooling | how a flow is summarised for the classifier | 512x512 + 512 | **low, direct readout** |
| dirPred / linearMetadataPred / swapped | the three pretraining readouts | 512x2, 512x3, 512x2 | **very low** |
| lm_head.dense / layer_norm / bias | MLM output distribution | 512x512, 512, 65600 | medium |

## 8. Constraints any patch must respect

- dtype F32, shapes fixed; `lm_head.decoder.weight` must never be edited directly
  (it is an alias of `word_embeddings.weight`; edit the embedding instead).
- the four `encoder.layer.*.position_embeddings.weight` are one tensor; patch layer 3 only.
- attention is **post-LN** (residual + LayerNorm inside `SelfOutput`), MLP is pre-LN:
  scaling an attention output-dense changes the residual scale entering the next block.
- `L` must be an exact multiple of `batch_max_burst_length` (= max burst length + 1
  in the batch); CLS = 65537, MASK = 65538, PAD = 0, field tokens are value+1.
- no gradient is used anywhere in this pipeline; patches are closed-form vector edits.

## 9. Open items before Phase 2

1. **Target capability is undefined.** Every later phase is conditional on it.
2. Evaluation data format: the released checkpoint was trained with `strip_payload=True`;
   the repo's `_strip_payload` stride (12 header + 6 payload) does not match the C++
   writer's layout (14 IP + 20 TCP + 12 payload), so re-deriving tokens from PCAPs is
   ambiguous. The Zenodo sampler in netFound arrow format (8 files x ~2 GB,
   ~0.45 MB/s here) is the unambiguous in-distribution source.
3. CPU-only: a 876-token forward pass is ~0.2 s/batch of 8; evaluation sets should stay
   in the low thousands of flows.
