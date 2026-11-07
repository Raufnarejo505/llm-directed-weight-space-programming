import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, struct, collections
import numpy as np
from safetensors.numpy import load_file
t = load_file(os.path.join(ROOT, "checkpoints", "original", "model.safetensors"))
tot = sum(v.size for v in t.values())
def n(k): return t[k].size
buckets = collections.OrderedDict()
buckets["token embedding (word_embeddings)"] = n("base_transformer.embeddings.word_embeddings.weight")
buckets["metadata MLP (4->1024->512)"] = sum(v.size for k,v in t.items() if "metaEmbeddingLayer" in k)
buckets["protocol embedding (18x512)"] = n("base_transformer.embeddings.protoEmbedding.weight")
buckets["embedding fusion (compressEmbeddings 1536->512)"] = sum(v.size for k,v in t.items() if "compressEmbeddings" in k)
buckets["embedding LayerNorm"] = sum(v.size for k,v in t.items() if k.startswith("base_transformer.embeddings.LayerNorm"))
buckets["token_type embeddings (UNUSED)"] = n("base_transformer.embeddings.token_type_embeddings.weight")
for blk in ["burst_encoder","flow_encoder"]:
    buckets[f"{blk} (x4 layers): attention"] = sum(v.size for k,v in t.items() if f"{blk}.attention" in k)
    buckets[f"{blk} (x4 layers): MLP + norm"] = sum(v.size for k,v in t.items() if f"{blk}.mlp" in k)
buckets["flow-level position embeddings (13x512, tied x4)"] = n("base_transformer.encoder.layer.3.position_embeddings.weight")
buckets["RoPE tables (burst 73x128, flow 13x128)"] = sum(v.size for k,v in t.items() if "encoder.burst_positions" in k or "encoder.flow_positions" in k)
buckets["LM head: dense + norm"] = sum(v.size for k,v in t.items() if k.startswith("lm_head.dense") or k.startswith("lm_head.layer_norm"))
buckets["LM head: decoder bias (vocab)"] = n("lm_head.bias")
buckets["attentive pooling"] = sum(v.size for k,v in t.items() if k.startswith("attentivePooling"))
buckets["pretraining heads (dir/meta/swap)"] = sum(v.size for k,v in t.items() if k.startswith(("dirPred","linearMetadataPred","swappedClassifier")))
s = sum(buckets.values())
print(f"accounted {s:,} / {tot:,}  (diff {tot-s})")
print(f"{'component':52s} {'params':>10s} {'%':>6s}")
for k,v in buckets.items():
    print(f"{k:52s} {v:10,d} {100*v/tot:6.2f}")
# per-layer totals
for i in range(4):
    p = sum(v.size for k,v in t.items() if f"encoder.layer.{i}." in k)
    print(f"layer {i}: {p:,}")
