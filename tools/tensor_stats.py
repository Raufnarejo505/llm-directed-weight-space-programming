import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, struct, sys
import numpy as np

PATH = os.path.join(ROOT, "checkpoints", "original", "model.safetensors")

class ST:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n).decode("utf-8"))
        self.base = 8 + n
        self.hdr.pop("__metadata__", None)
    def get(self, name):
        v = self.hdr[name]
        b, e = v["data_offsets"]
        dt = {"F32": np.float32, "F16": np.float16, "BF16": None}[v["dtype"]]
        self.f.seek(self.base + b)
        buf = self.f.read(e - b)
        return np.frombuffer(buf, dtype=dt).reshape(v["shape"]).copy()

st = ST(PATH)
names = sorted(st.hdr)
print(f"tensors={len(names)}  total={sum(int(np.prod(st.hdr[n]['shape'])) for n in names):,}")

def summarize(name, arr):
    a = np.asarray(arr, dtype=np.float32).ravel()
    l2 = float(np.sqrt((a.astype(np.float64)**2).sum()))
    return dict(n=a.size, mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()),
                absmax=float(np.abs(a).max()), l2=l2, n_zero=int((a == 0).sum()))

GROUPS = {}
for n in names:
    if n.startswith("base_transformer.embeddings.word_embeddings"): g = "emb.word"
    elif n.startswith("base_transformer.embeddings.protoEmbedding"): g = "emb.protocol"
    elif n.startswith("base_transformer.embeddings.metaEmbeddingLayer1"): g = "emb.meta_l1"
    elif n.startswith("base_transformer.embeddings.metaEmbeddingLayer2"): g = "emb.meta_l2"
    elif n.startswith("base_transformer.embeddings.compressEmbeddings"): g = "emb.compress"
    elif n.startswith("base_transformer.embeddings.LayerNorm"): g = "emb.norm"
    elif n.startswith("base_transformer.embeddings.token_type"): g = "emb.toktype"
    elif "encoder.layer." in n:
        lyr = n.split("encoder.layer.")[1][0]
        kind = "burst" if "burst_encoder" in n else ("flow" if "flow_encoder" in n else "pos")
        sub = ("attn.qkv" if ".attention.self." in n else
               "attn.out" if ".attention.output.dense" in n else
               "attn.norm" if ".attention.output.LayerNorm" in n else
               "mlp.Wi" if ".mlp.Wi" in n else
               "mlp.Wo" if ".mlp.Wo" in n else
               "mlp.norm" if "mlp_norm" in n else "position_emb")
        g = f"L{lyr}.{kind}.{sub}"
    elif n.startswith("lm_head.dense"): g = "head.dense"
    elif n.startswith("lm_head.layer_norm"): g = "head.norm"
    elif n.startswith("lm_head.bias"): g = "head.decoder_bias"
    elif n.startswith("attentivePooling"): g = "pool"
    elif n.startswith("dirPred"): g = "head.dirPred"
    elif n.startswith("linearMetadataPred"): g = "head.metaPred"
    elif n.startswith("swappedClassifier"): g = "head.swap"
    elif n.startswith("base_transformer.encoder.burst_positions"): g = "rope.burst"
    elif n.startswith("base_transformer.encoder.flow_positions"): g = "rope.flow"
    else: g = "other"
    GROUPS.setdefault(g, []).append(n)

print(f"\n{'group':22s} {'tensors':>4s} {'params':>10s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s} {'frac_zero':>9s}")
for g in sorted(GROUPS):
    ns = GROUPS[g]
    tot = sum(int(np.prod(st.hdr[n]['shape'])) for n in ns)
    s = [summarize(n, st.get(n)) for n in ns]
    w = np.array([x["n"] for x in s], dtype=np.float64)
    mean = float(np.sum([x["mean"]*x["n"] for x in s]) / w.sum())
    # pooled std across tensors
    var = float(np.sum([(x["std"]**2)*x["n"] for x in s]) / w.sum())
    mn = min(x["min"] for x in s); mx = max(x["max"] for x in s)
    z = sum(x["n_zero"] for x in s) / w.sum()
    print(f"{g:22s} {len(ns):4d} {tot:10,d} {mean:9.5f} {var**0.5:9.5f} {mn:9.4f} {mx:9.4f} {z:9.4f}")
