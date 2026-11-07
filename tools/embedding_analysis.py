import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import numpy as np, json, struct
from safetensors.numpy import load_file
p = os.path.join(ROOT, "checkpoints", "original", "model.safetensors")
t = load_file(p)
E = t["base_transformer.embeddings.word_embeddings.weight"].astype(np.float64)  # [65600,512]
print("E:", E.shape)
norms = np.linalg.norm(E, axis=1)
print(f"row-norm: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")
for tid, lbl in [(0,"PAD"),(1,"tok=0"),(255,"tok=254"),(65536,"tok=65535"),(65537,"CLS"),(65538,"MASK"),(65540,"EOS")]:
    print(f"  {lbl:10s} id={tid:6d} norm={norms[tid]:.4f}")
# how much of the norm is the common mean component
mu = E.mean(axis=0)
print("mean-vector norm:", float(np.linalg.norm(mu)))
Ec = E - mu
print("centered row-norm mean:", float(np.linalg.norm(Ec, axis=1).mean()))
# spectrum
s = np.linalg.svd(Ec, compute_uv=False)
ev = s**2 / (s**2).sum()
print("top-10 singular values:", np.round(s[:10], 3).tolist())
print("var explained top1..10:", np.round(np.cumsum(ev)[:10], 3).tolist())
print("effective rank (exp entropy):", float(np.exp(-(ev* np.log(ev + 1e-30)).sum())))
# is token id linearly encoded? regress id onto embeddings
ids = np.arange(E.shape[0], dtype=np.float64)
X = Ec
w, *_ = np.linalg.lstsq(X, ids, rcond=None)
pred = X @ w
print(f"id-vs-embedding R^2: {1 - ((ids-pred)**2).sum()/((ids-ids.mean())**2).sum():.4f}")
