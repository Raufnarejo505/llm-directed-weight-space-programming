import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Bootstrap CIs for geometry metrics (between-class distance, separation ratio)."""
import json, os, sys
import numpy as np
ROOT = ROOT; RES = os.path.join(ROOT, "results")
ORDER = ["original", "p1a-k1-rank1-wow", "p1a-k2-rank1-wow", "p1a-k5-rank1-wow",
         "p1a-k10-rank1-wow", "p2-single-unit", "p3-control-pooling"]

def stats(E, y):
    mu = {c: E[y == c].mean(0) for c in (0, 1)}
    between = float(np.linalg.norm(mu[1] - mu[0]))
    within = {c: float(np.sqrt(((E[y == c] - mu[c]) ** 2).sum(1).mean())) for c in (0, 1)}
    pooled = float(np.sqrt(np.mean([within[0] ** 2, within[1] ** 2])))
    return between, pooled, between / (pooled + 1e-12)

def main(n=1000, seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for pid in ORDER:
        f = os.path.join(RES, f"emb_{pid}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); Es, ys, Et, yt = z["Es"], z["ys"], z["Et"], z["yt"]
        i1, i0 = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
        bs, ss = [], []
        for _ in range(n):
            idx = np.concatenate([rng.choice(i1, len(i1), True), rng.choice(i0, len(i0), True)])
            b, w, r = stats(Et[idx], yt[idx])
            bs.append(b); ss.append(r)
        b0, w0, r0 = stats(Et, yt)
        out[pid] = dict(between=b0, between_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        sep_ratio=r0, sep_ci=[float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))])
        print(f"{pid:22s} between={b0:6.2f} [{out[pid]['between_ci'][0]:5.2f},{out[pid]['between_ci'][1]:5.2f}]  "
              f"sep={r0:5.2f} [{out[pid]['sep_ci'][0]:4.2f},{out[pid]['sep_ci'][1]:4.2f}]")
    json.dump(out, open(os.path.join(RES, "geometry_ci.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
