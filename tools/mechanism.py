import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Mechanism: how far does each group move, and along which direction?"""
import json, sys, os
import numpy as np
ROOT = ROOT; RES = os.path.join(ROOT, "results")
st = np.load(os.path.join(ROOT, "data", "steering.npz"))
u = st["u"]

def load_pred(pid):
    z = np.load(os.path.join(RES, f"emb_{pid}.npz"))
    Es, ys, Et, yt = z["Es"], z["ys"], z["Et"], z["yt"]
    cent = np.stack([Es[ys == c].mean(0) for c in (0, 1)])
    d = ((Et[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return (d[:, 1] < d[:, 0]).astype(int), Et, Et - Et, yt, cent

def main(pid):
    z = np.load(os.path.join(RES, f"emb_{pid}.npz"))
    Et, yt = z["Et"], z["yt"]
    zo = np.load(os.path.join(RES, "emb_original.npz"))
    Eto, yto = zo["Et"], zo["yt"]
    assert np.array_equal(yt, yto)
    p, _, _, _, cent = load_pred(pid)
    po, _, _, _, cento = load_pred("original")
    disp = Et - Eto
    dn = np.linalg.norm(disp, axis=1)
    proj = disp @ u
    groups = {
        "scan (TP)": (yt == 1) & (p == 1),
        "benign TN": (yt == 0) & (p == 0),
        "benign FP": (yt == 0) & (p == 1),
    }
    print(f"\n[{pid}]  total drift mean={dn.mean():.3f}  centroid shift={np.linalg.norm(cent[1]-cento[1]):.3f} (scan) "
          f"{np.linalg.norm(cent[0]-cento[0]):.3f} (benign)")
    for name, m in groups.items():
        if m.sum() == 0: continue
        print(f"  {name:12s} n={int(m.sum()):4d}  |disp|={dn[m].mean():7.3f}  proj_u={proj[m].mean():7.3f}")
    # margin
    d0 = ((Et - cent[0]) ** 2).sum(-1) ** 0.5; d1 = ((Et - cent[1]) ** 2).sum(-1) ** 0.5
    d0o = ((Eto - cento[0]) ** 2).sum(-1) ** 0.5; d1o = ((Eto - cento[1]) ** 2).sum(-1) ** 0.5
    marg = d1 - d0; margo = d1o - d0o
    print(f"  margin (d_scan - d_benign): scan {margo[yt==1].mean():.2f}->{marg[yt==1].mean():.2f} | "
          f"benignTN {margo[(yt==0)&(p==0)].mean():.2f}->{marg[(yt==0)&(p==0)].mean():.2f} | "
          f"benignFP {margo[(yt==0)&(p==1)].mean():.2f}->{marg[(yt==0)&(p==1)].mean():.2f}")
    print(f"  margin change: scan {(marg-margo)[yt==1].mean():+.3f}  benignTN {(marg-margo)[(yt==0)&(p==0)].mean():+.3f} "
          f"benignFP {(marg-margo)[(yt==0)&(p==1)].mean():+.3f}")

if __name__ == "__main__":
    for pid in sys.argv[1:]:
        main(pid)
