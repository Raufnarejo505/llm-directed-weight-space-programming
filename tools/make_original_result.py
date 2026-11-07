import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, sys, os, shutil, numpy as np
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_patch import geometry, bootstrap, metrics, auc, nearest_centroid
ROOT = ROOT; RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
z = np.load(os.path.join(ROOT, "data", "emb_orig.npz"))
Es, ys, Et, yt = z["Es"], z["ys"], z["Et"], z["yt"]
np.savez(os.path.join(RES, "emb_original.npz"), Es=Es, ys=ys, Et=Et, yt=yt, caps=z["cap_t"])
pred, d = nearest_centroid(Es, ys, Et)
scores = d[:, 0] - d[:, 1]
info = dict(patch_id="original",
            checkpoint_sha256="9e0080bbf00f6cd54e1394eb7ad484dd3f5b0758404c80d9ac91a77243e06a70",
            metrics=metrics(yt, pred), auc=float(auc(scores, yt)),
            geometry=geometry(Es, ys, Et, yt), bootstrap=bootstrap(yt, pred, scores),
            drift=dict(mean_rel_test=0.0, mean_rel_support=0.0, cosine_test=1.0, centroid_shift=0.0))
json.dump(info, open(os.path.join(RES, "original.json"), "w"), indent=2)
print(json.dumps(info["metrics"], indent=1)); print("auc", info["auc"]); print(json.dumps(info["geometry"], indent=1))
