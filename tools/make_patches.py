import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, sys, numpy as np
sys.path.insert(0, os.path.join(ROOT, "tools"))
import patch as P

st = np.load(os.path.join(ROOT, "data", "steering.npz"))
u, v = st["u"], st["v"]
e_c = 9.107146521228037
phi_scan = st["phi_mean_scan"]
phi_ben = st["phi_mean_scan"] - st["dphi"]     # saved benign mean was corrupted; recover from the difference
phi_std_scan, phi_std_ben = st["phi_std_scan"], st["phi_std_ben"]

# P2: single most discriminative MLP unit
gap = phi_scan - phi_ben
pooled_std = 0.5 * (phi_std_scan + phi_std_ben) + 1e-9
score = np.abs(gap) / pooled_std
k = int(np.argmax(score))
print(f"best unit k={k} gap={gap[k]:.3f} pooled_std={pooled_std[k]:.3f} score={score[k]:.3f}")

def mk(pid, ops, hyp, notes=""):
    return {"patch_id": pid, "created": "2026-11-06",
            "target_capability": "flow-level port-scan vs benign separability (fixed nearest-centroid readout)",
            "hypothesis": hyp, "notes": notes, "base_checkpoint": P.BASE, "operations": ops}

W = "base_transformer.encoder.layer.3.burst_encoder.mlp.Wo.weight"
patches = []
for kappa in (1, 2, 5, 10):
    a = kappa / e_c
    patches.append(mk(f"p1a-k{kappa}-rank1-wow", [
        {"op": "rank1", "tensor": W, "u": u.tolist(), "v": v.tolist(), "scale": float(a)}],
        f"Rank-1 steering of the last burst-encoder MLP readout along the class-mean direction; "
        f"kappa={kappa} means the average token is displaced by {kappa} units along u (class separation is {float(np.linalg.norm(st['dh'])):.1f}).",
        "u = unit normalised mean final-hidden difference (scan - benign); v = unit normalised mean layer-3 MLP hidden difference."))
e_phi = 0.5 * (phi_scan[k] + phi_ben[k])
col = (1.0 / (abs(e_phi) + 1e-9)) * u
print(f"single-unit column: E[phi_k]={e_phi:.3f} -> column scale={abs(1/e_phi):.4f}")
patches.append(mk("p2-single-unit", [
    {"op": "add", "tensor": W, "slice": [None, [k, k + 1]], "value": col.reshape(-1, 1).tolist()}],
    f"Single-column edit: add column to unit {k}, the MLP unit with the largest standardised scan/benign activation gap; "
    f"scaled so the average token is displaced by 1 unit along u.",
    "Touches 512 of 393,216 values (0.13% of the tensor)."))
patches.append(mk("p3-control-pooling", [
    {"op": "scale", "tensor": "attentivePooling.lin_proj.weight", "factor": 1.05}],
    "Negative control: scale the attentive-pooling projection, which is NOT on the masked-mean embedding path.",
    "Expected effect on the readout: exactly zero. If this patch changes the metric, the harness is leaking."))

for p in patches:
    path = os.path.join(ROOT, f"patches/{p['patch_id']}.json")
    json.dump(p, open(path, "w"), indent=1)
    rep = P.dry_run(p)
    print(f"{p['patch_id']:24s} errors={rep['errors']} touched={rep['n_params_touched']:,} ({100*rep['frac_params_touched']:.4f}%)")
