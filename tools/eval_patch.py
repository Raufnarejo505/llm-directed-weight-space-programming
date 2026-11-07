import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Task E: apply one patch, re-embed the fixed evaluation set, measure metrics + geometry."""
import json, sys, os, time, hashlib
import numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok, patch as P
from embed_all import embed_bucketed, nearest_centroid, metrics, auc
torch.set_num_threads(2)

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
SC = ["ctu-idseval-6-malicious-portscan-1", "ctu-idseval-6-malicious-portscan-2", "ctu-idseval-6-malicious-portscan-3"]
BEN = "ctu-idseval-6-benign-user-traffic-1"


def get_split():
    """Exactly the selection used for the baseline (seed 0, 4000 scan flows)."""
    rows = [json.loads(l) for l in open(FLOWS)]
    ben = sorted([r for r in rows if r["capture"] == BEN], key=lambda r: r["t0"])
    scn = sorted([r for r in rows if r["capture"] in SC], key=lambda r: (r["capture"], r["t0"]))
    rng = np.random.default_rng(0)
    if len(scn) > 4000:
        idx = rng.choice(len(scn), 4000, replace=False); idx.sort(); scn = [scn[i] for i in idx]
    hb = len(ben) // 2
    ben_s, ben_t = ben[:hb], ben[hb:]
    scan_s = [r for c in SC for r in [x for x in scn if x["capture"] == c][: max(1, len([x for x in scn if x["capture"] == c]) // 2)]]
    scan_t = [r for c in SC for r in [x for x in scn if x["capture"] == c][max(1, len([x for x in scn if x["capture"] == c]) // 2):]]
    return scan_s, ben_s, scan_t, ben_t


def geometry(Es, ys, Et, yt):
    mu = {c: Es[ys == c].mean(0) for c in (0, 1)}
    mu_t = {c: Et[yt == c].mean(0) for c in (0, 1)}
    between = float(np.linalg.norm(mu[1] - mu[0]))
    within = {c: float(np.sqrt(((Es[ys == c] - mu[c]) ** 2).sum(1).mean())) for c in (0, 1)}
    within_pooled = float(np.sqrt(np.mean([within[0] ** 2, within[1] ** 2])))
    between_test = float(np.linalg.norm(mu_t[1] - mu_t[0]))
    return dict(between_support=between, between_test=between_test,
                within_scan=within[1], within_benign=within[0], within_pooled=within_pooled,
                separation_ratio_support=between / (within_pooled + 1e-12),
                separation_ratio_test=between_test / (within_pooled + 1e-12))


def bootstrap(yt, pred, scores, n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    idx1 = np.flatnonzero(yt == 1); idx0 = np.flatnonzero(yt == 0)
    out = []
    for _ in range(n_boot):
        i1 = rng.choice(idx1, len(idx1), replace=True)
        i0 = rng.choice(idx0, len(idx0), replace=True)
        idx = np.concatenate([i1, i0])
        m = metrics(yt[idx], pred[idx])
        out.append((m["macro_f1"], m["precision"], m["recall"], m["fpr"], auc(scores[idx], yt[idx])))
    out = np.array(out)
    names = ["macro_f1", "precision", "recall", "fpr", "auc"]
    return {n: dict(mean=float(out[:, i].mean()), lo=float(np.percentile(out[:, i], 2.5)),
                    hi=float(np.percentile(out[:, i], 97.5))) for i, n in enumerate(names)}


def main(patch_id):
    t0 = time.time()
    f = os.path.join(RES, f"{patch_id}.json")
    if os.path.exists(f):
        print("already done", patch_id); return
    scan_s, ben_s, scan_t, ben_t = get_split()
    assert (len(scan_s), len(ben_s), len(scan_t), len(ben_t)) == (1999, 380, 2001, 381), \
        (len(scan_s), len(ben_s), len(scan_t), len(ben_t))

    info = {"patch_id": patch_id}
    if patch_id != "original":
        ppath = os.path.join(ROOT, "patches", patch_id + ".json")
        p = P.load_patch(ppath)
        out_dir = os.path.join(ROOT, "checkpoints", "patched", patch_id)
        if not os.path.isdir(out_dir):
            out_dir, man = P.apply_patch(ppath)
        else:
            man = json.load(open(os.path.join(out_dir, "patch_manifest.json")))
        info.update(patch_sha256=P.sha256_file(ppath),
                    checkpoint_sha256=man["patched_sha256"],
                    base_sha256=man["base_sha256"],
                    tensors=list(man["summary"].get("ops", [])),
                    n_params_changed=man["summary"]["n_params_changed"],
                    frac_params_changed=man["summary"]["frac_params_changed"],
                    global_rel_l2=man["summary"]["global_rel_l2"],
                    per_tensor=man["tensors"])
        model = nf.load_model(out_dir)
    else:
        info.update(checkpoint_sha256=P.sha256_file(os.path.join(ROOT, "checkpoints", "original", "model.safetensors")))
        model = nf.load_model()

    print("embedding support", flush=True)
    Es = np.concatenate([embed_bucketed(model, scan_s), embed_bucketed(model, ben_s)])
    print("embedding test", flush=True)
    Et = np.concatenate([embed_bucketed(model, scan_t), embed_bucketed(model, ben_t)])
    ys = np.array([1] * len(scan_s) + [0] * len(ben_s)); yt = np.array([1] * len(scan_t) + [0] * len(ben_t))
    np.savez(os.path.join(RES, f"emb_{patch_id}.npz"), Es=Es, ys=ys, Et=Et, yt=yt,
             caps=np.array([r["capture"] for r in scan_t] + [r["capture"] for r in ben_t]))

    pred, d = nearest_centroid(Es, ys, Et)
    scores = d[:, 0] - d[:, 1]
    m = metrics(yt, pred)
    info.update(metrics=m, auc=float(auc(scores, yt)), geometry=geometry(Es, ys, Et, yt),
                bootstrap=bootstrap(yt, pred, scores))
    caps = np.array([r["capture"] for r in scan_t] + [r["capture"] for r in ben_t])
    info["per_capture_accuracy"] = {c: float((pred[caps == c] == yt[caps == c]).mean()) for c in sorted(set(caps))}
    # drift vs original embeddings
    orig = np.load(os.path.join(ROOT, "data", "emb_orig.npz"))
    if orig["Et"].shape == Et.shape:
        dEt = Et - orig["Et"]; dEs = Es - orig["Es"]
        info["drift"] = dict(mean_rel_test=float(np.linalg.norm(dEt, axis=1).mean() / (np.linalg.norm(orig["Et"], axis=1).mean() + 1e-9)),
                             mean_rel_support=float(np.linalg.norm(dEs, axis=1).mean() / (np.linalg.norm(orig["Es"], axis=1).mean() + 1e-9)),
                             cosine_test=float(np.mean(np.sum(Et * orig["Et"], 1) / (np.linalg.norm(Et, axis=1) * np.linalg.norm(orig["Et"], axis=1) + 1e-9))),
                             centroid_shift=float(np.linalg.norm(Et.mean(0) - orig["Et"].mean(0))))
    info["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(info, open(f, "w"), indent=2)
    print(json.dumps({k: info[k] for k in ("patch_id", "metrics", "auc", "geometry", "drift") if k in info}, indent=1), flush=True)
    print("done in", info["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
