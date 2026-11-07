import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Baseline: fixed nearest-centroid readout on frozen netFound flow embeddings."""
import json, sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf
import nf_tokenize as tok

FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
OUT = os.path.join(ROOT, "data")
SCAN_CAPS = ["ctu-idseval-6-malicious-portscan-1", "ctu-idseval-6-malicious-portscan-2",
             "ctu-idseval-6-malicious-portscan-3"]
BENIGN_CAP = "ctu-idseval-6-benign-user-traffic-1"


def load_flows(caps, limit_per_cap=None):
    out = []
    with open(FLOWS) as f:
        for line in f:
            r = json.loads(line)
            if r["capture"] in caps and r["protocol"] == 6:
                out.append(r)
    out.sort(key=lambda r: (r["capture"], r["t0"]))
    if limit_per_cap:
        by = {}
        for r in out:
            by.setdefault(r["capture"], []).append(r)
        out = [r for c in by for r in by[c][:limit_per_cap]]
    return out


def split_support_test(flows):
    """Chronological 50/50 split inside every capture (no cross-capture shuffling)."""
    sup, tst = [], []
    by = {}
    for r in flows:
        by.setdefault(r["capture"], []).append(r)
    for c, rs in by.items():
        rs.sort(key=lambda r: r["t0"])
        h = len(rs) // 2
        sup += rs[:h]; tst += rs[h:]
    return sup, tst


def nearest_centroid(Xs, ys, Xt, metric="l2"):
    if metric == "cos":
        n = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        Xs, Xt = n(Xs), n(Xt)
        cent = np.stack([Xs[ys == c].mean(0) for c in (0, 1)])
        cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        d = 1.0 - Xt @ cent.T
    else:
        cent = np.stack([Xs[ys == c].mean(0) for c in (0, 1)])
        d = ((Xt[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return (d[:, 1] < d[:, 0]).astype(int), d


def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1p = 2 * prec * rec / max(prec + rec, 1e-9)
    neg_prec = tn / max(tn + fn, 1); neg_rec = tn / max(tn + fp, 1)
    f1n = 2 * neg_prec * neg_rec / max(neg_prec + neg_rec, 1e-9)
    fpr = fp / max(fp + tn, 1); fnr = fn / max(fn + tp, 1)
    acc = (tp + tn) / max(len(y), 1)
    return dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                f1_positive=f1p, f1_negative=f1n, macro_f1=(f1p + f1n) / 2,
                fpr=fpr, fnr=fnr, accuracy=acc)


def auroc(scores_pos, scores_neg):
    """rank-based AUC, no sklearn dependency"""
    s = np.concatenate([scores_pos, scores_neg])
    order = s.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(scores_pos), len(scores_neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time()
    model = nf.load_model()
    scan = load_flows(SCAN_CAPS)
    ben = load_flows([BENIGN_CAP])
    print(f"scan flows: {len(scan)}  benign flows: {len(ben)}")
    sup_s, tst_s = split_support_test(scan)
    sup_b, tst_b = split_support_test(ben)
    print(f"scan support/test: {len(sup_s)}/{len(tst_s)}  benign support/test: {len(sup_b)}/{len(tst_b)}")

    rng = np.random.default_rng(0)
    def sub(rs, n):
        if len(rs) <= n: return rs
        idx = rng.choice(len(rs), n, replace=False); idx.sort()
        return [rs[i] for i in idx]
    sup_s, tst_s = sub(sup_s, 3000), sub(tst_s, 3000)

    def run(name, sup, tst):
        Es = tok.embed(model, sup, batch_size=64)
        Et = tok.embed(model, tst, batch_size=64)
        ys = np.array([1] * len(sup) + [0] * len(sup_b))
        Es = np.concatenate([Es, tok.embed(model, sup_b, batch_size=64)], 0)
        yt = np.array([1] * len(tst) + [0] * len(tst_b))
        Et = np.concatenate([Et, tok.embed(model, tst_b, batch_size=64)], 0)
        print(f"[{name}] embeddings {Es.shape} {Et.shape} in {time.time()-t0:.0f}s")
        np.savez(os.path.join(OUT, f"emb_{name}.npz"), Es=Es, ys=ys, Et=Et, yt=yt)
        for metric in ("l2", "cos"):
            p, d = nearest_centroid(Es, ys, Et, metric)
            m = metrics(yt, p)
            score = -d[:, 1] + d[:, 0] if metric == "l2" else d[:, 0] - d[:, 1]
            auc = auroc(score[yt == 1], score[yt == 0])
            print(f"  [{metric}] macroF1={m['macro_f1']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
                  f"FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} acc={m['accuracy']:.4f} AUC={auc:.4f} "
                  f"| TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        return Es, ys, Et, yt

    run("orig", sup_s, tst_s)
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
