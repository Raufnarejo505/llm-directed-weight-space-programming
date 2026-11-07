import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Remaining Task-C evidence: hard-subset statistics, metadata control, regression fingerprint."""
import json, sys, time, numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok
from embed_all import metrics, auc
from analysis_c import embed_layers, nc_auc, load
from regression_suite import build_suite, evaluate
torch.set_num_threads(8)

def hard_subset_stats(ben, scn):
    def npk(r): return int(sum(r["counts"]))
    b1 = [r for r in ben if npk(r) == 1]
    s1 = [r for r in scn if npk(r) == 1]
    print(f"benign flows: {len(ben)}  of which 1-packet: {len(b1)}")
    print(f"scan   flows: {len(scn)}  of which 1-packet: {len(s1)}")
    if b1:
        sig = lambda r: tuple(r["tokens"][0][:12])
        from collections import Counter
        print("  benign 1-packet signature (first burst tokens):", Counter(sig(r) for r in b1).most_common(3))
        print("  scan   1-packet signature (first burst tokens):", Counter(sig(r) for r in s1).most_common(3))
        same = len(set(sig(r) for r in b1) & set(sig(r) for r in s1))
        print(f"  identical 12-token signatures shared by both classes: {same}")
    return b1, s1

def permute_meta(flows, rng):
    """permute metadata VALUES across bursts within the same flow (keeps lengths aligned)"""
    out = []
    for r in flows:
        q = dict(r)
        for k in ("counts", "bytes", "iats"):
            v = list(r[k])
            if len(v) > 1:
                q[k] = [v[i] for i in rng.permutation(len(v))]
        out.append(q)
    return out

def main():
    t0 = time.time()
    ben, scn = load()
    b1, s1 = hard_subset_stats(ben, scn)
    model = nf.load_model()
    rng = np.random.default_rng(0)
    # metadata-only control on a small set
    nb, ns = 100, 300
    bs, bt = ben[:nb], ben[-nb:]
    ss, st = s1[:ns], s1[-ns:]
    Ls = embed_layers(model, ss + bs); Lt = embed_layers(model, st + bt)
    ys = np.array([1]*len(ss) + [0]*len(bs)); yt = np.array([1]*len(st) + [0]*len(bt))
    m, a = nc_auc(Ls[4], ys, Lt[4], yt)
    print(f"[control] 1-packet subset, unmodified: macroF1={m['macro_f1']:.4f} FPR={m['fpr']:.4f} AUC={a:.4f}", flush=True)
    ps = permute_meta(ss + bs, rng); pt = permute_meta(st + bt, rng)
    Ls2 = embed_layers(model, ps); Lt2 = embed_layers(model, pt)
    m, a = nc_auc(Ls2[4], ys, Lt2[4], yt)
    print(f"[control] 1-packet subset, metadata permuted: macroF1={m['macro_f1']:.4f} FPR={m['fpr']:.4f} AUC={a:.4f}", flush=True)
    # regression fingerprint
    flows = build_suite()
    r = evaluate(model, flows)
    print("[regression] " + json.dumps(r), flush=True)
    json.dump(r, open(os.path.join(ROOT, "data", "regression_orig.json"), "w"), indent=2)
    print("elapsed", round(time.time()-t0), "s", flush=True)

if __name__ == "__main__":
    main()
