import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Aggregate Task-E results: metrics, geometry, paired bootstrap, decisions."""
import json, os, glob
import numpy as np

RES = os.path.join(ROOT, "results")
ORDER = ["original", "p1a-k1-rank1-wow", "p1a-k2-rank1-wow", "p1a-k5-rank1-wow",
         "p1a-k10-rank1-wow", "p2-single-unit", "p3-control-pooling"]


def load_pred(pid):
    z = np.load(os.path.join(RES, f"emb_{pid}.npz"))
    Es, ys, Et, yt = z["Es"], z["ys"], z["Et"], z["yt"]
    cent = np.stack([Es[ys == c].mean(0) for c in (0, 1)])
    d = ((Et[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return (d[:, 1] < d[:, 0]).astype(int), d[:, 0] - d[:, 1], yt, Et, Es, yt


def flip_count(pa, pb):
    return int((pa != pb).sum())


def m(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    f1p = 2 * pr * rc / max(pr + rc, 1e-9)
    np_ = tn / max(tn + fn, 1); nr = tn / max(tn + fp, 1)
    f1n = 2 * np_ * nr / max(np_ + nr, 1e-9)
    return dict(macro_f1=(f1p + f1n) / 2, precision=pr, recall=rc, fpr=fp / max(fp + tn, 1),
                fnr=fn / max(fn + tp, 1), acc=(tp + tn) / len(y), tp=tp, fp=fp, tn=tn, fn=fn)


def paired_bootstrap(pa, pb, y, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0)
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i1, len(i1), True), rng.choice(i0, len(i0), True)])
        out.append(m(y[idx], pa[idx])["macro_f1"] - m(y[idx], pb[idx])["macro_f1"])
    out = np.array(out)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def main():
    res = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        j = json.load(open(f))
        if "patch_id" in j:
            res[j["patch_id"]] = j
    p_orig, s_orig, yt, Et0, Es0, _ = load_pred("original")
    print(f"{'patch':22s} {'macroF1':>8s} {'95% CI':>16s} {'prec':>7s} {'rec':>7s} {'FPR':>7s} {'AUC':>7s} {'dF1 vs orig':>22s} {'flips':>6s}")
    rows = []
    for pid in ORDER:
        if pid not in res:
            continue
        r = res[pid]; mm = r["metrics"]
        p, s, _, Et, Es, _ = load_pred(pid)
        dm, dlo, dhi, pnull = paired_bootstrap(p, p_orig, yt)
        ci = r["bootstrap"]["macro_f1"]
        fl = flip_count(p, p_orig)
        r["flips_vs_original"] = fl
        # decision margin: |d0 - d1| (larger = more confident)
        cent = np.stack([Es[ys == c].mean(0) for c in (0, 1)]) if (ys := res[pid] and np.load(os.path.join(RES, f"emb_{pid}.npz"))["ys"]) is not None else None
        print(f"{pid:22s} {mm['macro_f1']:8.4f} [{ci['lo']:.4f},{ci['hi']:.4f}] {mm['precision']:7.4f} "
              f"{mm['recall']:7.4f} {mm['fpr']:7.4f} {r['auc']:7.4f}  {dm:+.4f} [{dlo:+.4f},{dhi:+.4f}] p0={pnull:.3f} {fl:6d}")
        rows.append((pid, r, dm, dlo, dhi))
    print()
    for pid, r, *_ in rows:
        g = r["geometry"]; d = r.get("drift", {})
        print(f"{pid:22s} between={g['between_support']:7.2f} within={g['within_pooled']:7.2f} "
              f"sep={g['separation_ratio_support']:6.3f} drift_rel={d.get('mean_rel_test', 0):.4f} "
              f"cos={d.get('cosine_test', 1):.5f} centroid_shift={d.get('centroid_shift', 0):.4f}")
    print()
    for pid, r, *_ in rows:
        if "n_params_changed" in r:
            print(f"{pid:22s} params={r['n_params_changed']:,} frac={r['frac_params_changed']:.5f} "
                  f"relL2={r['global_rel_l2']:.5f} ckpt={r['checkpoint_sha256'][:12]}")
    json.dump({pid: {k: v for k, v in r.items() if k != "per_tensor"} for pid, r, *_ in rows},
              open(os.path.join(RES, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
