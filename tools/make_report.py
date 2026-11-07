import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Generate reports/03_taskE.md from results/*.json"""
import json, os, glob
import numpy as np

ROOT = ROOT; RES = os.path.join(ROOT, "results")
ORDER = ["original", "p1a-k1-rank1-wow", "p1a-k2-rank1-wow", "p1a-k5-rank1-wow",
         "p1a-k10-rank1-wow", "p2-single-unit", "p3-control-pooling"]


def fmt(x, n=4):
    return f"{x:.{n}f}"


def main():
    res = {}
    for pid in ORDER:
        f = os.path.join(RES, pid + ".json")
        if os.path.exists(f):
            res[pid] = json.load(open(f))
    reg = {}
    for f in glob.glob(os.path.join(RES, "regression_*.json")):
        reg[os.path.basename(f)[11:-5]] = json.load(open(f))
    L = []
    A = L.append
    A("# Task E - Comparative weight-space intervention report")
    A("")
    A("Six independent patches, each applied to its own checkpoint copy. The original")
    A("checkpoint is never modified. No gradients were used anywhere in this pipeline.")
    A("")
    A("Readout is fixed for every condition: flow embedding = attention-mask-weighted mean")
    A("of the final hidden states; class centroids from the 1,999 scan / 380 benign support")
    A("flows; nearest centroid (Euclidean). Test set: 2,001 scan / 381 benign.")
    A("")
    A("## 1. Provenance")
    A("")
    A("| checkpoint | sha256 (first 16) |")
    A("|---|---|")
    for pid in res:
        A(f"| {pid} | {res[pid]['checkpoint_sha256'][:16]} |")
    A("")
    if "p1a-k1-rank1-wow" in res:
        A("| patch | patch sha256 | tensor modified | params changed | % of 53,383,495 | rel. L2 |")
        A("|---|---|---|---|---|---|")
        for pid in ORDER[1:]:
            if pid not in res: continue
            r = res[pid]
            tn = ", ".join(o["tensor"] for o in r["tensors"]) if r.get("tensors") else "-"
            A(f"| {pid} | {r['patch_sha256'][:16]} | {tn} | {r['n_params_changed']:,} | "
              f"{100*r['frac_params_changed']:.4f} | {r['global_rel_l2']:.5f} |")
        A("")
    A("## 2. Downstream classification (fixed readout)")
    A("")
    A("| condition | macro F1 | 95% CI | precision | recall | FPR | AUC | TP | FP | TN | FN |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for pid in res:
        r = res[pid]; m = r["metrics"]; ci = r["bootstrap"]["macro_f1"]
        A(f"| {pid} | {fmt(m['macro_f1'])} | [{fmt(ci['lo'])}, {fmt(ci['hi'])}] | {fmt(m['precision'])} | "
          f"{fmt(m['recall'])} | {fmt(m['fpr'])} | {fmt(r['auc'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
    A("")
    A("## 3. Representation geometry")
    A("")
    A("| condition | between-class | within(scan) | within(benign) | separation ratio | drift (rel.) | cosine | centroid shift |")
    A("|---|---|---|---|---|---|---|---|")
    for pid in res:
        g = res[pid]["geometry"]; d = res[pid].get("drift", {})
        A(f"| {pid} | {fmt(g['between_support'],2)} | {fmt(g['within_scan'],2)} | {fmt(g['within_benign'],2)} | "
          f"{fmt(g['separation_ratio_support'],3)} | {fmt(d.get('mean_rel_test',0),5)} | "
          f"{fmt(d.get('cosine_test',1),5)} | {fmt(d.get('centroid_shift',0),4)} |")
    A("")
    A("## 4. Edit-strength response (rank-1, kappa)")
    A("")
    A("| kappa | alpha | macro F1 | FPR | separation ratio | drift |")
    A("|---|---|---|---|---|---|")
    alph = {"p1a-k1-rank1-wow": 0.1098, "p1a-k2-rank1-wow": 0.2196, "p1a-k5-rank1-wow": 0.5491,
            "p1a-k10-rank1-wow": 1.0982}
    for pid, a in alph.items():
        if pid not in res: continue
        r = res[pid]; m = r["metrics"]
        A(f"| {pid.split('-')[1][1:]} | {a:.4f} | {fmt(m['macro_f1'])} | {fmt(m['fpr'])} | "
          f"{fmt(r['geometry']['separation_ratio_support'],3)} | {fmt(r['drift']['mean_rel_test'],5)} |")
    A("")
    A("## 5. Regression on pretraining heads")
    A("")
    if reg:
        A("| condition | MLM loss | MLM top-1 | direction acc | metadata MAE | swap loss |")
        A("|---|---|---|---|---|---|")
        for pid in res:
            key = "orig" if pid == "original" else pid
            if key not in reg: continue
            g = reg[key]
            A(f"| {pid} | {fmt(g['mlm_loss'],3)} | {fmt(g['mlm_top1'])} | {fmt(g['dir_acc'])} | "
              f"{fmt(g['meta_mae'],4)} | {fmt(g['swap_loss'])} |")
        A("")
    else:
        A("(regression results pending)")
        A("")
    A("## 6. Interpretation")
    A("")
    A("Separated by construction: weight-space intervention (what was edited), representation")
    A("geometry (between/within distances, drift), downstream classification (macro F1 on this")
    A("corpus), and capability acquisition (not claimed). The corpus is separable by IP TTL,")
    A("TCP window and IP length alone (single-feature AUC 1.000), so no result here is evidence")
    A("of port-scan detection.")
    A("")
    open(os.path.join(ROOT, "reports", "03_taskE.md"), "w").write("\n".join(L))
    print("wrote", len(L), "lines")


if __name__ == "__main__":
    main()
