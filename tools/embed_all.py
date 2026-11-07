import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, sys, time, numpy as np, torch, collections
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok

torch.set_num_threads(8)
BOS = 65537

def bucket_key(r):
    return (len(r["tokens"]), max(len(b) for b in r["tokens"]))

@torch.no_grad()
def embed_bucketed(model, flows, bs=32, log=None):
    order = sorted(range(len(flows)), key=lambda i: bucket_key(flows[i]))
    out = np.zeros((len(flows), 512), dtype=np.float32)
    i = 0
    t0 = time.time()
    while i < len(order):
        key = bucket_key(flows[order[i]])
        j = i
        while j < len(order) and bucket_key(flows[order[j]]) == key and j - i < bs:
            j += 1
        idx = order[i:j]
        b = tok.collate([flows[k] for k in idx])
        B, L = b["input_ids"].shape
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        o = model.base_transformer(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                   position_ids=pos, direction=b["direction"], iats=b["iats"],
                                   bytes=b["bytes"], pkt_count=b["pkt_count"], protocol=b["protocol"],
                                   dataset_burst_sizes=b["dataset_burst_sizes"], return_dict=True)
        h = o.last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[idx] = v.numpy()
        i = j
        if log and (i % 160 == 0):
            log(f"  {i}/{len(order)} flows, {time.time()-t0:.0f}s")
    return out


def nearest_centroid(Xs, ys, Xt):
    cent = np.stack([Xs[ys == c].mean(0) for c in (0, 1)])
    d = ((Xt[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return (d[:, 1] < d[:, 0]).astype(int), d


def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1p = 2 * prec * rec / max(prec + rec, 1e-9)
    np_ = tn / max(tn + fn, 1); nr = tn / max(tn + fp, 1)
    f1n = 2 * np_ * nr / max(np_ + nr, 1e-9)
    return dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                f1_pos=f1p, f1_neg=f1n, macro_f1=(f1p + f1n) / 2,
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1), acc=(tp + tn) / len(y))


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    allv = np.concatenate([pos, neg]); o = allv.argsort()
    ranks = np.empty(len(allv)); ranks[o] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    t0 = time.time()
    model = nf.load_model()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "flows_tcp.jsonl"))]
    SC = ["ctu-idseval-6-malicious-portscan-1", "ctu-idseval-6-malicious-portscan-2", "ctu-idseval-6-malicious-portscan-3"]
    ben = sorted([r for r in rows if r["capture"] == "ctu-idseval-6-benign-user-traffic-1"], key=lambda r: r["t0"])
    scn = sorted([r for r in rows if r["capture"] in SC], key=lambda r: (r["capture"], r["t0"]))
    print("benign", len(ben), "scan", len(scn))
    rng = np.random.default_rng(0)
    if len(scn) > 4000:
        idx = rng.choice(len(scn), 4000, replace=False); idx.sort(); scn = [scn[i] for i in idx]
    hb = len(ben) // 2
    ben_s, ben_t = ben[:hb], ben[hb:]
    hsk = len([r for r in scn if r["capture"] == SC[0]]) // 2
    scan_s = [r for c in SC for r in [x for x in scn if x["capture"] == c][: max(1, len([x for x in scn if x["capture"] == c]) // 2)]]
    scan_t = [r for c in SC for r in [x for x in scn if x["capture"] == c][max(1, len([x for x in scn if x["capture"] == c]) // 2):]]
    print(f"support: scan {len(scan_s)} benign {len(ben_s)} | test: scan {len(scan_t)} benign {len(ben_t)}")

    print("embedding benign support");  Es_b = embed_bucketed(model, ben_s, log=print)
    print("embedding benign test");     Et_b = embed_bucketed(model, ben_t, log=print)
    print("embedding scan support");    Es_s = embed_bucketed(model, scan_s, log=print)
    print("embedding scan test");       Et_s = embed_bucketed(model, scan_t, log=print)
    print("embed time", round(time.time() - t0, 0))

    Es = np.concatenate([Es_s, Es_b]); Et = np.concatenate([Et_s, Et_b])
    ys = np.array([1] * len(Es_s) + [0] * len(Es_b)); yt = np.array([1] * len(Et_s) + [0] * len(Et_b))
    np.savez(os.path.join(ROOT, "data", "emb_orig.npz"), Es=Es, ys=ys, Et=Et, yt=yt,
             cap_s=np.array([r["capture"] for r in scan_s + ben_s]),
             cap_t=np.array([r["capture"] for r in scan_t + ben_t]),
             t_s=np.array([r["t0"] for r in scan_s + ben_s]), t_t=np.array([r["t0"] for r in scan_t + ben_t]))
    p, d = nearest_centroid(Es, ys, Et)
    m = metrics(yt, p)
    score = d[:, 0] - d[:, 1]
    print("ORIGINAL MODEL  macroF1=%.4f prec=%.4f rec=%.4f FPR=%.4f FNR=%.4f acc=%.4f AUC=%.4f" %
          (m["macro_f1"], m["precision"], m["recall"], m["fpr"], m["fnr"], m["acc"], auc(score, yt)))
    print("  confusion:", {k: m[k] for k in ("tp", "fp", "tn", "fn")})
    # per-capture recall
    caps = np.array([r["capture"] for r in scan_t] + [r["capture"] for r in ben_t])
    for c in sorted(set(caps)):
        sel = caps == c
        print(f"  {c:42s} n={sel.sum():4d} correct={(p[sel]==yt[sel]).mean():.4f}")
    np.save(os.path.join(ROOT, "data", "pred_orig.npy"), p)
    print("total", round(time.time() - t0, 0), "s")


if __name__ == "__main__":
    main()
