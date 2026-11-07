import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Task C: where does scan/benign separability live, and which input channels carry it?"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok
from embed_all import metrics, auc
torch.set_num_threads(8)

FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")

def load():
    rows = [json.loads(l) for l in open(FLOWS)]
    ben = sorted([r for r in rows if r["capture"] == "ctu-idseval-6-benign-user-traffic-1"], key=lambda r: r["t0"])
    scn = sorted([r for r in rows if "portscan" in r["capture"]], key=lambda r: (r["capture"], r["t0"]))
    return ben, scn

def bucket_key(r): return (len(r["tokens"]), max(len(b) for b in r["tokens"]))

@torch.no_grad()
def embed_layers(model, flows, bs=24):
    """returns pooled [N,512] for embedding output + each of 4 layers"""
    order = sorted(range(len(flows)), key=lambda i: bucket_key(flows[i]))
    L5 = [np.zeros((len(flows), 512), np.float32) for _ in range(5)]
    i = 0
    while i < len(order):
        key = bucket_key(flows[order[i]]); j = i
        while j < len(order) and bucket_key(flows[order[j]]) == key and j - i < bs: j += 1
        idx = order[i:j]
        b = tok.collate([flows[k] for k in idx]); B, L = b["input_ids"].shape
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        o = model.base_transformer(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=pos,
                                   direction=b["direction"], iats=b["iats"], bytes=b["bytes"], pkt_count=b["pkt_count"],
                                   protocol=b["protocol"], dataset_burst_sizes=b["dataset_burst_sizes"],
                                   return_dict=True, output_hidden_states=True)
        m = b["attention_mask"].unsqueeze(-1).float()
        hs = o.hidden_states  # tuple: embeddings + 4 layers
        for li in range(5):
            h = hs[li]; v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            L5[li][idx] = v.numpy()
        i = j
    return L5

def nc_auc(Xs, ys, Xt, yt):
    cent = np.stack([Xs[ys == c].mean(0) for c in (0, 1)])
    d = ((Xt[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    s = d[:, 0] - d[:, 1]
    p = (d[:, 1] < d[:, 0]).astype(int)
    return metrics(yt, p), auc(s, yt)

def shuffle_channels(flows, rng, header=(), meta=()):
    out = [dict(r) for r in flows]; perm = rng.permutation(len(flows))
    for i, r in enumerate(out):
        src = flows[perm[i]]
        toks = [list(b) for b in r["tokens"]]; st = [list(b) for b in src["tokens"]]
        for bi, b in enumerate(toks):
            for ti in range(len(b)):
                if ti % 12 in header and bi < len(st) and ti < len(st[bi]):
                    b[ti] = st[bi][ti]
        out[i]["tokens"] = toks
        for k in meta: out[i][k] = src[k]
        if "counts" in meta: out[i]["counts"] = src["counts"]
    return out

def main():
    t0 = time.time()
    ben, scn = load(); rng = np.random.default_rng(0)
    nb = 160
    ben_s, ben_t = ben[:nb], ben[-nb:]
    s3 = [r for r in scn if r["capture"].endswith("portscan-3")]
    sc_s, sc_t = s3[:400], s3[-400:]
    model = nf.load_model()
    print(f"support {len(sc_s)}scan/{len(ben_s)}ben  test {len(sc_t)}scan/{len(ben_t)}ben", flush=True)

    Ls = embed_layers(model, sc_s + ben_s)
    Lt = embed_layers(model, sc_t + ben_t)
    ys = np.array([1]*len(sc_s) + [0]*len(ben_s)); yt = np.array([1]*len(sc_t) + [0]*len(ben_t))
    names = ["embeddings", "layer0", "layer1", "layer2", "layer3(final)"]
    print("\n== per-layer pooled separability (nearest centroid) ==", flush=True)
    for i, nm in enumerate(names):
        m, a = nc_auc(Ls[i], ys, Lt[i], yt)
        print(f"  {nm:14s} macroF1={m['macro_f1']:.4f} FPR={m['fpr']:.4f} AUC={a:.4f}", flush=True)
    for i, nm in enumerate(names):
        Es, Et = Ls[i], Lt[i]
        cs = np.concatenate([Es[ys == 1].mean(0), Es[ys == 0].mean(0)])
        cent = np.stack([Es[ys == c].mean(0) for c in (0, 1)])
        dn = cent[1] - cent[0]; dn /= np.linalg.norm(dn) + 1e-9
        proj_s = Es @ dn; proj_t = Et @ dn
        sep = (proj_t[yt == 1].mean() - proj_t[yt == 0].mean()) / (0.5*(proj_t[yt == 1].std() + proj_t[yt == 0].std()) + 1e-9)
        print(f"  {nm:14s} |d|={np.linalg.norm(cent[1]-cent[0]):.3f}  Cohen d(proj)={sep:.2f}", flush=True)
    print("elapsed", round(time.time()-t0), flush=True)

    print("\n== channel-shuffle controls (final layer, test half) ==", flush=True)
    for tag, hdr, mt in [("ttl+win+iplen", (2,4,6), ()), ("meta(counts,bytes,iats)", (), ("counts","bytes","iats")),
                         ("both", (2,4,6), ("counts","bytes","iats"))]:
        ss = shuffle_channels(sc_s + ben_s, rng, header=hdr, meta=mt)
        st = shuffle_channels(sc_t + ben_t, rng, header=hdr, meta=mt)
        Es = embed_layers(model, ss)[4]; Et = embed_layers(model, st)[4]
        m, a = nc_auc(Es, ys, Et, yt)
        print(f"  {tag:24s} macroF1={m['macro_f1']:.4f} FPR={m['fpr']:.4f} AUC={a:.4f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(ROOT, "data", "layers_orig.npz"), **{f"sup_{i}": Ls[i] for i in range(5)},
             **{f"tst_{i}": Lt[i] for i in range(5)}, ys=ys, yt=yt)
    print("done", round(time.time()-t0), flush=True)

if __name__ == "__main__":
    main()
