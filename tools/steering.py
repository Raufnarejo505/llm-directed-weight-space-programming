import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Task D: compute gradient-free steering vectors from the support set (no weight change)."""
import json, sys, time, numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok
torch.set_num_threads(8)

FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
LAYER = 3
TENSOR = "base_transformer.encoder.layer.3.burst_encoder.mlp.Wo.weight"

def load_split():
    rows = [json.loads(l) for l in open(FLOWS)]
    ben = sorted([r for r in rows if r["capture"] == "ctu-idseval-6-benign-user-traffic-1"], key=lambda r: r["t0"])
    scn = sorted([r for r in rows if "portscan" in r["capture"]], key=lambda r: (r["capture"], r["t0"]))
    return ben[:len(ben)//2], scn

def bucket_key(r): return (len(r["tokens"]), max(len(b) for b in r["tokens"]))

@torch.no_grad()
def collect(model, flows, bs=16, max_flows=None):
    """accumulate token means of the layer-3 MLP hidden (phi) and final hidden (h)"""
    if max_flows: flows = flows[:max_flows]
    order = sorted(range(len(flows)), key=lambda i: bucket_key(flows[i]))
    phi_sum = np.zeros(768, np.float64); h_sum = np.zeros(512, np.float64); n = 0
    phi_sq = np.zeros(768, np.float64); h_sq = np.zeros(512, np.float64)
    cap = {}
    def hook(mod, inp):
        cap["phi"] = inp[0].detach()
    m = model.base_transformer.encoder.layer[LAYER].burst_encoder.mlp.Wo
    hd = m.register_forward_pre_hook(hook)
    i = 0
    t0 = time.time()
    while i < len(order):
        key = bucket_key(flows[order[i]]); j = i
        while j < len(order) and bucket_key(flows[order[j]]) == key and j - i < bs: j += 1
        idx = order[i:j]
        b = tok.collate([flows[k] for k in idx]); B, L = b["input_ids"].shape
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        o = model.base_transformer(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=pos,
                                   direction=b["direction"], iats=b["iats"], bytes=b["bytes"], pkt_count=b["pkt_count"],
                                   protocol=b["protocol"], dataset_burst_sizes=b["dataset_burst_sizes"], return_dict=True)
        h = o.last_hidden_state
        mask = b["attention_mask"].bool()
        phi = cap["phi"].reshape(B, -1, 768)   # [B*nb, mbl, 768] -> [B, L, 768]
        assert phi.shape[1] == L, (phi.shape, L)
        phi_m = phi[mask].numpy().astype(np.float64)
        h_m = h[mask].numpy().astype(np.float64)
        phi_sum += phi_m.sum(0); h_sum += h_m.sum(0)
        phi_sq += (phi_m**2).sum(0); h_sq += (h_m**2).sum(0)
        n += phi_m.shape[0]
        i = j
        if i % 128 < bs:
            print(f"    {i}/{len(order)} flows, {n} tokens, {time.time()-t0:.0f}s", flush=True)
    hd.remove()
    return dict(n=n, phi_mean=phi_sum/n, h_mean=h_sum/n,
                phi_std=np.sqrt(np.maximum(phi_sq/n - (phi_sum/n)**2, 0)),
                h_std=np.sqrt(np.maximum(h_sq/n - (h_sum/n)**2, 0)))

def main():
    t0 = time.time()
    ben_s, scn_s = load_split()
    print(f"benign support {len(ben_s)}, scan support {len(scn_s)}", flush=True)
    model = nf.load_model()
    print("collecting benign...", flush=True)
    B = collect(model, ben_s)
    print("collecting scan...", flush=True)
    S = collect(model, scn_s, max_flows=600)
    print(f"tokens: benign {B['n']}, scan {S['n']}", flush=True)
    dphi = S["phi_mean"] - B["phi_mean"]; dh = S["h_mean"] - B["h_mean"]
    v = dphi / (np.linalg.norm(dphi) + 1e-12)
    u = dh / (np.linalg.norm(dh) + 1e-12)
    print(f"||dphi||={np.linalg.norm(dphi):.4f}  ||dh||={np.linalg.norm(dh):.4f}", flush=True)
    # separability of the scalar projection c = v.phi
    print(f"phi proj: scan mean {np.dot(v, S['phi_mean']):.3f} benign mean {np.dot(v, B['phi_mean']):.3f}", flush=True)
    # scale: choose alpha so mean added shift is kappa * ||dh||
    e_c = 0.5 * (np.dot(v, S["phi_mean"]) + np.dot(v, B["phi_mean"]))
    e_c = abs(e_c) + 1e-9
    alphas = {k: (k / e_c) for k in (0.25, 0.5, 1.0, 2.0)}
    print("e|c| =", e_c, "alpha grid:", {k: round(v, 4) for k, v in alphas.items()}, flush=True)
    np.savez(os.path.join(ROOT, "data", "steering.npz"), u=u, v=v, dphi=dphi, dh=dh,
             phi_mean_scan=S["phi_mean"], phi_mean_ben=S["phi_mean"], h_mean_scan=S["h_mean"],
             h_mean_ben=B["h_mean"], phi_std_scan=S["phi_std"], phi_std_ben=B["phi_std"],
             h_std_scan=S["h_std"], h_std_ben=B["h_std"], n_ben=B["n"], n_scan=S["n"],
             alpha_025=alphas[0.25], alpha_05=alphas[0.5], alpha_1=alphas[1.0], alpha_2=alphas[2.0])
    print("saved steering statistics; elapsed", round(time.time()-t0), "s", flush=True)

if __name__ == "__main__":
    main()
