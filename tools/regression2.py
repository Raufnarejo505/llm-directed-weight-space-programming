import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Regression suite (memory-safe): pretraining heads on a fixed deterministic batch."""
import json, sys, numpy as np, torch
sys.path.insert(0, os.path.join(ROOT, "tools"))
import nfcommon as nf, nf_tokenize as tok

FLOWS = os.path.join(ROOT, "data", "flows_tcp.jsonl")
MASK = 65538


def build_suite(n_ben=24, n_scan=64, n_mal=32, seed=1234):
    rows = [json.loads(l) for l in open(FLOWS)]
    ben = [r for r in rows if r["capture"] == "ctu-idseval-6-benign-user-traffic-1"]
    scn = [r for r in rows if "portscan" in r["capture"]]
    mal = [r for r in rows if "malware" in r["capture"]]
    rng = np.random.default_rng(seed)
    pick = lambda xs, n: [xs[i] for i in sorted(rng.choice(len(xs), min(n, len(xs)), replace=False))]
    return pick(ben, n_ben) + pick(scn, n_scan) + pick(mal, n_mal)


@torch.no_grad()
def run(model, flows, seed=7, bs=16, chunk=256):
    torch.manual_seed(seed); np.random.seed(seed)
    mlm_loss_sum, correct, nmask = 0.0, 0, 0
    dir_correct, dir_total = 0, 0
    meta_abs, meta_n = 0.0, 0
    swap_loss_sum, swap_n = 0.0, 0
    ce = torch.nn.functional.cross_entropy
    order = sorted(range(len(flows)), key=lambda i: (len(flows[i]["tokens"]), max(len(b) for b in flows[i]["tokens"])))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        b = tok.collate([flows[k] for k in idx])
        B, L = b["input_ids"].shape
        nb = b["dataset_burst_sizes"].shape[1]
        ids = b["input_ids"].clone()
        special = (ids == 0) | (ids == 65537) | (ids == 65538)
        prob = torch.full(ids.shape, 0.15); prob[special] = 0.0
        masked = torch.bernoulli(prob).bool()
        labels = ids.clone(); labels[~masked] = -100
        ids[masked] = MASK
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        h = model.base_transformer(input_ids=ids, attention_mask=b["attention_mask"], position_ids=pos,
                                   direction=b["direction"], iats=b["iats"], bytes=b["bytes"], pkt_count=b["pkt_count"],
                                   protocol=b["protocol"], dataset_burst_sizes=b["dataset_burst_sizes"], return_dict=True).last_hidden_state
        # MLM on masked rows only
        flat_h = h.reshape(-1, h.shape[-1]); flat_lab = labels.reshape(-1); flat_m = masked.reshape(-1)
        sel = torch.nonzero(flat_m, as_tuple=False).flatten()
        for j in range(0, len(sel), chunk):
            rows = sel[j:j + chunk]
            logits = model.lm_head(flat_h[rows])
            lab = flat_lab[rows]
            mlm_loss_sum += float(ce(logits, lab, reduction="sum")); nmask += len(rows)
            correct += int((logits.argmax(-1) == lab).sum())
        reps = h[:, ::b["max_burst_length"], :]
        dlog = model.dirPred(reps)
        dtrue = b["direction"][:, ::b["max_burst_length"]].long()
        valid = (b["dataset_burst_sizes"] > 0)
        dir_correct += int(((dlog.argmax(-1) == (dtrue > 0).long()) & valid).sum()); dir_total += int(valid.sum())
        mlog = model.linearMetadataPred(reps)
        tgt = torch.stack([b["iats"][:, ::b["max_burst_length"]], b["bytes"][:, ::b["max_burst_length"]] / 1000,
                           b["pkt_count"][:, ::b["max_burst_length"]]], dim=2)
        meta_abs += float((mlog - tgt).abs()[valid].sum()); meta_n += int(valid.sum()) * 3
        # swap head
        from modules.netFoundPoolings import poolingByAttention
        pooled = poolingByAttention(model.attentivePooling, h, b["max_burst_length"])
        slog = model.swappedClassifierHiddenLayer(pooled)
        swap_loss_sum += float(ce(slog, torch.zeros(B, dtype=torch.long), reduction="sum")); swap_n += B
    return dict(mlm_loss=mlm_loss_sum / max(nmask, 1), mlm_top1=correct / max(nmask, 1),
                dir_acc=dir_correct / max(dir_total, 1), meta_mae=meta_abs / max(meta_n, 1),
                swap_loss=swap_loss_sum / max(swap_n, 1), n_masked=nmask, n_bursts=dir_total)


def main(ckpt, out):
    model = nf.load_model(ckpt) if ckpt else nf.load_model()
    flows = build_suite()
    r = run(model, flows)
    r["checkpoint"] = ckpt or "original"
    json.dump(r, open(out, "w"), indent=2)
    print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "orig" else None,
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "regression_orig.json"))
