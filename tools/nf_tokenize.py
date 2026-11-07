import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""netFound input construction from flow records (mirrors netFoundTokenizer semantics)."""
import numpy as np, torch

BOS, PAD, MASK = 65537, 0, 65538


def collate(flows):
    """flows: list of dicts with tokens/directions/bytes/counts/iats/protocol.
    Mirrors SimpleDataCollator: batch_max_burst_length = max burst len + 1."""
    max_burst_len = max(max(len(b) for b in f["tokens"]) for f in flows) + 1
    max_bursts = max(len(f["tokens"]) for f in flows)
    B, L = len(flows), max_bursts * max_burst_len
    ids = np.zeros((B, L), np.int64); att = np.zeros((B, L), np.int64)
    dr = np.zeros((B, L), np.float32); ia = np.zeros((B, L), np.float32)
    by = np.zeros((B, L), np.float32); ct = np.zeros((B, L), np.float32)
    sz = np.zeros((B, max_bursts), np.int64); pr = np.zeros(B, np.int64)
    for i, f in enumerate(flows):
        pr[i] = f["protocol"]
        for j, burst in enumerate(f["tokens"]):
            off = j * max_burst_len
            ids_row = [BOS] + [int(t) + 1 for t in burst]
            n = len(ids_row)
            ids[i, off:off+n] = ids_row
            att[i, off:off+n] = 1
            dr[i, off:off+n] = f["directions"][j]
            by[i, off:off+n] = f["bytes"][j]
            ct[i, off:off+n] = f["counts"][j]
            ia[i, off:off+n] = f["iats"][j]
            sz[i, j] = len(burst)
    return dict(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(att),
                direction=torch.from_numpy(dr), iats=torch.from_numpy(ia),
                bytes=torch.from_numpy(by), pkt_count=torch.from_numpy(ct),
                protocol=torch.from_numpy(pr), dataset_burst_sizes=torch.from_numpy(sz),
                max_burst_length=max_burst_len)


@torch.no_grad()
def embed(model, flows, batch_size=64, pool="masked_mean"):
    """Return [N,512] flow embeddings + optional per-layer pooled representations."""
    outs = []
    for i in range(0, len(flows), batch_size):
        b = collate(flows[i:i+batch_size])
        B, L = b["input_ids"].shape
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        out = model.base_transformer(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=pos,
            direction=b["direction"], iats=b["iats"], bytes=b["bytes"], pkt_count=b["pkt_count"],
            protocol=b["protocol"], dataset_burst_sizes=b["dataset_burst_sizes"], return_dict=True)
        h = out.last_hidden_state
        if pool == "masked_mean":
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            v = h.mean(1)
        outs.append(v)
    return torch.cat(outs, 0).numpy()
