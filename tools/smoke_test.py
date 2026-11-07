import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import sys, os
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np, torch
import nfcommon as nf

torch.manual_seed(0); np.random.seed(0)
model = nf.load_model()
n = sum(p.numel() for p in model.parameters())
print("PARAMS:", f"{n:,}")
print("DTYPE:", next(model.parameters()).dtype)
w = model.base_transformer.embeddings.word_embeddings.weight
print("decoder tied to embeddings:", model.lm_head.decoder.weight.data_ptr() == w.data_ptr())
pes = [l.position_embeddings.weight.data_ptr() for l in model.base_transformer.encoder.layer]
print("position_embeddings shared:", len(set(pes)) == 1)

rng = np.random.default_rng(0)
flows = []
for i in range(4):
    nb = rng.integers(2, 6)
    bursts = []
    dirs, byts, cnts, ias = [], [], [], []
    for j in range(nb):
        nb_pk = int(rng.integers(1, 4))
        toks = list(rng.integers(0, 65536, size=nb_pk * 12))
        bursts.append(toks)
        dirs.append(1); byts.append(float(rng.integers(40, 1500))); cnts.append(float(nb_pk)); ias.append(float(rng.integers(0, 500)))
    flows.append(dict(tokens=bursts, directions=dirs, bytes=byts, counts=cnts, iats=ias, protocol=6))

batch = nf.build_batch(flows)
print("input_ids:", tuple(batch["input_ids"].shape), "burst_len:", batch["max_burst_length"])
out = nf.encode(model, batch)
print("last_hidden_state:", tuple(out.last_hidden_state.shape))
print("hidden mean/std:", float(out.last_hidden_state.mean()), float(out.last_hidden_state.std()))

# full pretraining-head forward with labels to check loss plumbing
labels = batch["input_ids"].clone()
labels[:, :] = -100
labels[:, 5] = batch["input_ids"][:, 5]
swapped = torch.zeros(4, dtype=torch.long)
meta_mask = torch.zeros((4, batch["dataset_burst_sizes"].shape[1]), dtype=torch.float32)
meta_mask[:, 0] = 1.0
res = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            labels=labels, direction=batch["direction"], iats=batch["iats"], bytes=batch["bytes"],
            pkt_count=batch["pkt_count"], protocol=batch["protocol"],
            swappedLabels=swapped, burstMetasToBeMasked=meta_mask.bool(),
            dataset_burst_sizes=batch["dataset_burst_sizes"], return_dict=True)
print("pretrain loss:", float(res.loss))
print("logits:", tuple(res.logits[0].shape), tuple(res.logits[1].shape))
