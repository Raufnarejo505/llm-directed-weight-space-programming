import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Shared utilities for loading and running the netFound-small checkpoint."""
import os, sys, json, contextlib
import numpy as np
import torch

REF = os.path.join(ROOT, "ref", "netFound-main")
SRC = os.path.join(REF, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CKPT_ORIG = os.path.join(ROOT, "checkpoints", "original")


class _NullWriter:
    def add_scalar(self, *a, **k):
        pass


def import_model_modules():
    import modules.utils as utils
    utils.TB_WRITER = _NullWriter()
    utils.GLOBAL_STEP = 0
    from modules.netFoundConfigBase import netFoundConfig
    from modules.netFoundModels import netFoundLanguageModelling, netFoundFinetuningModel
    return utils, netFoundConfig, netFoundLanguageModelling, netFoundFinetuningModel


def load_config(ckpt_dir=CKPT_ORIG, **overrides):
    _, netFoundConfig, _, _ = import_model_modules()
    cfg = netFoundConfig.from_pretrained(ckpt_dir)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def load_model(ckpt_dir=CKPT_ORIG, dtype=torch.float32, **cfg_overrides):
    _, _, netFoundLanguageModelling, _ = import_model_modules()
    cfg = load_config(ckpt_dir, **cfg_overrides)
    model = netFoundLanguageModelling.from_pretrained(ckpt_dir, config=cfg)
    model.to(dtype=dtype)
    model.eval()
    return model


def build_batch(flows, pad_token_id=0, meta_dtype=torch.float32):
    """flows: list of dicts with keys
         tokens: list[list[int]] (per burst, no CLS), directions/bytes/counts/iats: list[float] per burst,
         protocol: int
       Returns collated tensors matching SimpleDataCollator semantics.
    """
    max_burst_length = max(max(len(b) for b in f["tokens"]) for f in flows) + 1
    max_bursts = max(len(f["tokens"]) for f in flows)
    B = len(flows)
    L = max_bursts * max_burst_length
    input_ids = np.zeros((B, L), dtype=np.int64)
    attn = np.zeros((B, L), dtype=np.int64)
    direction = np.zeros((B, L), dtype=np.float32)
    iats = np.zeros((B, L), dtype=np.float32)
    bytes_ = np.zeros((B, L), dtype=np.float32)
    counts = np.zeros((B, L), dtype=np.float32)
    sizes = np.zeros((B, max_bursts), dtype=np.int64)
    proto = np.zeros((B,), dtype=np.int64)
    for i, f in enumerate(flows):
        proto[i] = f["protocol"]
        for j, burst in enumerate(f["tokens"]):
            off = j * max_burst_length
            ids = [65537] + [int(t) + 1 for t in burst]  # CLS + add_one
            n = len(ids)
            input_ids[i, off:off + n] = ids
            attn[i, off:off + n] = 1
            direction[i, off:off + n] = f["directions"][j]
            iats[i, off:off + n] = f["iats"][j]
            bytes_[i, off:off + n] = f["bytes"][j]
            counts[i, off:off + n] = f["counts"][j]
            sizes[i, j] = len(burst)
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attn),
        "direction": torch.from_numpy(direction).to(meta_dtype),
        "iats": torch.from_numpy(iats).to(meta_dtype),
        "bytes": torch.from_numpy(bytes_).to(meta_dtype),
        "pkt_count": torch.from_numpy(counts).to(meta_dtype),
        "protocol": torch.from_numpy(proto),
        "dataset_burst_sizes": torch.from_numpy(sizes),
        "total_bursts": torch.from_numpy(sizes.shape[0] * 0 + np.array([len(f["tokens"]) for f in flows], dtype=np.int64)),
        "max_burst_length": max_burst_length,
    }


@torch.no_grad()
def encode(model, batch, return_hidden=False):
    B, L = batch["input_ids"].shape
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    out = model.base_transformer(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=position_ids,
        direction=batch["direction"],
        iats=batch["iats"],
        bytes=batch["bytes"],
        pkt_count=batch["pkt_count"],
        protocol=batch["protocol"],
        dataset_burst_sizes=batch["dataset_burst_sizes"],
        return_dict=True,
        output_hidden_states=return_hidden,
    )
    return out
