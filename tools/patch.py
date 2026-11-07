import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

"""Reversible, validated weight-patch engine for netFound-small.

A patch is a JSON document describing edits to named tensors. Applying a patch
never writes to the base checkpoint: it materialises a NEW checkpoint directory
(patched/<patch_id>/) from the base, applies only the listed edits, and records a
full per-tensor manifest so the edit can be audited and reverted.

OPS
  set     : overwrite a slice with given values
  add     : add given values to a slice
  scale   : multiply a slice by a scalar
  rank1   : A[slice] += scale * u (x) v      (outer product, low-rank update)
  rows    : convenience: full-row update on the 1st axis (index list)
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
from safetensors.numpy import load_file, save_file

BASE = os.path.join(ROOT, "checkpoints", "original")
PATCH_DIR = os.path.join(ROOT, "patches")
PATCHED = os.path.join(ROOT, "checkpoints", "patched")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_patch(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _slice_index(shape, sl):
    """sl: list of [start,end] or {"index": [..]} entries, one per axis (None/omitted = whole axis)."""
    idx = []
    if sl is None:
        return tuple(slice(None) for _ in shape)
    for ax, s in enumerate(sl):
        if s is None:
            idx.append(slice(None))
        elif isinstance(s, dict) and "index" in s:
            idx.append(np.asarray(s["index"], dtype=np.int64))
        else:
            idx.append(slice(int(s[0]), int(s[1])))
    return tuple(idx)


def _as_array(op, ref_shape):
    if "npz" in op:
        z = np.load(os.path.join(PATCH_DIR, op["npz"]))
        return z["value"]
    return np.asarray(op["value"], dtype=np.float32)


def dry_run(patch, base_tensors=None):
    """Validate structure/shape/dtype of a patch without writing anything."""
    base_tensors = base_tensors or load_file(os.path.join(BASE, "model.safetensors"))
    report = {"patch_id": patch.get("patch_id"), "errors": [], "warnings": [], "ops": []}
    touched = {}
    for i, op in enumerate(patch["operations"]):
        name = op["tensor"]
        if name not in base_tensors:
            report["errors"].append(f"op{i}: unknown tensor {name!r}")
            continue
        W = base_tensors[name]
        sl = op.get("slice")
        try:
            tgt_shape = W[_slice_index(W.shape, sl)].shape
        except Exception as e:
            report["errors"].append(f"op{i}: bad slice on {name}: {e}")
            continue
        kind = op["op"]
        if kind in ("set", "add"):
            v = _as_array(op, tgt_shape)
            if v.shape != tgt_shape:
                report["errors"].append(f"op{i}: value shape {v.shape} != target {tgt_shape} on {name}")
                continue
            if not np.isfinite(v).all():
                report["errors"].append(f"op{i}: non-finite values on {name}")
                continue
        elif kind == "rank1":
            u = np.asarray(op["u"], dtype=np.float32)
            v = np.asarray(op["v"], dtype=np.float32)
            if tgt_shape is not None and (u.shape[0] != tgt_shape[0] or v.shape[0] != tgt_shape[-1]):
                report["errors"].append(f"op{i}: rank1 u{u.shape} v{v.shape} incompatible with {tgt_shape}")
                continue
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                report["errors"].append(f"op{i}: non-finite rank1 factors")
                continue
        elif kind in ("scale", "rows"):
            pass
        else:
            report["errors"].append(f"op{i}: unknown op {kind!r}")
            continue
        n = int(np.prod(tgt_shape)) if tgt_shape else 0
        touched[name] = touched.get(name, 0) + n
        report["ops"].append(dict(op=i, kind=kind, tensor=name, slice=str(sl), n=n, target=str(tgt_shape)))

    total = sum(int(np.prod(t.shape)) for t in base_tensors.values())
    report["n_tensors_touched"] = len(touched)
    report["n_params_touched"] = int(sum(touched.values()))
    report["frac_params_touched"] = report["n_params_touched"] / total
    report["total_params"] = total
    return report


def apply_patch(patch_path, out_root=PATCHED, overwrite=False):
    patch = load_patch(patch_path)
    pid = patch["patch_id"]
    out_dir = os.path.join(out_root, pid)
    base_dir = patch.get("base_checkpoint", BASE)
    tensors = load_file(os.path.join(base_dir, "model.safetensors"))
    rep = dry_run(patch, base_tensors=tensors)
    if rep["errors"]:
        raise ValueError("patch failed validation: " + "; ".join(rep["errors"]))
    if os.path.isdir(out_dir) and not overwrite:
        raise FileExistsError(out_dir + " exists (use overwrite=True)")

    before = {k: v.copy() for k, v in tensors.items()}

    for op in patch["operations"]:
        name, kind = op["tensor"], op["op"]
        sl = _slice_index(tensors[name].shape, op.get("slice"))
        if kind == "set":
            tensors[name][sl] = _as_array(op, tensors[name][sl].shape)
        elif kind == "add":
            tensors[name][sl] += _as_array(op, tensors[name][sl].shape)
        elif kind == "scale":
            tensors[name][sl] *= np.float32(op["factor"])
        elif kind == "rank1":
            u = np.asarray(op["u"], dtype=np.float32)
            v = np.asarray(op["v"], dtype=np.float32)
            tensors[name][sl] += np.float32(op.get("scale", 1.0)) * np.outer(u, v)
        elif kind == "rows":
            idx = np.asarray(op["index"], dtype=np.int64)
            v = _as_array(op, tensors[name][idx].shape)
            tensors[name][idx] = v

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    for f in ("config.json", "README.md"):
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, f))

    # audit manifest: per-tensor delta stats
    manifest = {
        "patch_id": pid, "applied_at": datetime.now(timezone.utc).isoformat(),
        "patch_file": os.path.abspath(patch_path),
        "base_checkpoint": os.path.abspath(base_dir),
        "base_sha256": sha256_file(os.path.join(base_dir, "model.safetensors")),
        "patched_sha256": sha256_file(os.path.join(out_dir, "model.safetensors")),
        "tensors": {}, "summary": dict(rep, ops=rep["ops"]),
    }
    tot_delta_sq = 0.0
    tot_base_sq = 0.0
    n_changed = 0
    for k in tensors:
        d = tensors[k].astype(np.float64) - before[k].astype(np.float64)
        nz = int((d != 0).sum())
        b = before[k].astype(np.float64)
        tot_delta_sq += float((d ** 2).sum())
        tot_base_sq += float((b ** 2).sum())
        n_changed += nz
        if nz:
            manifest["tensors"][k] = {
                "n_changed": nz,
                "frac_changed_of_tensor": nz / d.size,
                "delta_l2": float(np.sqrt((d ** 2).sum())),
                "base_l2": float(np.sqrt((b ** 2).sum())),
                "rel_l2": float(np.sqrt((d ** 2).sum()) / max(np.sqrt((b ** 2).sum()), 1e-12)),
                "delta_max_abs": float(np.abs(d).max()),
                "finite_after": bool(np.isfinite(tensors[k]).all()),
            }
    manifest["summary"].update(
        n_params_changed=n_changed,
        frac_params_changed=n_changed / rep["total_params"],
        global_rel_l2=float(np.sqrt(tot_delta_sq) / np.sqrt(tot_base_sq)),
        all_finite=bool(all(np.isfinite(v).all() for v in tensors.values())),
    )
    with open(os.path.join(out_dir, "patch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "applied_patch.json"), "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2)
    return out_dir, manifest


def revert(patch_id, out_root=PATCHED):
    """Reverting = deleting the materialised copy; the base checkpoint is never modified."""
    d = os.path.join(out_root, patch_id)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False
