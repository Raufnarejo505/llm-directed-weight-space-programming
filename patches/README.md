# Patches

Each JSON is a self-contained, reversible edit applied to a *copy* of the base checkpoint.
The base checkpoint (`checkpoints/original/model.safetensors`, sha256 9e0080bb...) is never
written.

| file | edit |
|---|---|
| `p1a-k1-rank1-wow.json` | rank-1 update of `encoder.layer.3.burst_encoder.mlp.Wo.weight`, kappa=1 |
| `p1a-k2-rank1-wow.json` | same, kappa=2 |
| `p1a-k5-rank1-wow.json` | same, kappa=5 |
| `p1a-k10-rank1-wow.json` | same, kappa=10 |
| `p2-single-unit.json` | one column (unit 656) of the same matrix |
| `p3-control-pooling.json` | negative control: scale `attentivePooling.lin_proj.weight` |
| `p000-identity-selftest.json` | harness self-test: scale by 1.0, changes nothing |

## Format

    {
      "patch_id": "p1a-k5-rank1-wow",
      "target_capability": "...",
      "hypothesis": "...",
      "base_checkpoint": "checkpoints/original",
      "operations": [
        {"op": "rank1", "tensor": "base_transformer...mlp.Wo.weight",
         "u": [...512...], "v": [...768...], "scale": 0.5491}
      ]
    }

Supported operations: `set`, `add`, `scale`, `rank1`, `rows`. `tools/patch.py:dry_run`
checks tensor names, shapes, dtypes, slice dimensions and finiteness before anything is
written; `apply_patch` materialises `checkpoints/patched/<patch_id>/` with a manifest of
per-tensor delta L2, relative L2 and changed-value counts.

Reverting a patch means deleting that directory; the base checkpoint was never touched.
