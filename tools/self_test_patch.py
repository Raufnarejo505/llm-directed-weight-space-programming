import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WSP_ROOT", os.path.dirname(_HERE))

import json, sys
sys.path.insert(0, os.path.join(ROOT, "tools"))
import patch as P

p = {
  "patch_id": "p000-identity-selftest",
  "created": "2026-11-06",
  "target_capability": "NONE (plumbing self-test; zero net change)",
  "hypothesis": "Verify that the patch engine validates, materialises, audits and can be reverted without altering any weight value.",
  "base_checkpoint": P.BASE,
  "operations": [
    {"op": "scale", "tensor": "dirPred.weight", "factor": 1.0},
    {"op": "add", "tensor": "dirPred.bias", "value": [0.0, 0.0]}
  ]
}
with open(os.path.join(ROOT, "patches", "p000-identity-selftest.json"), "w") as f:
    json.dump(p, f, indent=2)

rep = P.dry_run(p)
print("DRY RUN errors:", rep["errors"], "warnings:", rep["warnings"])
print("tensors touched:", rep["n_tensors_touched"], "params touched:", rep["n_params_touched"],
      f"({100*rep['frac_params_touched']:.6f}% of {rep['total_params']:,})")

d, man = P.apply_patch(os.path.join(ROOT, "patches", "p000-identity-selftest.json"))
print("applied to:", d)
print("n_params_changed:", man["summary"]["n_params_changed"], "global_rel_l2:", man["summary"]["global_rel_l2"])
print("base sha:", man["base_sha256"][:16], "...")
print("patched sha:", man["patched_sha256"][:16], "...")
print("identical bytes:", man["base_sha256"] == man["patched_sha256"])
P.revert("p000-identity-selftest")
print("reverted:", not __import__("os").path.isdir(d))

# negative tests: shape mismatch and unknown tensor must be rejected
bad = {"patch_id": "bad", "operations": [
    {"op": "set", "tensor": "dirPred.bias", "value": [0.0, 0.0, 0.0]},
    {"op": "add", "tensor": "no.such.tensor", "value": [1.0]}]}
r2 = P.dry_run(bad)
print("negative-test errors:")
for e in r2["errors"]: print("  -", e)
