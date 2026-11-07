"""Unit tests for the reversible patch engine (no model, no checkpoint required)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import patch as P  # noqa: E402


@pytest.fixture()
def base(tmp_path):
    tensors = {
        "a.weight": np.round(np.random.RandomState(0).randn(4, 3).astype(np.float32) * 0.1, 6),
        "b.bias": np.zeros(2, dtype=np.float32),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    (tmp_path / "config.json").write_text("{}")
    return tensors, tmp_path


def patch_doc(base_dir, ops, pid="p-test"):
    return {"patch_id": pid, "target_capability": "unit test", "hypothesis": "n/a",
            "base_checkpoint": str(base_dir), "operations": ops}


def test_unknown_tensor_is_rejected(base):
    tensors, d = base
    rep = P.dry_run(patch_doc(d, [{"op": "set", "tensor": "nope", "value": [0.0]}]), tensors)
    assert rep["errors"] and "unknown tensor" in rep["errors"][0]


def test_shape_mismatch_is_rejected(base):
    tensors, d = base
    rep = P.dry_run(patch_doc(d, [{"op": "add", "tensor": "b.bias", "value": [0.0, 0.0, 0.0]}]), tensors)
    assert rep["errors"] and "value shape" in rep["errors"][0]


def test_non_finite_is_rejected(base):
    tensors, d = base
    rep = P.dry_run(patch_doc(d, [{"op": "set", "tensor": "b.bias", "value": [float("nan"), 0.0]}]), tensors)
    assert rep["errors"] and "non-finite" in rep["errors"][0]


def test_rank1_dimension_check(base):
    tensors, d = base
    rep = P.dry_run(patch_doc(d, [{"op": "rank1", "tensor": "a.weight", "u": [1.0], "v": [1.0, 0.0, 0.0]}]), tensors)
    assert rep["errors"]


def test_valid_rank1_counts(base):
    tensors, d = base
    rep = P.dry_run(patch_doc(d, [{"op": "rank1", "tensor": "a.weight",
                                   "u": [1.0, 0, 0, 0], "v": [1.0, 0, 0], "scale": 0.5}]), tensors)
    assert rep["errors"] == []
    assert rep["n_params_touched"] == 12
    assert rep["n_tensors_touched"] == 1


def test_apply_then_revert_leaves_base_untouched(base, tmp_path):
    tensors, d = base
    ops = [{"op": "add", "tensor": "b.bias", "value": [0.25, -0.25]}]
    p = patch_doc(d, ops, pid="p-add")
    ppath = tmp_path / "patch.json"
    ppath.write_text(json.dumps(p))

    base_hash = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    out, man = P.apply_patch(str(ppath), out_root=str(tmp_path / "out"))
    assert man["summary"]["n_params_changed"] == 2
    assert man["base_sha256"] == base_hash
    written = load_file(os.path.join(out, "model.safetensors"))
    np.testing.assert_allclose(written["b.bias"], tensors["b.bias"] + np.array([0.25, -0.25], np.float32))
    # base checkpoint is bit-identical after applying
    assert hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest() == base_hash
    assert P.revert("p-add", out_root=str(tmp_path / "out")) is True


def test_identity_patch_is_byte_identical(base, tmp_path):
    tensors, d = base
    ops = [{"op": "scale", "tensor": "a.weight", "factor": 1.0}]
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(patch_doc(d, ops, pid="p-noop")))
    out, man = P.apply_patch(str(ppath), out_root=str(tmp_path / "out"))
    assert man["summary"]["n_params_changed"] == 0
    assert man["base_sha256"] == man["patched_sha256"]
