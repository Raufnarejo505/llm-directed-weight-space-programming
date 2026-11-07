"""Invariants over the committed result tables.

These guard the claims in the README: identical decisions across all conditions,
a monotone geometry dose-response, and an exact negative control.
"""
import csv
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(REPO, "results", "tables")


def read(name):
    with open(os.path.join(TABLES, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_metrics_all_conditions_present():
    rows = read("metrics.csv")
    names = {r["condition"] for r in rows}
    assert names == {"original", "p1a-k1-rank1-wow", "p1a-k2-rank1-wow", "p1a-k5-rank1-wow",
                     "p1a-k10-rank1-wow", "p2-single-unit", "p3-control-pooling"}


def test_confusion_matrix_sums():
    for r in read("metrics.csv"):
        total = sum(int(r[k]) for k in ("tp", "fp", "tn", "fn"))
        assert total == 2382, (r["condition"], total)


def test_no_decision_changes_anywhere():
    rows = read("metrics.csv")
    base = rows[0]
    for r in rows[1:]:
        for k in ("macro_f1", "fpr", "tp", "fp", "tn", "fn"):
            assert r[k] == base[k], (r["condition"], k)


def test_geometry_dose_response_is_monotone():
    g = {r["condition"]: r for r in read("geometry.csv")}
    ladder = ["original", "p1a-k1-rank1-wow", "p1a-k2-rank1-wow", "p1a-k5-rank1-wow", "p1a-k10-rank1-wow"]
    vals = [float(g[c]["between_support"]) for c in ladder]
    assert vals == sorted(vals) and len(set(vals)) == len(vals)
    assert vals[-1] > 1.9 * vals[0]


def test_negative_control_is_exact():
    g = {r["condition"]: r for r in read("geometry.csv")}
    assert float(g["p3-control-pooling"]["drift_rel"]) < 1e-6
    assert g["p3-control-pooling"]["between_support"] == g["original"]["between_support"]


def test_edit_budget_bounded():
    for r in read("edits.csv"):
        assert float(r["pct_of_model"]) < 1.0
