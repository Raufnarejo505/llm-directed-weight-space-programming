# Scripts

These are the exact scripts used for the study, with absolute paths replaced by
`$WSP_ROOT` (default: the repository root). They are research code, not a library.

| script | purpose |
|---|---|
| `nfcommon.py` | checkpoint loading, batch construction for the netFound architecture |
| `nf_tokenize.py` | flow records -> model tensors, masked-mean embedding |
| `patch.py` | patch validation (`dry_run`), application, per-tensor delta manifest |
| `build_dataset.py` | pcap -> netFound flow records (port of upstream preprocessing) |
| `embed_all.py` | baseline embeddings + nearest-centroid metrics |
| `make_original_result.py` | writes results/original.json from the baseline cache |
| `steering.py` | support-set activation means -> u, v steering vectors |
| `make_patches.py` | emits the six patch JSONs |
| `eval_patch.py` | apply one patch, re-embed, metrics + geometry + bootstrap |
| `compare.py` | aggregation, paired bootstrap, flip counts |
| `geometry_bootstrap.py` | bootstrap CIs for between-class distance / separation ratio |
| `mechanism.py` | per-group displacement and decision-margin analysis |
| `regression2.py`, `run_regressions.py` | fixed pretraining-head regression suite |
| `self_test_patch.py` | patch-engine self-test (identity patch, zero net change) |

`compare.py` and `mechanism.py` need the embedding caches written by `eval_patch.py`
(`results/emb_*.npz`), which are not committed.

Path resolution: every script defines

    ROOT = os.environ.get("WSP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

so running from a checkout works out of the box.
