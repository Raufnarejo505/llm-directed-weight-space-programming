PY ?= python

.PHONY: help install data baseline patches eval test report clean

help:
	@echo "install   install python dependencies"
	@echo "data      convert pcaps to netFound flow records"
	@echo "baseline  embeddings + original metrics"
	@echo "patches   regenerate the patch JSONs from support-set statistics"
	@echo "eval      apply and evaluate all six patches"
	@echo "test      run the unit tests"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) tools/build_dataset.py "data/ctu/pcap/pcap/*.pcap" data/flows_tcp.jsonl

baseline:
	$(PY) tools/embed_all.py
	$(PY) tools/make_original_result.py

patches:
	$(PY) tools/steering.py
	$(PY) tools/make_patches.py

eval:
	@for p in p1a-k1-rank1-wow p1a-k2-rank1-wow p1a-k5-rank1-wow p1a-k10-rank1-wow p2-single-unit p3-control-pooling; do \
		$(PY) tools/eval_patch.py $$p || exit 1; \
	done
	$(PY) tools/compare.py
	$(PY) tools/geometry_bootstrap.py

test:
	$(PY) -m pytest

clean:
	@rm -rf checkpoints/patched data/emb_*.npz results/emb_*.npz
