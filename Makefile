# Phase 1 pipeline. Every target is resumable; re-running a stage is safe.
#
# Ordering constraints that the targets below encode:
#   * lid-train must precede any `lid` run (both languages share one classifier).
#   * `cross` needs BOTH languages already deduped, and must precede `split`,
#     so no document that later gets dropped can influence a split assignment.
#   * `sweep` needs splits; `tokenizer` needs the vocab_size chosen from the
#     sweep; `encode` needs the tokenizer. That chain is why finish-* is split
#     from phase1-*.
PY := python -m lmacorpus.cli
WORKERS ?=

.PHONY: test smoke probe discover crawl lid-train phase1-hi phase1-ne cross all

test:            ## unit tests
	python -m pytest tests -q

smoke:           ## end-to-end pipeline check on synthetic data
	python scripts/smoke_test.py

probe:           ## rank candidate crawl domains (run FIRST, prune sources.yaml)
	python scripts/probe_domains.py --lang hindi
	python scripts/probe_domains.py --lang nepali

discover:        ## expand sitemaps/APIs into the SQLite frontier
	$(PY) discover --lang hindi
	$(PY) discover --lang nepali

# The crawl walks `category_priority` from sources.yaml one tier at a time, so
# the under-represented registers are collected before news. Long-running:
# prefer scripts/crawl_forever.sh under screen/nohup.
crawl-hi:
	$(PY) crawl --lang hindi  --rps 4 --concurrency 8
crawl-ne:
	$(PY) crawl --lang nepali --rps 4 --concurrency 8

lid-train:       ## train the six-class Devanagari classifier (once, both langs)
	$(PY) lid-train --out models/langid_devanagari.pkl

phase1-hi:
	$(PY) download --lang hindi
	$(PY) clean    --lang hindi $(if $(WORKERS),--workers $(WORKERS),)
	$(PY) lid      --lang hindi
	$(PY) dedup    --lang hindi

phase1-ne:
	$(PY) download --lang nepali
	$(PY) clean    --lang nepali $(if $(WORKERS),--workers $(WORKERS),)
	$(PY) lid      --lang nepali
	$(PY) dedup    --lang nepali

cross:           ## cross-corpus dedup: MUST run after both dedups, before splits
	$(PY) cross --lang-a hindi --lang-b nepali

# sweep writes report/figures/<lang>_vocab_sweep.png. Read it and set vocab_size
# in <lang>/configs/tokenizer.yaml BEFORE running the tokenizer target.
finish-hi:
	$(PY) split --lang hindi && $(PY) sweep --lang hindi
	$(PY) tokenizer --lang hindi && $(PY) encode --lang hindi && $(PY) stats --lang hindi
finish-ne:
	$(PY) split --lang nepali && $(PY) sweep --lang nepali
	$(PY) tokenizer --lang nepali && $(PY) encode --lang nepali && $(PY) stats --lang nepali

all: lid-train phase1-hi phase1-ne cross finish-hi finish-ne
