#!/usr/bin/env bash
# Phase 1 for nepali. Resumable: re-running any stage is safe.
# The crawl (steps 1-2) runs separately and continuously — see README.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python -m lmacorpus.cli"

$PY download  --lang nepali      # stage 2  public corpora (streaming)
$PY clean     --lang nepali      # stage 3-4 normalize + boilerplate + filters
$PY lid       --lang nepali      # stage 5  six-class Devanagari classifier
$PY dedup     --lang nepali      # stage 6a exact + near-duplicate

echo "Now run cross-corpus dedup ONCE for both languages:"
echo "  $PY cross --lang-a hindi --lang-b nepali"
echo "then re-run this script from the split step below."

$PY split     --lang nepali      # stage 7  group-aware manifests
$PY sweep     --lang nepali      # stage 8b vocab sweep -> set vocab_size in configs
$PY tokenizer --lang nepali      # stage 8a final SentencePiece model
$PY encode    --lang nepali      # stage 8c uint16 memmaps + meta.json
$PY stats     --lang nepali      # stage 8  tables + figures
