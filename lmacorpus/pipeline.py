"""Stage runners: thin orchestration over the clean / langid / dedup modules.

Each runner reads one shard directory and writes another, plus a JSON stats file
under ``<lang>/data/reports/``.  Keeping orchestration here (rather than in the
CLI) keeps the CLI to argument parsing and makes the stages importable from a
notebook for the report.

**Parallelism.**  Both runners here are CPU-bound and embarrassingly parallel
over shard files, so they fan out across cores with a process pool.  Workers
write into ``<out_dir>/part-XXXXX/`` subdirectories; ``shard_paths`` globs
recursively, so every downstream stage reads the result unchanged and shard
names can never collide between workers.  Pass ``workers=1`` to run in-process
(used by the smoke test and when debugging a crash inside a worker).

Nothing in this pipeline benefits from a GPU: every stage is regex, hashing,
sparse linear algebra or compression.  The scaling axis is CPU cores.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .clean.boilerplate import make_blocklist, strip_boilerplate
from .clean.filters import FilterConfig, QualityFilter
from .clean.langid import LanguageIdentifier
from .clean.normalize import normalize, strip_invisible
from .io_utils import ShardWriter, dump_json, read_dir, read_shard, shard_paths

MIN_LINE_CHARS = 3
LID_BATCH = 512


def _default_workers(n_tasks: int, workers: int | None) -> int:
    """Clamp the requested worker count to something sane for this machine."""
    if workers is None:
        workers = os.cpu_count() or 1
    return max(1, min(workers, n_tasks))


# --------------------------------------------------------------- stage 3+4


def _count_lines_shard(args) -> tuple[dict, dict]:
    """Pass-1 worker: per-source line document-frequency for one shard."""
    path, map_digits = args
    line_counts: dict[str, Counter] = defaultdict(Counter)
    n_docs: Counter = Counter()
    for d in read_shard(path):
        text = normalize(d.text, map_digits)
        line_counts[d.source].update(
            {l.strip() for l in text.split("\n") if len(l.strip()) >= MIN_LINE_CHARS}
        )
        n_docs[d.source] += 1
    return {s: dict(c) for s, c in line_counts.items()}, dict(n_docs)


_CLEAN_CTX: dict = {}


def _clean_init(cfg: dict, blocklists: dict) -> None:
    """Pass-2 initializer: hand each worker the blocklists once, not per task."""
    _CLEAN_CTX["cfg"] = cfg
    _CLEAN_CTX["blocklists"] = blocklists


def _clean_shard(args) -> tuple[int, dict, dict, dict]:
    """Pass-2 worker: normalize -> de-boilerplate -> filter one shard."""
    path, out_dir, idx = args
    cfg = _CLEAN_CTX["cfg"]
    blocklists = _CLEAN_CTX["blocklists"]
    map_digits = cfg.get("map_devanagari_digits", True)
    qf = QualityFilter(FilterConfig(**cfg.get("filters", {})))

    invisible: Counter = Counter()
    kept = 0
    part = Path(out_dir) / f"part-{idx:05d}"
    with ShardWriter(part) as w:
        for d in read_shard(path):
            _, n_inv = strip_invisible(d.text)
            invisible[d.source] += n_inv
            d.text = normalize(d.text, map_digits)
            d.text = strip_boilerplate(d.text, blocklists.get(d.source, set()))
            if not d.text:
                continue
            if qf.keep(d):
                d.finalize()
                w.add(d)
                kept += 1
    return (
        kept,
        {s: dict(c) for s, c in qf.rejected.items()},
        dict(qf.kept),
        dict(invisible),
    )


def run_clean(in_dir, out_dir, report_path, cfg: dict, workers: int | None = None, fast_single_pass: bool = True) -> dict:
    """Stage 3+4: normalize, strip boilerplate, apply quality filters.

    If fast_single_pass=True, runs directly in a single high-speed pass over raw shards.
    """
    paths = shard_paths(in_dir)
    if not paths:
        raise FileNotFoundError(f"no shards under {in_dir}")
    map_digits = cfg.get("map_devanagari_digits", True)
    max_doc_freq = cfg.get("boilerplate_max_doc_freq", 0.01)
    n_workers = _default_workers(len(paths), workers)

    blocklists = {}
    if not fast_single_pass:
        # Pass 1: per-source boilerplate line frequencies over normalized text.
        line_counts: dict[str, Counter] = defaultdict(Counter)
        n_docs: Counter = Counter()
        pass1_args = [(p, map_digits) for p in paths]
        if n_workers == 1:
            results = map(_count_lines_shard, pass1_args)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_count_lines_shard, pass1_args, chunksize=1))
        for counts, docs in results:
            for source, c in counts.items():
                line_counts[source].update(c)
            n_docs.update(docs)

        blocklists = {
            s: make_blocklist(c, n_docs[s], max_doc_freq) for s, c in line_counts.items()
        }
        del line_counts  # can be gigabytes on a large crawl

    # Single Pass: normalize -> de-boilerplate -> filter.
    pass2_args = [(p, out_dir, i) for i, p in enumerate(paths)]
    if n_workers == 1:
        _clean_init(cfg, blocklists)
        results2 = [_clean_shard(a) for a in pass2_args]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_clean_init, initargs=(cfg, blocklists)
        ) as ex:
            results2 = list(ex.map(_clean_shard, pass2_args, chunksize=1))

    kept = 0
    merged = QualityFilter(FilterConfig(**cfg.get("filters", {})))
    invisible_removed: Counter = Counter()
    for n, rejected, kept_by_source, invisible in results2:
        kept += n
        for source, reasons in rejected.items():
            merged.rejected[source].update(reasons)
        merged.kept.update(kept_by_source)
        invisible_removed.update(invisible)

    stats = {
        "kept": kept,
        "workers": n_workers,
        "shards_in": len(paths),
        "filters": merged.report(),
        "filter_config": FilterConfig(**cfg.get("filters", {})).__dict__,
        "invisible_chars_removed": dict(invisible_removed),
        "boilerplate_lines_blocked": {s: len(b) for s, b in blocklists.items()},
    }
    dump_json(report_path, stats)
    return stats


# ----------------------------------------------------------------- stage 5


_LID_CTX: dict = {}

def _langid_init(model_path, target, min_conf):
    _LID_CTX["lid"] = LanguageIdentifier(model_path, target, min_conf)

def _langid_shard(args):
    path, out_dir, quar_dir, idx, batch_size = args
    lid = _LID_CTX["lid"]
    part_keep = Path(out_dir) / f"part-{idx:05d}"
    part_quar = Path(quar_dir) / f"part-{idx:05d}"
    part_keep.mkdir(parents=True, exist_ok=True)
    part_quar.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    kept = dropped = 0

    with ShardWriter(part_keep) as keep_w, ShardWriter(part_quar) as quar_w:
        batch: list = []
        for d in read_shard(path):
            batch.append(d)
            if len(batch) >= batch_size:
                for doc, ok in zip(batch, lid.accept_many(batch)):
                    c = counts.setdefault(doc.source, {})
                    pred = doc.flags["lid_pred"]
                    c[pred] = c.get(pred, 0) + 1
                    if ok:
                        keep_w.add(doc)
                        kept += 1
                    else:
                        quar_w.add(doc)
                        dropped += 1
                batch = []
        if batch:
            for doc, ok in zip(batch, lid.accept_many(batch)):
                c = counts.setdefault(doc.source, {})
                pred = doc.flags["lid_pred"]
                c[pred] = c.get(pred, 0) + 1
                if ok:
                    keep_w.add(doc)
                    kept += 1
                else:
                    quar_w.add(doc)
                    dropped += 1

    return kept, dropped, counts

def run_langid(in_dir, out_dir, quarantine_dir, model_path, target, report_path,
               min_conf: float = 0.90, batch_size: int = LID_BATCH, workers: int | None = 10) -> dict:
    """Stage 5: keep confidently in-target documents, quarantine the rest (Parallel across workers)."""
    paths = shard_paths(in_dir)
    if not paths:
        return {}
    n_workers = _default_workers(len(paths), workers)

    args = [(p, out_dir, quarantine_dir, i, batch_size) for i, p in enumerate(paths)]

    total_kept = 0
    total_dropped = 0
    merged_counts: dict[str, dict[str, int]] = {}

    if n_workers == 1:
        _langid_init(model_path, target, min_conf)
        results = [_langid_shard(a) for a in args]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_langid_init, initargs=(model_path, target, min_conf)
        ) as ex:
            results = list(ex.map(_langid_shard, args, chunksize=1))

    for k, d, cnts in results:
        total_kept += k
        total_dropped += d
        for src, pred_dict in cnts.items():
            s_c = merged_counts.setdefault(src, {})
            for pred, count in pred_dict.items():
                s_c[pred] = s_c.get(pred, 0) + count

    stats = {
        "target": target,
        "min_conf": min_conf,
        "batch_size": batch_size,
        "kept": total_kept,
        "quarantined": total_dropped,
        "predictions_by_source": merged_counts,
        "contamination_rate_by_source": {
            s: round(1 - (c.get(target, 0) / max(sum(c.values()), 1)), 4)
            for s, c in merged_counts.items()
        },
    }
    dump_json(report_path, stats)
    return stats
