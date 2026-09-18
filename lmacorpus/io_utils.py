"""Sharded, compressed JSONL I/O and YAML config loading.

Every pipeline stage consumes a directory of ``shard-*.jsonl.zst`` files and
produces another such directory.  Sharding buys three things:

1. Any stage can be re-run in isolation when a bug is found late.
2. Stages parallelise trivially with a process pool over shard files.
3. A crashed run resumes by skipping shards that already exist downstream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import yaml
import zstandard as zstd

from .schema import Document

SHARD_DOCS = 50_000
SUFFIX = ".jsonl.zst"


def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ShardWriter:
    """Buffered writer that rolls over to a new shard every ``shard_docs`` records.

    Use as a context manager so the final partial shard is always flushed.
    """

    def __init__(self, out_dir: str | Path, shard_docs: int = SHARD_DOCS, level: int = 6):
        import re

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self.level = level
        self.n_docs = 0
        self._buf: list[Document] = []

        existing = shard_paths(self.out_dir)
        nums = [
            int(m.group(1))
            for p in existing
            if (m := re.search(r"shard-(\d+)", p.name))
        ]
        self.n_shards = max(nums) + 1 if nums else 0

    def add(self, doc: Document) -> None:
        """Queue one document, flushing when the shard is full."""
        self._buf.append(doc)
        self.n_docs += 1
        if len(self._buf) >= self.shard_docs:
            self.flush()

    def flush(self) -> None:
        """Write the buffered documents as one shard."""
        if not self._buf:
            return
        path = self.out_dir / f"shard-{self.n_shards:05d}{SUFFIX}"
        write_shard(path, self._buf)
        self.n_shards += 1
        self._buf.clear()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.flush()


def write_shard(path: str | Path, docs: Iterable[Document]) -> None:
    """Write documents to one zstd-compressed JSONL shard."""
    cctx = zstd.ZstdCompressor(level=6)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh, cctx.stream_writer(fh) as w:
        for d in docs:
            w.write((d.to_json() + "\n").encode("utf-8"))
    os.replace(tmp, path)  # atomic: a killed run never leaves a half shard


def read_shard(path: str | Path) -> Iterator[Document]:
    """Stream documents out of one shard."""
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as r:
        for line in _lines(r):
            if line.strip():
                yield Document.from_json(line)


def _lines(binary_stream) -> Iterator[str]:
    """Decode a binary stream into text lines without loading it all in memory."""
    import io

    for line in io.TextIOWrapper(binary_stream, encoding="utf-8"):
        yield line


def shard_paths(root: str | Path | list | tuple) -> list[Path]:
    """All shards under ``root`` (or list of roots), recursively, in sorted (deterministic) order."""
    if isinstance(root, (list, tuple)):
        out = []
        for r in root:
            out.extend(Path(r).rglob(f"*{SUFFIX}"))
        return sorted(set(out))
    return sorted(Path(root).rglob(f"*{SUFFIX}"))


def read_dir(root: str | Path) -> Iterator[Document]:
    """Stream every document under a directory tree of shards."""
    for p in shard_paths(root):
        yield from read_shard(p)


def dump_json(path: str | Path, obj) -> None:
    """Write a JSON report file with stable key order and readable Devanagari."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
