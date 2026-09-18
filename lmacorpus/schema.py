"""Canonical document record for the Phase-1 corpus pipeline.

Every stage (acquire -> clean -> langid -> dedup -> split -> encode) reads and
writes this record.  Two fields carry hard project requirements:

* ``source_type``  -- "manual" | "downloaded".  Stamped once at ingestion and
  never modified.  The >=20% manual-token rule is computed from this field
  after tokenization, so it must survive every intermediate stage.
* ``doc_id``       -- blake2b content hash of the normalized text.  Gives exact
  deduplication for free and makes split manifests reproducible across re-runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
from typing import Any

MANUAL = "manual"
DOWNLOADED = "downloaded"
VALID_SOURCE_TYPES = (MANUAL, DOWNLOADED)


@dataclass
class Document:
    """A single corpus document plus its full provenance.

    Attributes:
        text: Document body.  Raw at ingestion, normalized after the clean stage.
        lang: ISO code assigned by *our* classifier ("hi"/"ne"), not by the
            upstream source, which is frequently wrong for Devanagari corpora.
        source: Concrete origin, e.g. "sangraha", "ekantipur", "wikisource_hi".
        source_type: "manual" or "downloaded"; drives the 20% manual accounting.
        url: Source URL when known (crawled docs, some HF corpora).
        fetched_at: UTC ISO timestamp of retrieval, for crawl provenance.
        doc_id: Content hash, filled by :meth:`finalize`.
        flags: Free-form per-stage measurements (script_ratio, lid_conf, ...).
    """

    text: str
    lang: str
    source: str
    source_type: str
    url: str | None = None
    fetched_at: str | None = None
    doc_id: str = ""
    flags: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> "Document":
        """Compute the content hash and cheap size statistics.

        Call after the text is in its final form for the current stage.  Text
        length is stored in both characters and UTF-8 bytes because bits-per-byte
        in Phase 2 needs the byte count of the raw held-out text.
        """
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"bad source_type: {self.source_type!r}")
        self.doc_id = blake2b(self.text.encode("utf-8"), digest_size=8).hexdigest()
        self.flags["n_chars"] = len(self.text)
        self.flags["n_bytes"] = len(self.text.encode("utf-8"))
        return self

    def to_json(self) -> str:
        """Serialize to a single JSONL line (Devanagari kept as-is, not escaped)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Document":
        """Inverse of :meth:`to_json`."""
        return Document(**json.loads(line))
