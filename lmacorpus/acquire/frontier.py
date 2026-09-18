"""Resumable crawl frontier backed by SQLite.

An in-memory queue dies with the Colab session; this survives.  The table also
*is* the crawl statistics for the report: URLs discovered, fetched, failed and
skipped per domain, plus characters yielded, all one GROUP BY away.

WAL journal mode lets a monitoring process read progress while the crawler
writes.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
  url        TEXT PRIMARY KEY,
  domain     TEXT NOT NULL,
  status     TEXT NOT NULL DEFAULT 'pending',   -- pending|claimed|done|failed|skipped
  attempts   INTEGER NOT NULL DEFAULT 0,
  fetched_at TEXT,
  n_chars    INTEGER
);
CREATE INDEX IF NOT EXISTS ix_status_domain ON urls(status, domain);
"""


class Frontier:
    """URL queue with per-domain claiming and crash-safe status tracking."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def add_many(self, urls, domain: str) -> int:
        """Insert URLs, ignoring ones already known.  Returns rows attempted."""
        rows = [(u, domain) for u in urls]
        self.db.executemany("INSERT OR IGNORE INTO urls(url, domain) VALUES (?,?)", rows)
        return len(rows)

    def claim(self, domain: str, n: int) -> list[str]:
        """Atomically take up to n pending URLs for a domain."""
        rows = self.db.execute(
            "SELECT url FROM urls WHERE status='pending' AND domain=? LIMIT ?",
            (domain, n),
        ).fetchall()
        urls = [r[0] for r in rows]
        if urls:
            self.db.executemany(
                "UPDATE urls SET status='claimed' WHERE url=?", [(u,) for u in urls]
            )
        return urls

    def mark(self, url: str, status: str, n_chars: int | None = None) -> None:
        """Record the outcome of one fetch."""
        self.db.execute(
            "UPDATE urls SET status=?, fetched_at=?, n_chars=?, attempts=attempts+1 "
            "WHERE url=?",
            (status, dt.datetime.utcnow().isoformat(timespec="seconds"), n_chars, url),
        )

    def requeue_claimed(self) -> int:
        """Return URLs claimed by a killed run to the pending pool."""
        cur = self.db.execute("UPDATE urls SET status='pending' WHERE status='claimed'")
        return cur.rowcount

    def domains(self) -> list[str]:
        """Domains that still have pending URLs."""
        return [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT domain FROM urls WHERE status='pending'"
            ).fetchall()
        ]

    def url_counts(self) -> dict[str, int]:
        """Total URL rows per domain, whatever their status.

        Used by ``discover`` to skip domains whose sitemaps have already been
        walked: re-walking them re-downloads the whole sitemap tree over the
        network only for ``INSERT OR IGNORE`` to discard every row.
        """
        return {
            d: n
            for d, n in self.db.execute(
                "SELECT domain, COUNT(*) FROM urls GROUP BY domain"
            )
        }

    def stats(self) -> dict:
        """Per-domain crawl statistics for the Phase-1 report."""
        rows = self.db.execute(
            "SELECT domain, status, COUNT(*), COALESCE(SUM(n_chars),0) "
            "FROM urls GROUP BY domain, status"
        ).fetchall()
        out: dict = {}
        for domain, status, n, chars in rows:
            d = out.setdefault(domain, {"chars": 0})
            d[status] = n
            d["chars"] += chars
        return out
