"""Rank candidate crawl domains before committing the crawler to them.

Run on day one:

    python scripts/probe_domains.py --lang hindi

For each configured site it checks sitemap availability, samples ~20 pages, and
measures extraction success and median article length.  The ``est_tokens``
column is what the seed list should be sorted on.  Ten minutes here avoids
discovering in week three that a domain blocks bots or renders via JavaScript.

3.2 characters per token is a rough Devanagari starting estimate; the real
figure comes from the tokenizer at stage 8.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmacorpus.acquire.crawler import Crawler  # noqa: E402
from lmacorpus.acquire.extract import extract_text  # noqa: E402
from lmacorpus.acquire.frontier import Frontier  # noqa: E402
from lmacorpus.clean.normalize import normalize, script_ratio  # noqa: E402
from lmacorpus.io_utils import dump_json, load_config  # noqa: E402

CHARS_PER_TOKEN = 3.2
SAMPLE = 20


async def probe_site(crawler: Crawler, site: dict) -> dict:
    """Measure one candidate domain's sitemap depth and detailed extraction metrics."""
    urls = await crawler.discover(site["root"], max_urls=1_000)
    discovered_urls = len(urls)
    row = {
        "name": site["name"],
        "domain": site["domain"],
        "discovered_urls": discovered_urls,
        "sampled_urls": 0,
        "robots_allowed": 0,
        "http_success": 0,
        "extraction_success": 0,
        "script_pass": 0,
        "median_chars": 0,
        "estimated_raw_tokens": 0,
        "extract_rate": 0.0,
        "verdict": "",
    }
    if not urls:
        row["verdict"] = "no sitemap — needs archive date-pagination"
        return row

    sample = random.sample(urls, min(SAMPLE, len(urls)))
    row["sampled_urls"] = len(sample)

    robots_allowed = 0
    http_success = 0
    extraction_success = 0
    script_pass = 0
    lengths = []

    for u in sample:
        if not await crawler.allowed(u):
            continue
        robots_allowed += 1

        txt = None
        domain = site["domain"]
        if "/wiki/" in u and any(
            w in domain
            for w in (
                "wikisource.org",
                "wikipedia.org",
                "wikibooks.org",
                "wikiquote.org",
                "kavitakosh.org",
            )
        ):
            txt = await crawler._fetch_mediawiki_text(u)
            if txt:
                http_success += 1

        if not txt:
            r = await crawler._get(u, tries=1)
            if r is None or r.status_code != 200:
                continue
            http_success += 1
            txt = extract_text(r.text, u)

        if not txt:
            continue
        extraction_success += 1

        if script_ratio(normalize(txt)) > 0.75:
            script_pass += 1
            lengths.append(len(txt))

        await asyncio.sleep(crawler.delay)

    row["robots_allowed"] = robots_allowed
    row["http_success"] = http_success
    row["extraction_success"] = extraction_success
    row["script_pass"] = script_pass
    row["extract_rate"] = round(script_pass / len(sample), 3) if sample else 0.0
    row["median_chars"] = int(statistics.median(lengths)) if lengths else 0
    row["estimated_raw_tokens"] = int(discovered_urls * row["median_chars"] / CHARS_PER_TOKEN)
    row["verdict"] = "seed" if (row["extract_rate"] >= 0.35 or row["estimated_raw_tokens"] >= 100_000) else "drop or investigate"
    return row


async def main_async(lang: str):
    cfg = load_config(Path(lang) / "configs" / "sources.yaml")
    fr = Frontier(Path(lang) / "data" / "probe.sqlite")
    rows = []
    async with Crawler(fr, "/tmp/probe", cfg["lang_code"]) as c:
        for site in cfg["manual_sites"]:
            row = await probe_site(c, site)
            rows.append(row)

    rows.sort(key=lambda r: -r["estimated_raw_tokens"])
    print(
        f"\n{'name':18s} {'disc':>7s} {'samp':>4s} {'robo':>4s} {'http':>4s} {'extr':>4s} {'pass':>4s} {'med_len':>7s} {'est_tokens':>12s}  verdict"
    )
    print("-" * 95)
    for r in rows:
        print(
            f"{r['name']:18s} {r['discovered_urls']:>7,} "
            f"{r['sampled_urls']:>4} {r['robots_allowed']:>4} "
            f"{r['http_success']:>4} {r['extraction_success']:>4} "
            f"{r['script_pass']:>4} {r['median_chars']:>7,} "
            f"{r['estimated_raw_tokens']:>12,}  {r['verdict']}"
        )

    out = Path(lang) / "data" / "reports" / "domain_probe.json"
    dump_json(out, rows)
    total = sum(r["estimated_raw_tokens"] for r in rows if r["verdict"] == "seed")
    print(f"\nEstimated manual tokens from seedable domains: {total:,}")
    print(f"Requirement is >=20% of final training tokens. Written to {out}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="hindi | nepali")
    a = ap.parse_args()
    asyncio.run(main_async(a.lang))
