"""Stage 1: manual collection.

The crawler is the only component whose value depends on *when* it starts, so it
runs from day one in the background while the rest of the pipeline is written.

Politeness: robots.txt is honoured per domain, concurrency is capped per domain,
and a fixed inter-request delay is applied.  The User-Agent carries a contact
address.

Raw HTML is archived gzipped before extraction.  If the extractor turns out to
be dropping article bodies on one domain, re-extraction is local instead of
re-crawling tens of thousands of pages.

Budget arithmetic (Hindi, >=100M manual tokens):
    ~900 tokens/article  ->  ~110k articles needed *after* cleaning and dedup
    cleaning+dedup keeps ~65%  ->  target ~170k fetched URLs
"""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import hashlib
import random
import urllib.robotparser
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from ..io_utils import ShardWriter
from ..schema import DOWNLOADED, MANUAL, Document
from .extract import (
    SITEMAP_HINTS,
    extract_html_links,
    extract_text,
    parse_sitemap,
    sitemaps_from_robots,
)
from .frontier import Frontier

USER_AGENT = "LMA-course-crawler/1.0 (+student project; contact: you@example.edu)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class Crawler:
    """Polite, resumable async crawler writing Document shards."""

    def __init__(
        self,
        frontier: Frontier,
        out_dir: str | Path,
        lang: str,
        raw_html_dir: str | Path | None = None,
        rps_per_domain: float = 1.5,
        concurrency_per_domain: int = 2,
        timeout: int = 30,
        max_connections: int | None = None,
    ):
        self.f = frontier
        self.lang = lang
        self.out_dir = Path(out_dir)
        self.raw_html_dir = Path(raw_html_dir) if raw_html_dir else None
        self.delay = 1.0 / rps_per_domain
        self.concurrency_per_domain = concurrency_per_domain
        self.sems: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(concurrency_per_domain)
        )
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.timeout = timeout
        # httpx defaults to max_connections=100, which silently becomes the real
        # ceiling once several dozen domains are in flight: the per-domain
        # semaphores let the work through and it then queues on the pool instead.
        # Size the pool for every domain running at full per-domain concurrency.
        self.max_connections = max_connections
        self.client = None
        self.n_ok = 0
        self.n_fail = 0
        # Extracted characters, seeded from the frontier at run() so a resumed
        # crawl counts what previous sessions already collected.
        self.chars_by_domain: Counter = Counter()
        self.total_chars = 0

    async def __aenter__(self):
        import httpx

        n_domains = max(len(self.f.domains()), 1)
        pool = self.max_connections or min(
            1024, max(100, n_domains * self.concurrency_per_domain)
        )
        self.client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=pool,
                max_keepalive_connections=max(20, pool // 2),
                keepalive_expiry=30.0,
            ),
        )
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()

    # ---------------------------------------------------------------- politeness

    async def allowed(self, url: str) -> bool:
        """Check robots.txt for a URL, caching one parser per domain."""
        domain = urlparse(url).netloc
        if domain not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = await self.client.get(f"https://{domain}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])  # unreachable robots.txt -> treat as permitted
            self.robots[domain] = rp
        try:
            return self.robots[domain].can_fetch(USER_AGENT, url)
        except Exception:
            return True

    async def _get(self, url: str, tries: int = 3):
        """GET with backoff on transient failures.  Returns a response or None."""
        for attempt in range(tries):
            try:
                r = await self.client.get(url)
                if r.status_code in RETRY_STATUS:
                    await asyncio.sleep(5 * 2**attempt + random.random())
                    continue
                return r
            except Exception:
                await asyncio.sleep(2**attempt + random.random())
        return None

    # ---------------------------------------------------------------- discovery

    async def discover_mediawiki(self, root: str, max_urls: int = 200_000) -> list[str]:
        """Query MediaWiki Action API for 100% complete article enumeration."""
        from urllib.parse import quote

        parsed = urlparse(root)
        domain = parsed.netloc
        scheme = parsed.scheme or "https"
        if "kavitakosh" in domain:
            api_url = f"{scheme}://{domain}/wiki/api.php"
        else:
            api_url = f"{scheme}://{domain}/w/api.php"

        pages = []
        apcontinue = None
        while len(pages) < max_urls:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": "0",
                "aplimit": "500",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            try:
                r = await self.client.get(api_url, params=params)
                if r is None or r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break

            query = data.get("query", {})
            allpages = query.get("allpages", [])
            for item in allpages:
                title = item.get("title", "")
                if title:
                    page_url = f"{scheme}://{domain}/wiki/{quote(title.replace(' ', '_'))}"
                    pages.append(page_url)
                    if len(pages) >= max_urls:
                        break

            cont = data.get("continue", {})
            apcontinue = cont.get("apcontinue")
            if not apcontinue or not allpages:
                break

            await asyncio.sleep(self.delay)
        return pages

    async def discover_link_crawl(self, root: str, max_urls: int = 50_000) -> list[str]:
        """Breadth-first HTML link crawler fallback for sites without XML sitemaps."""
        parsed = urlparse(root)
        domain = parsed.netloc
        scheme = parsed.scheme or "https"
        root_url = f"{scheme}://{domain}"

        seen = {root_url}
        queue = [root_url]
        pages = []
        depth = 0
        max_depth = 2

        while queue and len(pages) < max_urls and depth <= max_depth:
            next_queue = []
            for u in queue:
                if not await self.allowed(u):
                    continue
                r = await self._get(u, tries=1)
                if r is None or r.status_code != 200:
                    continue
                links = extract_html_links(r.text, u)
                for link in links:
                    if link not in seen:
                        seen.add(link)
                        pages.append(link)
                        next_queue.append(link)
                        if len(pages) >= max_urls:
                            break
                await asyncio.sleep(self.delay)
                if len(pages) >= max_urls:
                    break
            queue = next_queue
            depth += 1
        return pages

    async def discover_wordpress(self, root: str, max_urls: int = 50_000) -> list[str]:
        """Query WordPress REST API (/wp-json/wp/v2/posts) for article discovery."""
        parsed = urlparse(root)
        domain = parsed.netloc
        scheme = parsed.scheme or "https"
        api_url = f"{scheme}://{domain}/wp-json/wp/v2/posts"

        pages = []
        page_num = 1
        max_pages = max(1, min(max_urls // 100, 200))

        while len(pages) < max_urls and page_num <= max_pages:
            params = {"per_page": 100, "page": page_num, "_fields": "link"}
            try:
                r = await self.client.get(api_url, params=params)
                if r is None or r.status_code != 200:
                    break
                items = r.json()
                if not isinstance(items, list) or not items:
                    break
                for item in items:
                    u = item.get("link")
                    if u:
                        pages.append(u)
                        if len(pages) >= max_urls:
                            break
            except Exception:
                break
            page_num += 1
            await asyncio.sleep(self.delay)
        return pages

    async def discover(self, root: str, max_urls: int = 200_000) -> list[str]:
        """Expand a site's sitemap tree, MediaWiki API, WordPress API, or HTML links into article URLs."""
        domain = urlparse(root).netloc

        # 1. Check if domain is a MediaWiki instance
        if any(
            w in domain
            for w in (
                "wikisource.org",
                "wikipedia.org",
                "wikibooks.org",
                "wikiquote.org",
                "kavitakosh.org",
            )
        ):
            mw_urls = await self.discover_mediawiki(root, max_urls)
            if mw_urls:
                return mw_urls

        # 2. Standard sitemap and RSS/Atom feed discovery
        seen: set[str] = set()
        queue = [root.rstrip("/") + h for h in SITEMAP_HINTS]
        pages: list[str] = []
        while queue and len(pages) < max_urls:
            u = queue.pop()
            if u in seen:
                continue
            seen.add(u)
            r = await self._get(u, tries=2)
            if r is None or r.status_code != 200:
                continue
            if u.endswith("robots.txt"):
                queue.extend(sitemaps_from_robots(r.text))
                continue
            nested, found = parse_sitemap(r.content)
            queue.extend(n for n in nested if n not in seen)
            pages.extend(found)

        # De-duplicate while preserving order.
        out, seen_p = [], set()
        for p in pages:
            if p not in seen_p:
                seen_p.add(p)
                out.append(p)

        # 3. Try WordPress REST API if sitemap/feed discovery yields < 50 URLs
        if len(out) < 50:
            wp_urls = await self.discover_wordpress(root, max_urls=min(max_urls, 20_000))
            for u in wp_urls:
                if u not in seen_p:
                    seen_p.add(u)
                    out.append(u)

        # 4. Fallback to HTML link crawling if total URLs < 50
        if len(out) < 50:
            link_urls = await self.discover_link_crawl(root, max_urls=min(max_urls, 10_000))
            for u in link_urls:
                if u not in seen_p:
                    seen_p.add(u)
                    out.append(u)

        return out[:max_urls]


    # ---------------------------------------------------------------- fetching

    def _archive(self, url: str, html: str) -> None:
        """Store gzipped raw HTML so extraction can be redone without re-crawling."""
        if not self.raw_html_dir:
            return
        h = hashlib.blake2b(url.encode(), digest_size=8).hexdigest()
        d = self.raw_html_dir / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / f"{h}.html.gz", "wt", encoding="utf-8") as fh:
            fh.write(html)

    async def _fetch_mediawiki_text(self, url: str) -> str | None:
        """Directly fetch clean plaintext for MediaWiki articles via Action API."""
        from urllib.parse import unquote

        parsed = urlparse(url)
        domain = parsed.netloc
        scheme = parsed.scheme or "https"
        parts = parsed.path.split("/wiki/", 1)
        if len(parts) < 2:
            return None
        title = unquote(parts[1].replace("_", " "))
        if "kavitakosh" in domain:
            api_url = f"{scheme}://{domain}/wiki/api.php"
        else:
            api_url = f"{scheme}://{domain}/w/api.php"

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "format": "json",
            "redirects": "1",
        }
        try:
            r = await self.client.get(api_url, params=params)
            if r is None or r.status_code != 200:
                return None
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1" and "extract" in pdata:
                    txt = pdata["extract"]
                    if txt and len(txt) >= 200:
                        return txt
        except Exception:
            pass
        return None

    async def fetch_one(self, url: str, source: str, writer: ShardWriter) -> None:
        """Fetch, archive, extract and emit one page.

        The per-host semaphore covers only the network request.  Archiving
        (gzip) and extraction (lxml, via trafilatura) are CPU-bound and are
        pushed to worker threads: run inline they block the event loop, so every
        other domain's sockets stall while one page is parsed — which on a
        many-domain crawl costs far more than the parse itself.  Holding the
        semaphore across that work would also idle the host for no reason.
        """
        if not await self.allowed(url):
            self.f.mark(url, "skipped")
            return
        domain = urlparse(url).netloc
        html = None
        async with self.sems[domain]:
            # Try MediaWiki API extraction first for wiki URLs
            txt = None
            if "/wiki/" in url and any(
                w in domain
                for w in (
                    "wikisource.org",
                    "wikipedia.org",
                    "wikibooks.org",
                    "wikiquote.org",
                    "kavitakosh.org",
                )
            ):
                txt = await self._fetch_mediawiki_text(url)

            if not txt:
                r = await self._get(url)
                await asyncio.sleep(self.delay)
                if r is None or r.status_code != 200:
                    self.f.mark(url, "failed")
                    self.n_fail += 1
                    return
                html = r.text

        if html is not None:
            await asyncio.to_thread(self._archive, url, html)
            txt = await asyncio.to_thread(extract_text, html, url)

        if not txt:
            self.f.mark(url, "failed", 0)
            self.n_fail += 1
            return
        doc = Document(
            text=txt,
            lang=self.lang,
            source=source,
            source_type=MANUAL,
            url=url,
            fetched_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
        ).finalize()
        writer.add(doc)
        self.f.mark(url, "done", len(txt))
        self.n_ok += 1
        self.chars_by_domain[domain] += len(txt)
        self.total_chars += len(txt)


    async def run(
        self,
        source_of: dict[str, str],
        batch: int = 200,
        max_docs: int | None = None,
        **kwargs,
    ):
        """Drain the frontier domain by domain equally until empty or max_docs reached."""
        requeued = self.f.requeue_claimed()
        if requeued:
            print(f"[crawl] Requeued {requeued:,} previously claimed URLs from interrupted run", flush=True)

        with ShardWriter(self.out_dir, shard_docs=1_000) as writer:
            domains = self.f.domains()
            if not domains:
                print(f"[crawl] Frontier for '{self.lang}' has 0 pending URLs. Run 'discover' first:\n        python -m lmacorpus.cli discover --lang {self.lang}")
                return
            print(f"[crawl] {len(domains)} domains in flight simultaneously", flush=True)
            while True:
                domains = self.f.domains()
                if not domains:
                    break
                tasks = []
                for dom in domains:
                    for url in self.f.claim(dom, batch):
                        tasks.append(
                            self.fetch_one(url, source_of.get(dom, dom), writer)
                        )
                if not tasks:
                    break
                await asyncio.gather(*tasks)
                print(f"[crawl] domains={len(domains)} ok={self.n_ok} "
                      f"fail={self.n_fail} chars={self.total_chars:,}", flush=True)
                if max_docs and self.n_ok >= max_docs:
                    break

