"""Sitemap discovery and HTML-to-text extraction for the manual crawl.

Sitemaps first: most Indian news CMSs (Quintype, WordPress) publish
date-partitioned sitemap indexes, which hand over tens of thousands of article
URLs with no link-following at all -- faster and far more polite than crawling.
Archive date-pagination is the fallback for sites without them.

``trafilatura`` is a boilerplate remover, not a language model, so it is within
the project's "no pretrained models" constraint and is much more reliable across
many domains than hand-written CSS selectors.
"""

from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET

SITEMAP_HINTS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/sitemap.xml.gz",
    "/sitemap_news.xml",
    "/news-sitemap.xml",
    "/post-sitemap.xml",
    "/content-sitemap.xml",
    "/sitemap-posttype-post.xml",
    "/wp-sitemap.xml",
    "/sitemap/sitemap-index.xml",
    "/sitemap/sitemap.xml",
    "/sitemap/index.xml",
    "/arc/outboundfeeds/sitemap/",
    "/feed/",
    "/rss/",
    "/robots.txt",
)
SITEMAP_FROM_ROBOTS = re.compile(r"(?im)^\s*sitemap:\s*(\S+)")
MIN_EXTRACT_CHARS = 200


def _localname(tag: str) -> str:
    """Strip the XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(content: bytes) -> tuple[list[str], list[str]]:
    """Split one sitemap or feed document into (nested sitemap URLs, page URLs)."""
    if content[:2] == b"\x1f\x8b":  # gzipped sitemap
        try:
            content = gzip.decompress(content)
        except Exception:
            return [], []

    nested, pages = [], []
    try:
        root = ET.parse(io.BytesIO(content)).getroot()
        is_index = _localname(root.tag) == "sitemapindex"
        for el in root.iter():
            tag = _localname(el.tag).lower()
            if tag in ("loc", "link") and el.text:
                url = el.text.strip()
                if is_index or url.endswith((".xml", ".xml.gz")) or "sitemap" in url.lower():
                    nested.append(url)
                else:
                    pages.append(url)
    except ET.ParseError:
        pass

    # Direct regex fallback if ElementTree parsed nothing or failed
    if not nested and not pages:
        text = content.decode("utf-8", errors="ignore")
        loc_urls = re.findall(r"<loc>\s*(https?://[^\s<>]+)\s*</loc>", text, re.IGNORECASE)
        link_matches = re.findall(
            r"<link(?:\s+[^>]*href=[\"'](https?://[^\"']+)[\"']|\s*>(https?://[^\s<>]+)\s*</link>)",
            text,
            re.IGNORECASE,
        )
        found = list(loc_urls)
        for m in link_matches:
            u = m[0] or m[1]
            if u:
                found.append(u)

        for u in found:
            u = u.strip()
            if u.endswith((".xml", ".xml.gz")) or "sitemap" in u.lower():
                nested.append(u)
            else:
                pages.append(u)

    # De-duplicate while preserving order
    out_n, seen_n = [], set()
    for u in nested:
        if u not in seen_n:
            seen_n.add(u)
            out_n.append(u)

    out_p, seen_p = [], set()
    for u in pages:
        if u not in seen_p:
            seen_p.add(u)
            out_p.append(u)

    return out_n, out_p


def sitemaps_from_robots(text: str) -> list[str]:
    """Extract Sitemap: declarations from a robots.txt body."""
    return SITEMAP_FROM_ROBOTS.findall(text)


def extract_html_links(html: str, base_url: str) -> list[str]:
    """Extract internal article links from an HTML page for archive/link crawling fallback."""
    from urllib.parse import urljoin, urlparse

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    raw_links = re.findall(r'href=["\']([^"\'#\s]+)["\']', html, re.IGNORECASE)
    urls = []
    seen = set()

    skip_exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".css",
        ".js",
        ".pdf",
        ".zip",
        ".mp4",
        ".mp3",
        ".xml",
        ".json",
    )

    for link in raw_links:
        full_url = urljoin(base_url, link)
        p = urlparse(full_url)
        if p.netloc == base_domain and p.scheme in ("http", "https"):
            clean_url = f"{p.scheme}://{p.netloc}{p.path}"
            if clean_url in seen or clean_url.endswith(skip_exts):
                continue
            if len(p.path) > 1 and not clean_url.endswith(("/tag", "/category", "/author", "/login", "/wp-admin")):
                seen.add(clean_url)
                urls.append(clean_url)
    return urls


def _fallback_html_extract(html: str) -> str | None:
    """Fallback paragraph-level text extraction for custom news HTML structures."""
    clean_html = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
    clean_html = re.sub(r"(?is)<style.*?>.*?</style>", "", clean_html)
    clean_html = re.sub(r"(?is)<!--.*?-->", "", clean_html)

    paras = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", clean_html)
    cleaned_paras = []
    for p in paras:
        txt = re.sub(r"<[^>]+>", " ", p)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) > 25 and not any(nav in txt for nav in ("Copyright", "All Rights Reserved", "सर्वाधिकार सुरक्षित")):
            cleaned_paras.append(txt)

    if cleaned_paras:
        body = "\n\n".join(cleaned_paras)
        if len(body) >= MIN_EXTRACT_CHARS:
            return body
    return None


def extract_text(html: str, url: str | None = None) -> str | None:
    """Extract the article body from raw HTML.

    Returns None when extraction fails or yields too little text to be worth
    keeping. Tries favor_precision first, then standard extraction, then paragraph fallback.
    """
    import trafilatura

    txt = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not txt or len(txt) < MIN_EXTRACT_CHARS:
        txt = trafilatura.extract(
            html,
            url=url,
            favor_precision=False,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    if not txt or len(txt) < MIN_EXTRACT_CHARS:
        txt = _fallback_html_extract(html)

    if not txt or len(txt) < MIN_EXTRACT_CHARS:
        return None
    return txt



