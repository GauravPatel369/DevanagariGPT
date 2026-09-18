"""Tests for sitemap parsing, feed extraction, link extraction, and MediaWiki API discovery."""

import pytest
from lmacorpus.acquire.extract import extract_html_links, parse_sitemap, sitemaps_from_robots


def test_parse_sitemap_xml_and_feed():
    # Standard loc tags
    xml_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/article1</loc></url>
      <url><loc>https://example.com/sitemap_news.xml</loc></url>
    </urlset>"""
    nested, pages = parse_sitemap(xml_content)
    assert "https://example.com/sitemap_news.xml" in nested
    assert "https://example.com/article1" in pages

    # RSS feed with link tags
    rss_content = b"""<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <item><link>https://example.com/news1</link></item>
        <item><link>https://example.com/news2</link></item>
      </channel>
    </rss>"""
    nested, pages = parse_sitemap(rss_content)
    assert "https://example.com/news1" in pages
    assert "https://example.com/news2" in pages


def test_sitemaps_from_robots():
    robots = """User-agent: *
Disallow: /admin
Sitemap: https://example.com/sitemap1.xml
sitemap: https://example.com/sitemap2.xml
"""
    maps = sitemaps_from_robots(robots)
    assert len(maps) == 2
    assert "https://example.com/sitemap1.xml" in maps
    assert "https://example.com/sitemap2.xml" in maps


def test_extract_html_links():
    html = """<html><body>
    <a href="/news/123">News 123</a>
    <a href="https://example.com/article/456">Article 456</a>
    <a href="/image.jpg">Image</a>
    <a href="https://otherdomain.com/page">External</a>
    </body></html>"""
    links = extract_html_links(html, "https://example.com")
    assert "https://example.com/news/123" in links
    assert "https://example.com/article/456" in links
    assert not any("image.jpg" in l for l in links)
    assert not any("otherdomain.com" in l for l in links)
