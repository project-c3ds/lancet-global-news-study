#!/usr/bin/env python3
"""Extract publication dates from article URLs for sources with missing dates.

Fetches each URL, parses HTML for publication date using multiple strategies:
  1. JSON-LD (datePublished)
  2. Open Graph / meta tags (article:published_time, pubdate, etc.)
  3. <time> elements with datetime attribute
  4. URL path patterns (/2023/05/15/, /20230515-, etc.)

Usage:
    python extract_dates.py --source capital.bg --limit 20    # test run
    python extract_dates.py --source capital.bg                # full source
    python extract_dates.py --all --limit 50                   # test all sources
    python extract_dates.py --all                              # full run
"""

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

DB_PATH = "data/articles.db"

# Rate limit: seconds between requests per domain
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = 10

# Sources with >50% missing dates and no NewsAPI
# Tested 2026-03-28: see results/date_extraction_test.csv for hit rates
NO_DATE_SOURCES = [
    # High hit rate (>=75%)
    "aftonbladet.se", "nst.com.my", "mmbiztoday.com", "manager.co.th",
    "ihned.cz", "dnevnik.bg", "bt.dk", "bukedde.co.ug", "miadhu.mv",
    "napi.hu", "lephareonline.net", "capital.bg",
    # Medium hit rate (35-70%)
    "ycwb.com", "opinion.com.bo", "hani.co.kr", "ethnos.gr",
    "people.com.cn", "uzhurriyat.uz",
]

# Common date meta tag names/properties
DATE_META_NAMES = [
    "article:published_time",
    "og:article:published_time",
    "pubdate",
    "publishdate",
    "date",
    "dc.date",
    "dc.date.issued",
    "dcterms.date",
    "sailthru.date",
    "article.published",
    "published_time",
    "publication_date",
    "article_date_original",
    "cxenseparse:recs:publishtime",
]

# URL date patterns
URL_DATE_PATTERNS = [
    # /2023/05/15/ or /2023-05-15/
    re.compile(r"/(\d{4})[/-](\d{2})[/-](\d{2})(?:/|$|\?)"),
    # /202305/15/ (e.g. chinadaily.com.cn)
    re.compile(r"/(\d{4})(\d{2})/(\d{2})/"),
    # /20230515 in path
    re.compile(r"/(\d{4})(\d{2})(\d{2})(?:-|/|$)"),
]


def parse_date_string(s):
    """Try to parse a date string into YYYY-MM-DD format."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()

    # Try common formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]:
        try:
            dt = datetime.strptime(s[:30], fmt)
            if 2019 <= dt.year <= 2027:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try to find a date-like substring
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2019 <= y <= 2027 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def extract_from_jsonld(soup):
    """Extract datePublished from JSON-LD structured data."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle list of objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    date = item.get("datePublished") or item.get("dateCreated")
                    if date:
                        parsed = parse_date_string(date)
                        if parsed:
                            return parsed
        elif isinstance(data, dict):
            date = data.get("datePublished") or data.get("dateCreated")
            if date:
                parsed = parse_date_string(date)
                if parsed:
                    return parsed
            # Check nested @graph
            for item in data.get("@graph", []):
                if isinstance(item, dict):
                    date = item.get("datePublished") or item.get("dateCreated")
                    if date:
                        parsed = parse_date_string(date)
                        if parsed:
                            return parsed
    return None


def extract_from_meta(soup):
    """Extract date from meta tags."""
    for name in DATE_META_NAMES:
        # Try property attribute
        tag = soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            parsed = parse_date_string(tag["content"])
            if parsed:
                return parsed
        # Try name attribute
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            parsed = parse_date_string(tag["content"])
            if parsed:
                return parsed
        # Try itemprop
        tag = soup.find("meta", attrs={"itemprop": name.replace("article:", "")})
        if tag and tag.get("content"):
            parsed = parse_date_string(tag["content"])
            if parsed:
                return parsed
    return None


def extract_from_time_tag(soup):
    """Extract date from <time> elements."""
    for time_tag in soup.find_all("time", datetime=True):
        parsed = parse_date_string(time_tag["datetime"])
        if parsed:
            return parsed
    return None


def extract_from_url(url):
    """Extract date from URL path patterns."""
    for pattern in URL_DATE_PATTERNS:
        m = pattern.search(url)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 2019 <= y <= 2027 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def extract_date(html, url):
    """Try all extraction methods in priority order."""
    # Try URL first (cheapest)
    date = extract_from_url(url)
    if date:
        return date, "url"

    if not html:
        return None, None

    soup = BeautifulSoup(html, "lxml")

    date = extract_from_jsonld(soup)
    if date:
        return date, "jsonld"

    date = extract_from_meta(soup)
    if date:
        return date, "meta"

    date = extract_from_time_tag(soup)
    if date:
        return date, "time_tag"

    return None, None


def fetch_url(url, session):
    """Fetch URL with timeout, return HTML or None."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (compatible; research bot)"})
        if resp.status_code == 200:
            return resp.text
    except (requests.RequestException, Exception):
        pass
    return None


def process_source(conn, source_uri, limit=None, test_mode=False):
    """Extract dates for a single source. Returns stats dict."""
    cursor = conn.cursor()

    rows = cursor.execute(
        "SELECT id, url FROM articles "
        "WHERE source_uri = ? AND collection_method = 'scrapai' "
        "AND published_date IS NULL",
        (source_uri,),
    ).fetchall()

    if limit:
        rows = rows[:limit]

    if not rows:
        return {"source": source_uri, "total": 0}

    session = requests.Session()
    stats = {"source": source_uri, "total": len(rows),
             "url": 0, "jsonld": 0, "meta": 0, "time_tag": 0,
             "failed": 0, "fetch_error": 0}
    updates = []

    for i, (art_id, url) in enumerate(rows):
        if not url:
            stats["failed"] += 1
            continue

        # Try URL pattern first (no fetch needed)
        date, method = extract_from_url(url), "url" if extract_from_url(url) else None
        if not date:
            date, method = extract_from_url(url), None

        # Need to re-extract properly
        date = extract_from_url(url)
        if date:
            method = "url"
        else:
            html = fetch_url(url, session)
            if html is None:
                stats["fetch_error"] += 1
                continue
            date, method = extract_date(html, url)
            time.sleep(REQUEST_DELAY)

        if date and method:
            stats[method] += 1
            updates.append((date, art_id))
        else:
            stats["failed"] += 1

        if (i + 1) % 20 == 0:
            found = stats["url"] + stats["jsonld"] + stats["meta"] + stats["time_tag"]
            print("    [{}/{}] found: {}, failed: {}, fetch_error: {}".format(
                i + 1, len(rows), found, stats["failed"], stats["fetch_error"]))

    # Update SQLite
    if updates and not test_mode:
        cursor.executemany(
            "UPDATE articles SET published_date = ? WHERE id = ?",
            updates,
        )
        conn.commit()

    found = stats["url"] + stats["jsonld"] + stats["meta"] + stats["time_tag"]
    stats["found"] = found
    stats["hit_rate"] = round(found / len(rows) * 100, 1) if rows else 0
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract publication dates from URLs")
    parser.add_argument("--source", help="Process a single source")
    parser.add_argument("--all", action="store_true", help="Process all no-date sources")
    parser.add_argument("--limit", type=int, help="Limit articles per source (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Don't update SQLite")
    args = parser.parse_args()

    if not args.source and not args.all:
        parser.error("Specify --source or --all")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    if args.source:
        sources = [args.source]
    else:
        sources = NO_DATE_SOURCES

    print("Sources: {}".format(len(sources)))
    if args.limit:
        print("Limit: {} articles per source".format(args.limit))
    if args.dry_run:
        print("DRY RUN — no SQLite updates")
    print()

    all_stats = []
    for i, source in enumerate(sources, 1):
        print("[{}/{}] {}...".format(i, len(sources), source))
        stats = process_source(conn, source, limit=args.limit, test_mode=args.dry_run)

        if stats["total"] == 0:
            print("  No articles with missing dates")
        else:
            print("  {} articles: {}% hit rate (url={}, jsonld={}, meta={}, time={}, failed={}, fetch_err={})".format(
                stats["total"], stats["hit_rate"],
                stats["url"], stats["jsonld"], stats["meta"], stats["time_tag"],
                stats["failed"], stats["fetch_error"]))
        all_stats.append(stats)

    # Summary
    print("\n=== Summary ===")
    print("{:30s} {:>8s} {:>6s} {:>5s} {:>6s} {:>5s} {:>5s}".format(
        "source", "total", "hit%", "url", "jsonld", "meta", "time"))
    print("-" * 70)
    for s in all_stats:
        if s["total"] > 0:
            print("{:30s} {:>8,} {:>5.1f}% {:>5} {:>6} {:>5} {:>5}".format(
                s["source"], s["total"], s["hit_rate"],
                s["url"], s["jsonld"], s["meta"], s["time_tag"]))

    conn.close()


if __name__ == "__main__":
    main()
