#!/usr/bin/env python3
"""Collect HEALTH articles from Event Registry, one source at a time, for a given year.

For each source, searches health keywords (English + the source's local
language) and saves matches to data/newsapi_health_{year}/{source_uri}_health.jsonl.
Climate articles are collected in a separate pass by collect_newsapi_climate.py.

Uses QueryArticlesIter to paginate through all results (100 per page). The
corpus covers 2021-2025; invoke the script once per year.

Usage:
    python collection/collect_newsapi_health.py --year 2025 --source nytimes.com --dry-run
    python collection/collect_newsapi_health.py --year 2024 --ignore dailymail.co.uk
    python collection/collect_newsapi_health.py --year 2023
    python collection/collect_newsapi_health.py --year 2021 --lang eng
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from eventregistry import (
    EventRegistry, QueryArticlesIter, QueryItems,
    ReturnInfo, ArticleInfoFlags,
)

load_dotenv()

DATA_DIR = Path("data")
SOURCE_LANG_CSV = DATA_DIR / "keywords" / "source_languages.csv"
TRANSLATIONS_JSON = DATA_DIR / "keywords" / "keyword_translations.json"

DEFAULT_YEAR = 2025
PAGE_SIZE = 100
MAX_KEYWORDS = 80

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_source_languages():
    mapping = {}
    with open(SOURCE_LANG_CSV) as f:
        for row in csv.DictReader(f):
            if row["language_code"]:
                mapping[row["source_uri"]] = row["language_code"]
    return mapping


def load_translations():
    with open(TRANSLATIONS_JSON) as f:
        return json.load(f)


def get_keywords(translations, category, lang_code, max_count=None):
    """Get keywords for a category in the source's language only."""
    limit = max_count or MAX_KEYWORDS
    keywords = translations[category].get(lang_code, [])

    word_count = sum(len(kw.split()) for kw in keywords)
    if word_count > limit:
        # Trim keywords to fit within word limit
        trimmed = []
        count = 0
        for kw in keywords:
            w = len(kw.split())
            if count + w > limit:
                break
            trimmed.append(kw)
            count += w
        log.warning(f"  {lang_code} {category}: {word_count} words exceeds {limit}, trimmed to {count}")
        return trimmed

    return keywords


def count_existing(filepath):
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath) as f:
        for _ in f:
            count += 1
    return count


def collect_articles(er, source_uri, keywords, date_start, date_end, dry_run=False):
    kwargs = dict(
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
        keywords=QueryItems.OR(keywords),
        keywordSearchMode="phrase",
    )

    q = QueryArticlesIter(**kwargs)
    total = q.count(er)

    if dry_run:
        return [], total

    ri = ReturnInfo(articleInfo=ArticleInfoFlags(
        body=True, title=True, url=True, authors=True,
        image=True, sentiment=True, concepts=True,
        categories=True, links=True, videos=True,
        socialScore=True, location=True, extractedDates=True,
        originalArticle=True, storyUri=True,
    ))

    articles = []
    for art in q.execQuery(er, returnInfo=ri, maxItems=-1, sortBy="date"):
        articles.append(art)
        if len(articles) % 100 == 0:
            log.info(f"    collected {len(articles):,}/{total:,}")

    return articles, total


def save_articles(articles, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for art in articles:
            record = {
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "body": art.get("body", ""),
                "lang": art.get("lang", ""),
                "dateTime": art.get("dateTime", ""),
                "source_uri": art.get("source", {}).get("uri", ""),
                "authors": art.get("authors", []),
                "sentiment": art.get("sentiment"),
                "concepts": art.get("concepts", []),
                "categories": art.get("categories", []),
                "image": art.get("image", ""),
                "location": art.get("location"),
                "socialScore": art.get("socialScore"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collect health articles from Event Registry")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year to collect (default: 2025)")
    parser.add_argument("--source", help="Single source to process")
    parser.add_argument("--lang", help="Only process sources with this language code (e.g., eng)")
    parser.add_argument("--ignore", nargs="+", default=[], help="Source URIs to skip")
    parser.add_argument("--limit", type=int, help="Limit number of sources")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't fetch")
    parser.add_argument("--force", action="store_true", help="Re-collect even if output exists")
    parser.add_argument("--api-key-env", default="NEWSAPI_KEY", help="Env var name for API key")
    args = parser.parse_args()

    year = args.year
    date_start = f"{year}-01-01"
    date_end = f"{year}-12-31"
    output_dir = DATA_DIR / f"newsapi_health_{year}"

    api_key = os.environ[args.api_key_env]
    er = EventRegistry(apiKey=api_key)

    def log_usage(label=""):
        info = er.getUsageInfo()
        used = info.get("usedTokens", 0)
        available = info.get("availableTokens", 0)
        remaining = available - used
        log.info(f"Tokens {label}: used={used:,} / available={available:,} (remaining={remaining:,})")
        return used

    tokens_before = log_usage("before")

    source_langs = load_source_languages()
    translations = load_translations()
    log.info(f"Loaded {len(source_langs)} sources, {len(translations['health'])} health langs")

    sources = list(source_langs.items())
    if args.source:
        sources = [(s, l) for s, l in sources if s == args.source]
    if args.lang:
        sources = [(s, l) for s, l in sources if l == args.lang]
    if args.ignore:
        ignore_set = set(args.ignore)
        sources = [(s, l) for s, l in sources if s not in ignore_set]
    if args.limit:
        sources = sources[:args.limit]

    if not sources:
        log.error("No sources found")
        return

    log.info(f"Processing {len(sources)} sources for {year} {'(DRY RUN)' if args.dry_run else ''}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_health = 0
    total_sources_done = 0

    for i, (source_uri, lang_code) in enumerate(sources, 1):
        log.info(f"\n[{i}/{len(sources)}] {source_uri} (lang={lang_code})")

        health_kw = get_keywords(translations, "health", lang_code)
        if not health_kw:
            log.warning(f"  No health keywords for lang={lang_code}, skipping")
            continue

        word_count = sum(len(kw.split()) for kw in health_kw)
        log.info(f"  Health keywords: {len(health_kw)} phrases, {word_count} words")

        health_path = output_dir / f"{source_uri}_health.jsonl"
        existing = count_existing(health_path)

        if existing > 0 and not args.force:
            log.info(f"  Health: {existing:,} already collected, skipping")
            total_health += existing
        else:
            log.info(f"  Health: searching...")
            articles, count = collect_articles(
                er, source_uri, health_kw, date_start, date_end,
                dry_run=args.dry_run,
            )
            log.info(f"  Health: {count:,} found" + (f", {len(articles):,} collected" if not args.dry_run else " (dry run)"))
            if not args.dry_run and articles:
                save_articles(articles, health_path)
            total_health += count

        total_sources_done += 1

        if total_sources_done % 10 == 0:
            tokens_now = log_usage("checkpoint")
            log.info(f"  --- Tokens used so far: {tokens_now - tokens_before:,} ---")

    tokens_after = log_usage("after")
    tokens_used = tokens_after - tokens_before

    log.info(f"\n{'='*60}")
    log.info(f"Done. {total_sources_done} sources processed")
    log.info(f"  Health articles: {total_health:,}")
    log.info(f"  Tokens used:     {tokens_used:,}")
    if total_sources_done > 0:
        log.info(f"  Avg tokens/source: {tokens_used / total_sources_done:.0f}")
        estimated_total = (tokens_used / total_sources_done) * len(source_langs)
        log.info(f"  Estimated total for all {len(source_langs)} sources: {estimated_total:,.0f} tokens")


if __name__ == "__main__":
    main()
