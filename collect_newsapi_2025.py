#!/usr/bin/env python3
"""Collect climate and health articles from Event Registry for 2025.

For each source:
  1. Search climate keywords (eng + local language)
  2. Search health keywords (eng + local language), IGNORING climate keywords

Uses QueryArticlesIter to paginate through all results (100 per page).
Saves to data/newsapi_articles_{year}/{source_uri}_climate.jsonl

Usage:
    python collect_newsapi_2025.py --year 2025 --source nytimes.com --dry-run
    python collect_newsapi_2025.py --year 2024 --ignore dailymail.co.uk
    python collect_newsapi_2025.py --year 2023
    python collect_newsapi_2025.py --year 2025 --lang eng
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
MAX_KEYWORDS = 80  # total across keywords + ignoreKeywords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_source_languages():
    """Load source_uri -> language_code mapping."""
    mapping = {}
    with open(SOURCE_LANG_CSV) as f:
        for row in csv.DictReader(f):
            if row["language_code"]:
                mapping[row["source_uri"]] = row["language_code"]
    return mapping


def load_translations():
    """Load keyword translations. Returns {category: {lang_code: [keywords]}}."""
    with open(TRANSLATIONS_JSON) as f:
        return json.load(f)


def get_keywords(translations, category, lang_code, max_count=None):
    """Get combined eng + local keywords for a category, respecting keyword limit."""
    limit = max_count or MAX_KEYWORDS
    eng = translations[category].get("eng", [])
    local = translations[category].get(lang_code, []) if lang_code != "eng" else []

    # Deduplicate (case-insensitive for eng overlap)
    seen = set(k.lower() for k in eng)
    unique_local = [k for k in local if k.lower() not in seen]

    combined = eng + unique_local
    if len(combined) > limit:
        log.warning(f"  {lang_code} {category}: {len(combined)} keywords exceeds {limit}, truncating")
        combined = combined[:limit]

    return combined


def count_existing(filepath):
    """Count lines in existing output file."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath) as f:
        for _ in f:
            count += 1
    return count


def collect_articles(er, source_uri, keywords, date_start, date_end, ignore_keywords=None, dry_run=False):
    """Collect all articles matching keywords for a source.
    Returns (articles_list, total_count)."""

    kwargs = dict(
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
        keywords=QueryItems.OR(keywords),
        keywordSearchMode="phrase",
    )
    if ignore_keywords:
        kwargs["ignoreKeywords"] = QueryItems.OR(ignore_keywords)
        kwargs["ignoreKeywordSearchMode"] = "phrase"

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
    """Save articles to JSONL."""
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
    parser = argparse.ArgumentParser(description="Collect 2025 articles from Event Registry")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year to collect (default: 2025)")
    parser.add_argument("--source", help="Single source to process")
    parser.add_argument("--lang", help="Only process sources with this language code (e.g., eng)")
    parser.add_argument("--ignore", nargs="+", default=[], help="Source URIs to skip (e.g., --ignore dailymail.co.uk)")
    parser.add_argument("--limit", type=int, help="Limit number of sources")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't fetch")
    parser.add_argument("--force", action="store_true", help="Re-collect even if output exists")
    parser.add_argument("--api-key-env", default="NEWSAPI_KEY", help="Env var name for API key (default: NEWSAPI_KEY)")
    args = parser.parse_args()

    year = args.year
    date_start = f"{year}-01-01"
    date_end = f"{year}-12-31"
    output_dir = DATA_DIR / f"newsapi_articles_{year}"

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

    # Load data
    source_langs = load_source_languages()
    translations = load_translations()
    log.info(f"Loaded {len(source_langs)} sources, {len(translations['climate'])} climate langs, {len(translations['health'])} health langs")

    # Filter sources
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

    total_climate = 0
    total_sources_done = 0

    for i, (source_uri, lang_code) in enumerate(sources, 1):
        log.info(f"\n[{i}/{len(sources)}] {source_uri} (lang={lang_code})")

        climate_kw = get_keywords(translations, "climate", lang_code)
        log.info(f"  Climate keywords: {len(climate_kw)}")

        climate_path = output_dir / f"{source_uri}_climate.jsonl"
        existing_climate = count_existing(climate_path)

        if existing_climate > 0 and not args.force:
            log.info(f"  Climate: {existing_climate:,} already collected, skipping")
            total_climate += existing_climate
        else:
            log.info(f"  Climate: searching...")
            articles, count = collect_articles(er, source_uri, climate_kw, date_start, date_end, dry_run=args.dry_run)
            log.info(f"  Climate: {count:,} found" + (f", {len(articles):,} collected" if not args.dry_run else " (dry run)"))
            if not args.dry_run and articles:
                save_articles(articles, climate_path)
            total_climate += count

        total_sources_done += 1

        # Token check every 10 sources
        if total_sources_done % 10 == 0:
            tokens_now = log_usage("checkpoint")
            log.info(f"  --- Tokens used so far: {tokens_now - tokens_before:,} ---")

    # Final summary
    tokens_after = log_usage("after")
    tokens_used = tokens_after - tokens_before

    log.info(f"\n{'='*60}")
    log.info(f"Done. {total_sources_done} sources processed")
    log.info(f"  Climate articles: {total_climate:,}")
    log.info(f"  Tokens used:      {tokens_used:,}")
    if total_sources_done > 0:
        log.info(f"  Avg tokens/source: {tokens_used / total_sources_done:.0f}")
        estimated_total = (tokens_used / total_sources_done) * len(source_langs)
        log.info(f"  Estimated total for all {len(source_langs)} sources: {estimated_total:,.0f} tokens")


if __name__ == "__main__":
    main()
