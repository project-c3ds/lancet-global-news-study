#!/usr/bin/env python3
"""Collect articles from NewsAPI for scrapai sources with missing dates.

Downloads to data/newsapi_bad_low_sources/ for later ingestion.

Usage:
    python collect_nodate_sources.py
    python collect_nodate_sources.py --source blick.ch
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter, ReturnInfo, ArticleInfoFlags

load_dotenv()

OUTPUT_DIR = Path("data/newsapi_bad_low_sources")

# (source_uri, newsapi_uri) — newsapi_uri differs only for mingpao
SOURCES = [
    ("blick.ch", "blick.ch"),
    ("derstandard.at", "derstandard.at"),
    ("kompas.com", "kompas.com"),
    ("eluniverso.com", "eluniverso.com"),
    ("elheraldo.hn", "elheraldo.hn"),
    ("thedailystar.net", "thedailystar.net"),
    ("mg.co.za", "mg.co.za"),
    ("okaz.com.sa", "okaz.com.sa"),
    ("zaobao.com.sg", "zaobao.com.sg"),
    ("aawsat.com", "aawsat.com"),
    ("haaretz.co.il", "haaretz.co.il"),
    ("eleconomista.com.mx", "eleconomista.com.mx"),
    ("elkhabar.com", "elkhabar.com"),
    ("is.fi", "is.fi"),
    ("lemonde.fr", "lemonde.fr"),
    ("donga.com", "donga.com"),
    ("jutarnji.hr", "jutarnji.hr"),
    ("nikkei.com", "nikkei.com"),
    ("corriere.it", "corriere.it"),
    ("lastampa.it", "lastampa.it"),
    ("dailynews.co.th", "dailynews.co.th"),
    ("prothomalo.com", "prothomalo.com"),
    ("novilist.hr", "novilist.hr"),
    ("postcourier.com.pg", "postcourier.com.pg"),
    ("latercera.com", "latercera.com"),
    ("eldeber.com.bo", "eldeber.com.bo"),
    ("mathrubhumi.com", "mathrubhumi.com"),
    ("english.ahram.org.eg", "english.ahram.org.eg"),
    ("kauppalehti.fi", "kauppalehti.fi"),
    ("adressa.no", "adressa.no"),
    ("tiempo.hn", "tiempo.hn"),
    ("theeastafrican.co.ke", "theeastafrican.co.ke"),
    ("bharian.com.my", "bharian.com.my"),
    ("standaard.be", "standaard.be"),
    ("elbilad.net", "elbilad.net"),
    ("ce.cn", "ce.cn"),
    ("elcolombiano.com", "elcolombiano.com"),
    ("novosti.rs", "novosti.rs"),
    ("naszemiasto.pl", "naszemiasto.pl"),
    ("mwananchi.co.tz", "mwananchi.co.tz"),
    ("mingpao.com", "news.mingpao.com"),
]


def collect_source(er, source_uri, newsapi_uri, output_path):
    """Collect all articles for a source and write to JSONL."""
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            return existing, "skipped"

    q = QueryArticlesIter(
        sourceUri=newsapi_uri,
        dateStart="2020-01-01",
        dateEnd="2026-03-28",
    )

    ri = ReturnInfo(articleInfo=ArticleInfoFlags(
        body=True, title=True, concepts=False, categories=False,
        links=False, image=False, videos=False, extractedDates=False,
        socialScore=False, sentiment=False, location=False,
        duplicateList=False, originalArticle=False, storyUri=False,
    ))

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for art in q.execQuery(er, returnInfo=ri, sortBy="date"):
            record = {
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "body": art.get("body", ""),
                "lang": art.get("lang", ""),
                "dateTime": art.get("dateTime", ""),
                "source_uri": source_uri,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count, "done"


def main():
    parser = argparse.ArgumentParser(description="Collect no-date sources from NewsAPI")
    parser.add_argument("--source", help="Collect a single source")
    args = parser.parse_args()

    api_key = os.environ["NEWSAPI_KEY"]
    er = EventRegistry(apiKey=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = SOURCES
    if args.source:
        sources = [(s, n) for s, n in sources if s == args.source]

    usage_before = er.getUsageInfo()
    print("Tokens available: {:,}".format(usage_before["availableTokens"]))
    print("Sources to collect: {}".format(len(sources)))
    print()

    for i, (source_uri, newsapi_uri) in enumerate(sources, 1):
        output_path = OUTPUT_DIR / "{}.jsonl".format(source_uri)
        t0 = time.time()

        count, status = collect_source(er, source_uri, newsapi_uri, output_path)
        elapsed = time.time() - t0

        if status == "skipped":
            print("  [{}/{}] {} -> {}: {:,} articles (skipped)".format(
                i, len(sources), source_uri, newsapi_uri, count))
        else:
            print("  [{}/{}] {} -> {}: {:,} articles ({:.0f}s)".format(
                i, len(sources), source_uri, newsapi_uri, count, elapsed))

    usage_after = er.getUsageInfo()
    tokens_used = usage_after["usedTokens"] - usage_before["usedTokens"]
    print()
    print("Tokens used this run: {:,}".format(tokens_used))
    print("Total tokens used: {:,}".format(usage_after["usedTokens"]))
    print("Tokens remaining: {:,}".format(usage_after["availableTokens"]))


if __name__ == "__main__":
    main()
