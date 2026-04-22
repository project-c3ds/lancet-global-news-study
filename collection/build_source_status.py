#!/usr/bin/env python3
"""Build a CSV mapping sources to their collection status in premium folders."""

import csv
import gzip
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path("data")
COUNTRIES_FILE = DATA_DIR / "sources" / "countries.txt"
SOURCES_CSV = DATA_DIR / "sources" / "top5_per_country.csv"
OUTPUT_CSV = DATA_DIR / "sources" / "source_status.csv"
PREMIUM_FOLDERS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]


def url_to_folder_name(url):
    """Normalize a URL to the folder naming convention used in premium folders."""
    parsed = urlparse(url.strip().rstrip("/"))
    host = parsed.hostname or ""
    host = host.removeprefix("www.")
    return host.replace(".", "_")


def build_collection_lookup():
    """Map folder name -> (premium folder, list of crawl files)."""
    lookup = {}
    for folder in PREMIUM_FOLDERS:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            continue
        for source_dir in folder_path.iterdir():
            crawl_dir = source_dir / "crawls"
            if crawl_dir.is_dir():
                files = sorted(crawl_dir.glob("*.jsonl.gz"))
                if files:
                    lookup[source_dir.name] = (folder, files)
    return lookup


def count_articles(files):
    """Count total lines across JSONL files."""
    total = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def main():
    countries = set(COUNTRIES_FILE.read_text().strip().splitlines())
    lookup = build_collection_lookup()

    rows = []
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["country"] not in countries:
                continue

            folder_name = url_to_folder_name(row["website_url"])
            match = lookup.get(folder_name)

            if match:
                collection_id, files = match
                article_count = count_articles(files)
                collected = "Yes"
            else:
                collection_id = ""
                article_count = 0
                collected = "No"

            rows.append({
                "country": row["country"],
                "rank": row["rank"],
                "name": row["name"],
                "website_url": row["website_url"],
                "language": row.get("language", ""),
                "collected": collected,
                "article_count": article_count,
                "collection_id": collection_id,
                "folder_name": folder_name,
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "country", "rank", "name", "website_url", "language",
            "collected", "article_count", "collection_id", "folder_name",
        ])
        writer.writeheader()
        writer.writerows(rows)

    collected = sum(1 for r in rows if r["collected"] == "Yes")
    print(f"Total sources for target countries: {len(rows)}")
    print(f"Collected: {collected}")
    print(f"Not collected: {len(rows) - collected}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
