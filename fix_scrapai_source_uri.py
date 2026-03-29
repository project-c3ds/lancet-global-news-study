#!/usr/bin/env python3
"""Fix scrapai source_uri values in SQLite.

The original import derived source_uri from article URLs, which sometimes
include subdomains (e.g., 'bandung.kompas.com' instead of 'kompas.com').

The correct source_uri for each article is the domain derived from the
folder name in world_news_premium/ (e.g., folder 'kompas_com' -> 'kompas.com').

This script:
1. Scans every crawl file to build a mapping: URL domain -> folder domain
2. Applies UPDATE statements to fix source_uri in the database
3. Verifies all scrapai source_uris match known folder domains
"""

import gzip
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path("data")
PREMIUM_FOLDERS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]
DB_PATH = DATA_DIR / "articles.db"


def domain_from_url(url):
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.replace("www.", "")
        # Strip port numbers (e.g., ':443')
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return ""


def build_url_domain_to_folder_domain_map():
    """Scan all crawl files to map URL domains to their correct folder domain."""
    # url_domain -> set of folder_domains it appears in
    domain_map = defaultdict(set)
    folder_count = 0

    for pf in PREMIUM_FOLDERS:
        fp = DATA_DIR / pf
        if not fp.exists():
            continue
        for source_dir in sorted(fp.iterdir()):
            crawl_dir = source_dir / "crawls"
            if not crawl_dir.is_dir():
                continue
            files = sorted(crawl_dir.glob("*.jsonl.gz"))
            if not files:
                continue

            folder_domain = source_dir.name.replace("_", ".")
            folder_count += 1
            seen_domains = set()

            for f in files:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            art = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        url = (art.get("url") or "").strip()
                        d = domain_from_url(url)
                        if d and d not in seen_domains:
                            seen_domains.add(d)
                            domain_map[d].add(folder_domain)

            if folder_count % 25 == 0:
                print(f"  Scanned {folder_count} folders...", flush=True)

    print(f"  Scanned {folder_count} folders total")
    return domain_map


def main():
    print("Step 1: Scanning crawl files to build domain mapping...")
    t0 = time.time()
    domain_map = build_url_domain_to_folder_domain_map()
    elapsed = time.time() - t0
    print(f"  Found {len(domain_map)} distinct URL domains in {elapsed/60:.1f}m")

    # Build expected folder domains
    folder_domains = set()
    for pf in PREMIUM_FOLDERS:
        fp = DATA_DIR / pf
        if not fp.exists():
            continue
        for d in sorted(fp.iterdir()):
            if (d / "crawls").is_dir():
                folder_domains.add(d.name.replace("_", "."))

    # Build the fix mapping: wrong_source_uri -> correct_source_uri
    # We need to handle:
    # 1. Subdomains (bandung.kompas.com -> kompas.com)
    # 2. Port numbers (elfinanciero.com.mx:443 -> elfinanciero.com.mx)
    # 3. Completely different domains within a folder (karieri.bg -> capital.bg)
    fix_map = {}
    ambiguous = []

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(
        "SELECT DISTINCT source_uri FROM articles WHERE collection_method='scrapai'"
    )
    scrapai_uris = [r[0] for r in c.fetchall()]

    for uri in scrapai_uris:
        if uri in folder_domains:
            continue  # already correct

        # Strip port for lookup
        clean = uri.split(":")[0] if ":" in uri else uri

        # Look up in domain_map
        candidates = domain_map.get(clean, set()) | domain_map.get(uri, set())
        if len(candidates) == 1:
            fix_map[uri] = candidates.pop()
        elif len(candidates) > 1:
            ambiguous.append((uri, candidates))
        else:
            # Try: maybe the URI itself appears with port in domain_map
            found = False
            for dm_key, dm_vals in domain_map.items():
                if dm_key.split(":")[0] == clean:
                    if len(dm_vals) == 1:
                        fix_map[uri] = dm_vals.pop()
                        found = True
                        break
            if not found:
                ambiguous.append((uri, set()))

    print(f"\nStep 2: Fix mapping built")
    print(f"  URIs to fix: {len(fix_map)}")
    print(f"  Ambiguous (need manual review): {len(ambiguous)}")

    if ambiguous:
        print("\n  Ambiguous mappings:")
        for uri, candidates in ambiguous:
            c.execute(
                "SELECT COUNT(*) FROM articles WHERE source_uri=? AND collection_method='scrapai'",
                (uri,),
            )
            cnt = c.fetchone()[0]
            print(f"    {uri} ({cnt:,} articles) -> {candidates or 'NO MATCH'}")

    # Show sample fixes
    print("\n  Sample fixes:")
    shown = 0
    for old, new in sorted(fix_map.items()):
        if old != new:
            c.execute(
                "SELECT COUNT(*) FROM articles WHERE source_uri=? AND collection_method='scrapai'",
                (old,),
            )
            cnt = c.fetchone()[0]
            print(f"    {old} -> {new} ({cnt:,} articles)")
            shown += 1
            if shown >= 30:
                print(f"    ... and {len(fix_map) - 30} more")
                break

    # Count total articles affected
    total_affected = 0
    for old in fix_map:
        c.execute(
            "SELECT COUNT(*) FROM articles WHERE source_uri=? AND collection_method='scrapai'",
            (old,),
        )
        total_affected += c.fetchone()[0]
    print(f"\n  Total articles to update: {total_affected:,}")

    # Apply fixes
    print("\nStep 3: Applying fixes...")
    t0 = time.time()
    fixed_count = 0
    for old_uri, new_uri in sorted(fix_map.items()):
        c.execute(
            "UPDATE articles SET source_uri=? WHERE source_uri=? AND collection_method='scrapai'",
            (new_uri, old_uri),
        )
        rows = c.rowcount
        fixed_count += rows
        if rows > 0:
            print(f"    {old_uri} -> {new_uri}: {rows:,} rows")

    conn.commit()
    elapsed = time.time() - t0
    print(f"\n  Updated {fixed_count:,} rows in {elapsed:.1f}s")

    # Step 4: Verify
    print("\nStep 4: Verification...")
    c.execute(
        "SELECT DISTINCT source_uri FROM articles WHERE collection_method='scrapai'"
    )
    remaining_uris = {r[0] for r in c.fetchall()}
    unexpected = remaining_uris - folder_domains
    if unexpected:
        print(f"  WARNING: {len(unexpected)} source_uris still don't match folders:")
        for u in sorted(unexpected):
            c.execute(
                "SELECT COUNT(*) FROM articles WHERE source_uri=? AND collection_method='scrapai'",
                (u,),
            )
            cnt = c.fetchone()[0]
            print(f"    {u} ({cnt:,} articles)")
    else:
        print(f"  All {len(remaining_uris)} scrapai source_uris match folder domains!")

    # Show final stats
    c.execute(
        "SELECT COUNT(DISTINCT source_uri) FROM articles WHERE collection_method='scrapai'"
    )
    print(f"\n  Distinct scrapai source_uris: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM articles WHERE collection_method='scrapai'")
    print(f"  Total scrapai articles: {c.fetchone()[0]:,}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
