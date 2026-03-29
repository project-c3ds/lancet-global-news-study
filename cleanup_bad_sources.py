#!/usr/bin/env python3
"""Remove scrapai articles for sources that have better newsapi data.

Removes from both SQLite and Weaviate.

Usage:
    python cleanup_bad_sources.py --dry-run       # show what would be removed
    python cleanup_bad_sources.py                  # actually remove
    python cleanup_bad_sources.py --sqlite-only    # only clean SQLite
    python cleanup_bad_sources.py --weaviate-only  # only clean Weaviate
"""

import argparse
import sqlite3
import time

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

BAD_SOURCES_DOT = [
    "ekstrabladet.dk", "fakty.ua", "japantimes.co.jp", "lapresse.ca",
    "mb.com.ph", "segabg.com", "aif.ru", "chosun.com", "dn.no",
    "echoroukonline.com", "elcomercio.com", "ennaharonline.com", "ft.com",
    "guardian.ng", "ilsole24ore.com", "inquirer.net", "kleinezeitung.at",
    "lanacion.cl", "lequipe.fr", "liberation.fr", "lidovky.cz",
    "telegraaf.nl", "thechronicle.com.gh", "thisdaylive.com", "vanguardngr.com",
]

# Weaviate uses underscore format after the fix
BAD_SOURCES_UNDERSCORE = set(s.replace(".", "_") for s in BAD_SOURCES_DOT)

DB_PATH = "data/articles.db"


def cleanup_sqlite(dry_run=False):
    db = sqlite3.connect(DB_PATH)
    print("SQLite cleanup:")
    total = 0
    for source in BAD_SOURCES_DOT:
        count = db.execute(
            "SELECT COUNT(*) FROM articles WHERE source_uri=? AND collection_method='scrapai'",
            (source,),
        ).fetchone()[0]
        if count > 0:
            total += count
            print(f"  {source:25s} {count:>6,} scrapai to remove")

    print(f"  Total: {total:,}")

    if not dry_run and total > 0:
        placeholders = ",".join("?" for _ in BAD_SOURCES_DOT)
        db.execute(
            f"DELETE FROM articles WHERE source_uri IN ({placeholders}) AND collection_method='scrapai'",
            BAD_SOURCES_DOT,
        )
        db.commit()
        print(f"  Deleted {total:,} rows")

    db.close()


def cleanup_weaviate(dry_run=False):
    """Iterate all articles, delete scrapai ones for bad sources."""
    client = weaviate.connect_to_local(
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=300, insert=300)
        ),
    )
    try:
        col = client.collections.get("NewsArticles")
        print("\nWeaviate cleanup (iterating, deleting scrapai for bad sources):")

        t_start = time.time()
        scanned = 0
        to_delete = []
        kept = 0

        for obj in col.iterator(return_properties=["source", "collection_method"]):
            scanned += 1
            source = obj.properties.get("source", "")
            cm = obj.properties.get("collection_method", "")

            if source in BAD_SOURCES_UNDERSCORE and cm == "scrapai":
                to_delete.append(obj.uuid)
            elif source in BAD_SOURCES_UNDERSCORE:
                kept += 1

            if scanned % 500_000 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  scanned {scanned:,} | to_delete={len(to_delete):,} | kept={kept:,} | {scanned/elapsed:.0f}/s",
                    flush=True,
                )

        print(f"  Scan complete: {len(to_delete):,} to delete, {kept:,} newsapi to keep")

        if not dry_run and to_delete:
            print(f"  Deleting {len(to_delete):,} objects...", flush=True)
            deleted = 0
            for uid in to_delete:
                col.data.delete_by_id(uid)
                deleted += 1
                if deleted % 5000 == 0:
                    elapsed = time.time() - t_start
                    print(f"    {deleted:,}/{len(to_delete):,}...", flush=True)
            print(f"  Deleted {deleted:,} objects")

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sqlite-only", action="store_true")
    parser.add_argument("--weaviate-only", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN ===\n")

    if not args.weaviate_only:
        cleanup_sqlite(args.dry_run)

    if not args.sqlite_only:
        cleanup_weaviate(args.dry_run)

    if args.dry_run:
        print("\nNo changes made. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
