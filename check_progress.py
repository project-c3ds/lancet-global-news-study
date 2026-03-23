#!/usr/bin/env python3
"""Check embedding pipeline progress."""

import gzip
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "embeddings"
PREMIUM_GLOBS = ["world_news_premium", "world_news_premium_2", "world_news_premium_3"]

# Count total sources
total_sources = 0
for folder in PREMIUM_GLOBS:
    for d in (DATA_DIR / folder).iterdir():
        if (d / "crawls").is_dir():
            total_sources += 1

# Count completed (both .jsonl and .jsonl.gz)
seen = set()
done_articles = 0
total_bytes = 0

for f in sorted(OUTPUT_DIR.glob("*.jsonl")):
    name = f.stem
    seen.add(name)
    total_bytes += f.stat().st_size
    with open(f) as fh:
        done_articles += sum(1 for _ in fh)

for f in sorted(OUTPUT_DIR.glob("*.jsonl.gz")):
    name = f.name.removesuffix(".jsonl.gz")
    if name in seen:
        continue
    seen.add(name)
    total_bytes += f.stat().st_size
    try:
        with gzip.open(f, "rt") as fh:
            done_articles += sum(1 for _ in fh)
    except EOFError:
        pass  # in-progress file

done_sources = len(seen)
print(f"Sources:  {done_sources}/{total_sources} ({done_sources/total_sources*100:.0f}%)")
print(f"Articles: {done_articles:,}")
print(f"Disk:     {total_bytes / 1024**3:.1f} GB")

# Show last 5 log lines
log_file = OUTPUT_DIR / "progress.log"
if log_file.exists():
    lines = log_file.read_text().strip().splitlines()
    print(f"\nLast log entries:")
    for line in lines[-5:]:
        print(f"  {line}")
