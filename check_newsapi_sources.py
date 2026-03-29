#!/usr/bin/env python3
"""Check if bad scrapai sources are available via NewsAPI (Event Registry).

Looks up each source URI and counts articles since 2020.

Usage:
    python check_newsapi_sources.py
    python check_newsapi_sources.py --output results/newsapi_availability.csv
"""

import argparse
import csv
import os
import time

from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter

load_dotenv()

# 25 bad scrapai sources with no current newsapi replacement
BAD_SOURCES = [
    "5plus.mu",
    "aftonbladet.se",
    "aif.ru",
    "chosun.com",
    "dn.no",
    "echoroukonline.com",
    "elcomercio.com",
    "ennaharonline.com",
    "ft.com",
    "guardian.ng",
    "ilsole24ore.com",
    "inquirer.net",
    "kleinezeitung.at",
    "lanacion.cl",
    "lequipe.fr",
    "liberation.fr",
    "lidovky.cz",
    "mmbiztoday.com",
    "nst.com.my",
    "sol.sapo.pt",
    "telegraaf.nl",
    "thechronicle.com.gh",
    "thisdaylive.com",
    "vanguardngr.com",
    "vg.no",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/newsapi_availability.csv")
    args = parser.parse_args()

    er = EventRegistry(apiKey=os.environ["NEWSAPI_KEY"])

    results = []
    print(f"Checking {len(BAD_SOURCES)} sources in NewsAPI...\n")
    print(f"  {'source':30s} {'uri':35s} {'count':>10s}  status")
    print("  " + "-" * 85)

    for source in BAD_SOURCES:
        uri = er.getSourceUri(source)
        time.sleep(0.3)

        count = 0
        if uri:
            q = QueryArticlesIter(sourceUri=uri, dateStart="2020-01-01")
            count = q.count(er)
            time.sleep(0.3)

        status = "available" if count > 0 else "not found"
        print(f"  {source:30s} {str(uri or ''):35s} {count:>10,}  {status}")

        results.append({
            "source": source,
            "newsapi_uri": uri or "",
            "articles_since_2020": count,
            "status": status,
        })

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    available = [r for r in results if r["articles_since_2020"] > 0]
    print(f"\nAvailable in NewsAPI: {len(available)}/{len(BAD_SOURCES)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
