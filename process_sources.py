"""
Process sources from top10_per_country.csv in three phases:
1. Resolve source URIs from website URLs
2. Collect 20 most recent articles per source (for metadata like language)
3. Count climate-related articles (2020-present)

Each phase checkpoints to disk so you can resume after interruption.
"""
import pandas as pd
import json
import os
import time
from newsapi import fetch_source_uri, collect_recent_articles, count_articles, CLIMATE_KEYWORDS_EN

INPUT_CSV = 'data/top10_per_country.csv'
OUTPUT_CSV = 'data/top10_per_country_processed.csv'
ARTICLES_DIR = 'data/recent_articles'
DELAY = 0.5  # seconds between API calls


def load_data():
    df = pd.read_csv(INPUT_CSV)
    # Load checkpoint if it exists
    if os.path.exists(OUTPUT_CSV):
        df_out = pd.read_csv(OUTPUT_CSV)
        return df_out
    # Initialize new columns
    df['source_uri'] = None
    df['uri_lookup_done'] = False
    df['articles_collected'] = None
    df['most_recent_date'] = None
    df['most_frequent_language'] = None
    df['language_counts'] = None
    df['recent_articles_done'] = False
    df['article_count'] = None
    df['count_done'] = False
    return df


def save_checkpoint(df):
    df.to_csv(OUTPUT_CSV, index=False)


def phase1_resolve_uris(df):
    """Resolve source_uri for each source from its website_url."""
    todo = df[df['uri_lookup_done'] != True]
    print(f"\n--- Phase 1: Resolve URIs ({len(todo)} remaining) ---")

    for idx, row in todo.iterrows():
        url = row['website_url']
        name = row['name']
        country = row['country']

        if pd.isna(url) or not url.strip():
            print(f"  SKIP {country}/{name} - no URL")
            df.at[idx, 'uri_lookup_done'] = True
            continue

        try:
            uri = fetch_source_uri(url)
            df.at[idx, 'source_uri'] = uri if uri else None
            df.at[idx, 'uri_lookup_done'] = True
            print(f"  {country}/{name} -> {uri}")
        except Exception as e:
            print(f"  ERROR {country}/{name}: {e}")
            df.at[idx, 'uri_lookup_done'] = True

        if idx % 50 == 0:
            save_checkpoint(df)
        time.sleep(DELAY)

    save_checkpoint(df)
    resolved = df['source_uri'].notna().sum()
    print(f"Phase 1 complete: {resolved}/{len(df)} URIs resolved")
    return df


def phase2_collect_recent(df):
    """Collect 20 most recent articles per source for metadata."""
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    todo = df[(df['source_uri'].notna()) & (df['recent_articles_done'] != True)]
    print(f"\n--- Phase 2: Collect recent articles ({len(todo)} remaining) ---")

    for idx, row in todo.iterrows():
        source_uri = row['source_uri']
        name = row['name']
        country = row['country']

        # Use a safe filename
        safe_name = f"{country}_{name}".replace('/', '_').replace(' ', '_')[:100]
        filepath = os.path.join(ARTICLES_DIR, f"{safe_name}.json")

        try:
            result = collect_recent_articles(source_uri, max_items=20)

            # Save full articles to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result['articles'], f, ensure_ascii=False, indent=2)

            # Save metadata to dataframe
            df.at[idx, 'articles_collected'] = result['articles_collected']
            df.at[idx, 'most_recent_date'] = result['most_recent_date']
            df.at[idx, 'most_frequent_language'] = result['most_frequent_language']
            df.at[idx, 'language_counts'] = json.dumps(result['language_counts'])
            df.at[idx, 'recent_articles_done'] = True
            print(f"  {country}/{name}: {result['articles_collected']}/20 articles, "
                  f"lang={result['most_frequent_language']}, "
                  f"latest={result['most_recent_date']}")
        except Exception as e:
            print(f"  ERROR {country}/{name}: {e}")
            df.at[idx, 'recent_articles_done'] = True

        if idx % 20 == 0:
            save_checkpoint(df)
        time.sleep(DELAY)

    save_checkpoint(df)
    done = df['recent_articles_done'].sum()
    print(f"Phase 2 complete: {done} sources processed")
    return df


def phase3_count_articles(df, keywords=CLIMATE_KEYWORDS_EN):
    """Count climate-related articles for each source (2020-present)."""
    todo = df[(df['source_uri'].notna()) & (df['count_done'] != True)]
    print(f"\n--- Phase 3: Count articles ({len(todo)} remaining) ---")

    for idx, row in todo.iterrows():
        source_uri = row['source_uri']
        name = row['name']
        country = row['country']

        try:
            count = count_articles(source_uri, keywords=keywords)
            df.at[idx, 'article_count'] = count
            df.at[idx, 'count_done'] = True
            print(f"  {country}/{name}: {count} articles")
        except Exception as e:
            print(f"  ERROR {country}/{name}: {e}")
            df.at[idx, 'count_done'] = True

        if idx % 50 == 0:
            save_checkpoint(df)
        time.sleep(DELAY)

    save_checkpoint(df)
    counted = df['article_count'].notna().sum()
    total_articles = df['article_count'].sum()
    print(f"Phase 3 complete: {counted} sources counted, {total_articles:.0f} total articles")
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process news sources from CSV')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only a specific phase (1=URIs, 2=recent articles, 3=counts)')
    args = parser.parse_args()

    df = load_data()

    if args.phase is None or args.phase == 1:
        df = phase1_resolve_uris(df)
    if args.phase is None or args.phase == 2:
        df = phase2_collect_recent(df)
    if args.phase is None or args.phase == 3:
        df = phase3_count_articles(df)

    print(f"\nResults saved to {OUTPUT_CSV}")
