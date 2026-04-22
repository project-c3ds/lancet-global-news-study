# Collection

Scripts for collecting the 2.2M-article corpus across 75 countries. Collection proceeds in two stages:

1. **NewsAPI (≈89% of corpus)** — keyword-based queries in the source language, using the translated keyword set in `../translations/`. Sources indexed by [NewsAPI.ai / Event Registry](https://newsapi.ai/).
2. **ScrapAI (≈11% of corpus)** — custom crawler for sources not indexed by NewsAPI. See https://docs.scrapai.dev/introduction. ScrapAI output lands in its own SQLite (`data/articles.db`) and is then filtered+ingested into `data/climate.db`.

## Pipeline

### NewsAPI path

1. `collect_newsapi_climate.py --year <YEAR>` — query Event Registry for each source × climate keywords; writes `data/newsapi_articles_{year}/{source_uri}_climate.jsonl`. Invoke once per year (2021-2025).
2. `collect_newsapi_health.py --year <YEAR>` — separate second pass for health keywords; writes `data/newsapi_health_{year}/{source_uri}_health.jsonl`. Invoke once per year.
3. `ingest_newsapi_climate.py --year <YEAR>` — ingest the NewsAPI JSONLs into `data/climate.db`.

### ScrapAI path

1. ScrapAI output is stored separately in `data/articles.db` (built outside this repo).
2. `ingest_scrapai_climate.py` — filter articles in `articles.db` by climate keywords (in the article's language + English; word-boundary matching for Latin scripts, substring for CJK/Thai) and insert matches into `data/climate.db`.

### Shared

- `build_source_status.py` — tracks per-source collection status across both paths.

The NewsAPI collectors read `NEWSAPI_KEY` from `.env` and import the `eventregistry` SDK directly.
