"""Build analysis/analysis_data.parquet from the classified corpus + country covariates.

Joins three inputs:

1. `data/climate.db` — article metadata and BM25 scores (from the collection +
   BM25 scoring steps).
2. `data/classifications_slim.db` — 4-label classifications produced by the
   distilled model (`classification/classify_offline.py`).
3. `analysis/country_covariates.csv` — per-country UN region, HDI category,
   climate zone, and Climate Risk Index ranks for the 75 sample countries.

Applies the selection rule used throughout the paper:

- country in the 75-country sample (rows with empty country or `'GB'` raw are
  dropped / remapped to `'United Kingdom'`)
- `published_date` in 2021-2025
- at least one of the four labels is True

Writes `analysis/analysis_data.parquet` (~760K rows, 26 columns). This is the
dataset consumed by `estimation/estimate.sh`.

Usage:
    python analysis/build_analysis_dataset.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLIMATE_DB = ROOT / "data" / "climate.db"
CLASSIFICATIONS_DB = ROOT / "data" / "classifications_slim.db"
COVARIATES_CSV = ROOT / "analysis" / "country_covariates.csv"
OUTPUT_PARQUET = ROOT / "analysis" / "analysis_data.parquet"

LABELS = [
    "climate_change",
    "health",
    "health_effects_of_climate_change",
    "health_effects_of_extreme_weather",
]


def load_articles() -> pd.DataFrame:
    conn = sqlite3.connect(CLIMATE_DB)
    df = pd.read_sql(
        "SELECT id, url, title, source_uri, language, published_date, "
        "extracted_at, collection_method, country, bm25_climate, bm25_health, bm25_avg "
        "FROM articles",
        conn,
    )
    conn.close()
    df.loc[df["country"] == "GB", "country"] = "United Kingdom"
    return df


def load_classifications() -> pd.DataFrame:
    conn = sqlite3.connect(CLASSIFICATIONS_DB)
    df = pd.read_sql(f"SELECT id, {', '.join(LABELS)} FROM classifications", conn)
    conn.close()
    for col in LABELS:
        df[col] = df[col].astype(bool)
    return df


def build(output: Path) -> None:
    print(f"Reading metadata from {CLIMATE_DB} ...")
    articles = load_articles()
    print(f"  {len(articles):,} articles")

    print(f"Reading classifications from {CLASSIFICATIONS_DB} ...")
    classifications = load_classifications()
    print(f"  {len(classifications):,} classifications")

    print("Merging articles ⨝ classifications ...")
    df = articles.merge(classifications, on="id", how="inner")
    print(f"  {len(df):,} rows after join")

    print(f"Loading covariates from {COVARIATES_CSV} ...")
    covariates = pd.read_csv(COVARIATES_CSV)
    sample_countries = set(covariates["country"])
    print(f"  {len(covariates)} sample countries")

    print("Applying selection: country in sample, 2021-2025, any label True ...")
    any_label = df[LABELS].any(axis=1)
    in_country = df["country"].isin(sample_countries)
    year = df["published_date"].str[:4]
    in_year = year.isin({"2021", "2022", "2023", "2024", "2025"})
    df = df[any_label & in_country & in_year].copy()
    print(f"  {len(df):,} rows after selection")

    print("Merging country covariates ...")
    df = df.merge(covariates, on="country", how="left")

    print(f"Writing {output} ...")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT_PARQUET)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
