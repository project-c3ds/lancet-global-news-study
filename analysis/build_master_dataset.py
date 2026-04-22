"""Build the published master dataset: one row per (source_uri, country, month).

Aggregates the 4-label classifications + BM25 + covariates into a single
human-readable CSV that is the data release companion to the paper. All
downstream estimation inputs are derived from this file.

Inputs
------
- `data/climate_articles_with_classifications.parquet` — unfiltered 2.2M-row
  article-level parquet with 4-label classifications. Used so that `n_total`
  counts *all* articles (not only those with at least one label True).
- `analysis/country_covariates.csv` — per-country covariates (75 sample
  countries × region/climate zone/HDI/CRI).

Output
------
- `analysis/corpus_monthly.csv` — 1 row per (source_uri, country, month).

Columns
-------
    source_uri     str
    country        str   (75-country sample, `GB` remapped to `United Kingdom`)
    month          str   (`YYYY-MM`, 2021-01 through 2025-12)
    un_region      str
    climate_zone   str
    hdi_category   str   (may be empty for Taiwan)
    hdi_value      float (may be NaN for Taiwan)
    n_total        int   all articles in the cell
    n_cc           int   articles with climate_change=True
    k_hecc_cc      int   articles with climate_change=True AND HECC=True
    k_health_cc    int   articles with climate_change=True AND health=True

Usage:
    python analysis/build_master_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "climate_articles_with_classifications.parquet"
DEFAULT_COVARIATES = ROOT / "analysis" / "country_covariates.csv"
DEFAULT_OUTPUT = ROOT / "analysis" / "corpus_monthly.csv"

COVARIATE_COLS = ["un_region", "climate_zone", "hdi_category", "hdi_value"]


def build(input_path: Path, covariates_path: Path, output_path: Path) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_parquet(
        input_path,
        columns=[
            "source_uri", "country", "published_date",
            "climate_change", "health", "health_effects_of_climate_change",
        ],
    )
    print(f"  {len(df):,} rows")

    df.loc[df["country"] == "GB", "country"] = "United Kingdom"

    print(f"Reading {covariates_path} ...")
    covariates = pd.read_csv(covariates_path)[["country", *COVARIATE_COLS]]
    sample_countries = set(covariates["country"])
    print(f"  {len(covariates)} sample countries")

    print("Filtering: country in sample, 2021-2025 ...")
    df = df[df["country"].isin(sample_countries)].copy()
    ts = pd.to_datetime(df["published_date"], utc=True, format="mixed", errors="coerce")
    df["month"] = ts.dt.strftime("%Y-%m")
    df = df.dropna(subset=["month"])
    df = df[df["month"].between("2021-01", "2025-12")]
    print(f"  {len(df):,} rows after filter")

    for col in ["climate_change", "health", "health_effects_of_climate_change"]:
        df[col] = df[col].astype(bool)

    df["_k_hecc_cc"] = (df["climate_change"] & df["health_effects_of_climate_change"]).astype(int)
    df["_k_health_cc"] = (df["climate_change"] & df["health"]).astype(int)
    df["_n_cc"] = df["climate_change"].astype(int)

    print("Aggregating to (source_uri, country, month) ...")
    agg = df.groupby(["source_uri", "country", "month"], as_index=False).agg(
        n_total=("climate_change", "size"),
        n_cc=("_n_cc", "sum"),
        k_hecc_cc=("_k_hecc_cc", "sum"),
        k_health_cc=("_k_health_cc", "sum"),
    )

    print("Merging country covariates ...")
    agg = agg.merge(covariates, on="country", how="left")
    column_order = [
        "source_uri", "country", "month",
        *COVARIATE_COLS,
        "n_total", "n_cc", "k_hecc_cc", "k_health_cc",
    ]
    agg = agg[column_order].sort_values(["country", "source_uri", "month"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    sz = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path.relative_to(ROOT)}  ({len(agg):,} rows, {sz:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--covariates", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.input, args.covariates, args.output)


if __name__ == "__main__":
    main()
