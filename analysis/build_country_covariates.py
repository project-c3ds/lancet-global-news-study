"""Merge the 2026 LC guidance country names and groupings onto our 75-country
covariates table.

Reads
-----
- `backup/2026 Guidance_Country_Names_and_Groupings.csv` — official Lancet
  Countdown guidance: ISO3, formal country name, LC Grouping, WHO Region,
  HDI Group 2025. (Stored in `backup/` because that directory is gitignored;
  the file is upstream input, not a tracked artifact.)
- `analysis/country_covariates.csv` — existing per-country covariates (75
  sample countries, climate zone, HDI, CRI ranks, original `un_region`).

Writes
------
- `analysis/country_covariates.csv` (in-place) with four added columns:
    lc_country_name   formal name from guidance (e.g. "United States of America")
    lc_region         LC Grouping (Africa / Asia / Europe / Latin America /
                      Northern America / Oceania / SIDS)
    who_region        WHO Region
    hdi_2025          2025 HDI Group (Low / Medium / High / Very High)

Manual overrides
----------------
- Taiwan (TWN) is not in the LC guidance; we keep it in the corpus with
  lc_country_name="Taiwan", lc_region="Asia", who_region="Western Pacific",
  hdi_2025="" (blank → excluded from HDI analysis).
- Hong Kong (HKG) is listed as #N/A in the guidance HDI column. We set
  hdi_2025="" so HK is excluded from the HDI analysis.

Usage:
    python analysis/build_country_covariates.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GUIDANCE = ROOT / "backup" / "2026 Guidance_Country_Names_and_Groupings.csv"
DEFAULT_COVARIATES = ROOT / "analysis" / "country_covariates.csv"

NEW_COLS = ["lc_country_name", "lc_region", "who_region", "hdi_2025"]

TAIWAN_OVERRIDE = {
    "lc_country_name": "Taiwan",
    "lc_region": "Asia",
    "who_region": "Western Pacific",
    "hdi_2025": "",
}


def build(guidance_path: Path, covariates_path: Path) -> None:
    print(f"Reading guidance from {guidance_path} ...")
    g = pd.read_csv(guidance_path).dropna(subset=["ISO3"])
    g = g.rename(columns={
        "ISO3": "iso3",
        "Country Name to use": "lc_country_name",
        "LC Grouping": "lc_region",
        "WHO Region": "who_region",
        "HDI Group 2025": "hdi_2025",
    })[["iso3", "lc_country_name", "lc_region", "who_region", "hdi_2025"]]
    print(f"  {len(g)} country rows in guidance")

    print(f"Reading covariates from {covariates_path} ...")
    cov = pd.read_csv(covariates_path)
    print(f"  {len(cov)} covariate rows")

    # Drop any pre-existing LC columns so this script is idempotent.
    cov = cov.drop(columns=[c for c in NEW_COLS if c in cov.columns])

    merged = cov.merge(g, on="iso3", how="left")

    # Manual override: Taiwan not in guidance → keep with explicit values.
    twn = merged["iso3"] == "TWN"
    if twn.any():
        for col, val in TAIWAN_OVERRIDE.items():
            merged.loc[twn, col] = val
        print(f"  applied Taiwan override (n={twn.sum()})")

    # Manual override: Hong Kong HDI → missing (guidance lists #N/A as territory;
    # we use blank so HK is excluded from the HDI prevalence analysis).
    hkg = merged["iso3"] == "HKG"
    if hkg.any():
        merged.loc[hkg, "hdi_2025"] = ""
        print(f"  applied Hong Kong HDI=missing override (n={hkg.sum()})")

    # Treat the literal string "#N/A" coming from the guidance as missing too.
    merged["hdi_2025"] = merged["hdi_2025"].replace({"#N/A": ""}).fillna("")

    unmapped = merged[merged["lc_region"].isna()]
    if not unmapped.empty:
        raise SystemExit(f"Countries with no lc_region after merge:\n{unmapped[['country','iso3']]}")

    merged.to_csv(covariates_path, index=False)
    print(f"\nWrote {covariates_path.relative_to(ROOT)} ({len(merged)} rows, {len(merged.columns)} cols)")
    print(f"  columns: {list(merged.columns)}")
    print()
    print("LC region distribution:")
    print(merged["lc_region"].value_counts().to_string())
    print()
    print("HDI 2025 distribution (blank = excluded):")
    print(merged["hdi_2025"].replace("", "<missing>").value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guidance", type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument("--covariates", type=Path, default=DEFAULT_COVARIATES)
    args = parser.parse_args()
    build(args.guidance, args.covariates)


if __name__ == "__main__":
    main()
