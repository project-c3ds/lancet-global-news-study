"""Pre-aggregate monthly counts used by the appendix time-series figure.

Produces `analysis/plot_inputs/yearly_trends_monthly.csv` with columns:

    month      YYYY-MM-01 (timestamp)
    n_cc       climate-change articles that month
    n_cc_health  climate-change AND health articles
    n_cc_hecc  climate-change AND HECC articles

The plot script computes the two stacked bands as
`pct_hecc = n_cc_hecc / n_cc` and `pct_health_only = (n_cc_health - n_cc_hecc) / n_cc`.

Usage:
    python analysis/build_plot_inputs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "analysis" / "analysis_data.parquet"
DEFAULT_OUTPUT = ROOT / "analysis" / "plot_inputs" / "yearly_trends_monthly.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_parquet(
        args.input,
        columns=["published_date", "climate_change", "health", "health_effects_of_climate_change"],
    )
    print(f"  {len(df):,} rows")

    df = df[df["climate_change"].astype(bool)].copy()
    df["month"] = pd.to_datetime(df["published_date"], utc=True, format="mixed").dt.to_period("M").dt.to_timestamp()

    agg = df.groupby("month").agg(
        n_cc=("climate_change", "size"),
        n_cc_health=("health", "sum"),
        n_cc_hecc=("health_effects_of_climate_change", "sum"),
    ).reset_index()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output, index=False)
    print(f"  wrote {args.output.relative_to(ROOT)}  ({len(agg)} rows, {args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
