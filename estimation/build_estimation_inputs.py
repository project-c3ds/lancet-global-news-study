"""Derive the 4 estimation-input CSVs from the published master dataset.

Reads `analysis/corpus_monthly.csv` (one row per source_uri × country × month
with `n_total`, `n_cc`, `k_hecc_cc`, `k_health_cc` + country covariates) and
writes four per-analysis CSVs with the estimator's expected schema:

    source, country, group, [time_period,] k, n

Each row is a per-source sufficient statistic — `k` positive articles in the
cell out of `n` trials — that the Binomial likelihood consumes directly.

Analyses produced
-----------------
    prev_hecc_cc_region_yearly        HECC | CC, yearly × LC region
    prev_hecc_cc_climate_zone_monthly HECC | CC, monthly × climate zone
    prev_hecc_cc_hdi_category         HECC | CC, HDI 2025 group (no time)
    prev_health_cc_region_yearly      Health | CC, yearly × LC region

Region grouping is the 2026 Lancet Countdown LC Grouping
(Africa / Asia / Europe / Latin America / Northern America / Oceania / SIDS).
HDI grouping is the 2025 HDI Group (Low / Medium / High / Very High); rows
with empty `hdi_2025` (Taiwan, Hong Kong) are dropped from the HDI input.

(HEEW is not estimated — it exists only as a classification label, to keep
articles about extreme-weather health impacts out of HECC. See
`classification/README.md`.)

Usage:
    python estimation/build_estimation_inputs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "analysis" / "corpus_monthly.csv"
DEFAULT_OUTDIR = ROOT / "estimation" / "estimation_inputs"


def _year(month_str: pd.Series) -> pd.Series:
    return month_str.str.slice(0, 4)


def derive(master: pd.DataFrame, out: Path) -> None:
    # Build derived frames. Each result uses the estimator's column vocabulary:
    # source, country, group, [time_period,] k, n.

    # Analysis 1 — HECC | CC, yearly × LC region
    a = master.assign(time_period=_year(master["month"]))
    a = a.groupby(["source_uri", "country", "lc_region", "time_period"], as_index=False).agg(
        k=("k_hecc_cc", "sum"), n=("n_cc", "sum"),
    )
    a = a[a["n"] > 0]
    a = a.rename(columns={"source_uri": "source", "lc_region": "group"})
    a[["source", "country", "group", "time_period", "k", "n"]].to_csv(
        out / "prev_hecc_cc_region_yearly.csv", index=False,
    )

    # Analysis 2 — HECC | CC, monthly × climate zone
    b = master.groupby(["source_uri", "country", "climate_zone", "month"], as_index=False).agg(
        k=("k_hecc_cc", "sum"), n=("n_cc", "sum"),
    )
    b = b[b["n"] > 0]
    b = b.rename(columns={"source_uri": "source", "climate_zone": "group", "month": "time_period"})
    b[["source", "country", "group", "time_period", "k", "n"]].to_csv(
        out / "prev_hecc_cc_climate_zone_monthly.csv", index=False,
    )

    # Analysis 3 — HECC | CC, HDI 2025 group (no time)
    c = master.dropna(subset=["hdi_2025"])
    c = c[c["hdi_2025"] != ""]
    c = c.groupby(["source_uri", "country", "hdi_2025"], as_index=False).agg(
        k=("k_hecc_cc", "sum"), n=("n_cc", "sum"),
    )
    c = c[c["n"] > 0]
    c = c.rename(columns={"source_uri": "source", "hdi_2025": "group"})
    c[["source", "country", "group", "k", "n"]].to_csv(
        out / "prev_hecc_cc_hdi_category.csv", index=False,
    )

    # Analysis 5 — Health | CC, yearly × LC region
    e = master.assign(time_period=_year(master["month"]))
    e = e.groupby(["source_uri", "country", "lc_region", "time_period"], as_index=False).agg(
        k=("k_health_cc", "sum"), n=("n_cc", "sum"),
    )
    e = e[e["n"] > 0]
    e = e.rename(columns={"source_uri": "source", "lc_region": "group"})
    e[["source", "country", "group", "time_period", "k", "n"]].to_csv(
        out / "prev_health_cc_region_yearly.csv", index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    print(f"Reading master from {args.input} ...")
    master = pd.read_csv(args.input, dtype={"month": str})
    print(f"  {len(master):,} rows")

    args.out.mkdir(parents=True, exist_ok=True)
    derive(master, args.out)
    print()
    for p in sorted(args.out.glob("*.csv")):
        nrows = sum(1 for _ in p.open()) - 1
        sz = p.stat().st_size / 1024
        print(f"  {p.relative_to(ROOT)}  ({nrows:,} rows, {sz:.1f} KB)")


if __name__ == "__main__":
    main()
