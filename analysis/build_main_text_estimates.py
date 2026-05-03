"""Refresh `analysis/main_text_estimates.csv` from the latest posterior traces.

Recomputes every quantity cited in the main text from the canonical .nc traces
in `analysis/results/prevalence/`. Differences between years and averages
across years are computed at the *posterior-sample* level (median of the
distribution of differences, not difference of medians) so the CrIs are
correctly propagated.

Inputs:  analysis/results/prevalence/trace_*.nc
Outputs: analysis/main_text_estimates.csv

Usage:
    python analysis/build_main_text_estimates.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "analysis" / "results" / "prevalence"
OUT = ROOT / "analysis" / "main_text_estimates.csv"


def summarize(samples: np.ndarray, scale: float = 100.0) -> tuple[float, float, float]:
    flat = samples.flatten() * scale
    return float(np.median(flat)), float(np.quantile(flat, 0.025)), float(np.quantile(flat, 0.975))


def main() -> None:
    rows: list[dict] = []

    # --- Region/year traces (Models 1 & 4) ---
    hecc_ry = az.from_netcdf(RESULTS / "trace_health_effects_of_climate_change_region_yearly.nc")
    health_ry = az.from_netcdf(RESULTS / "trace_health_region_yearly.nc")

    # prevalence_time has dim (chain, draw, prevalence_time_dim_0) where index 0..4 maps to 2021..2025
    hecc_prev_time = hecc_ry.posterior["prevalence_time"]   # 4 chains × 4000 draws × 5 years
    health_prev_time = health_ry.posterior["prevalence_time"]
    years = [2021, 2022, 2023, 2024, 2025]

    # Health 2025
    s = summarize(health_prev_time.isel({health_prev_time.dims[-1]: years.index(2025)}).values)
    rows.append({"quantity": "% of climate-change articles discussing health, 2025",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    # HECC 2025
    hecc_2025 = hecc_prev_time.isel({hecc_prev_time.dims[-1]: years.index(2025)}).values
    s = summarize(hecc_2025)
    rows.append({"quantity": "% of climate-change articles addressing HECC, 2025",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    # HECC 2021
    hecc_2021 = hecc_prev_time.isel({hecc_prev_time.dims[-1]: years.index(2021)}).values
    s = summarize(hecc_2021)
    rows.append({"quantity": "% of climate-change articles addressing HECC, 2021",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    # HECC change 2021->2025 — compute at sample level
    s = summarize(hecc_2025 - hecc_2021)
    rows.append({"quantity": "HECC change, 2021-2025 (percentage points)",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    # --- Climate-zone × month trace (Model 2) ---
    hecc_zm = az.from_netcdf(RESULTS / "trace_health_effects_of_climate_change_climate_zone_monthly.nc")
    pgt = hecc_zm.posterior["prevalence_group_time"]    # chain × draw × group × time
    # Need group and time labels — read them from the corresponding CSV
    pgt_df = pd.read_csv(RESULTS / "prevalence_group_time_health_effects_of_climate_change_climate_zone_monthly.csv")
    groups = sorted(pgt_df["climate_zone"].unique())
    months = sorted(pgt_df["time_period"].unique())     # YYYY-MM, 2021-01..2025-12

    g_dim, t_dim = pgt.dims[-2], pgt.dims[-1]
    pgt_arr = pgt.values   # shape (chain, draw, n_groups, n_months)

    def avg_month(zone: str, month: int) -> tuple[float, float, float]:
        gi = groups.index(zone)
        cols = [i for i, m in enumerate(months) if int(m.split("-")[1]) == month]
        return summarize(pgt_arr[:, :, gi, cols].mean(axis=-1))

    def avg_djf(zone: str) -> tuple[float, float, float]:
        gi = groups.index(zone)
        cols = [i for i, m in enumerate(months) if int(m.split("-")[1]) in (12, 1, 2)]
        return summarize(pgt_arr[:, :, gi, cols].mean(axis=-1))

    s = avg_month("Northern temperate", 7)
    rows.append({"quantity": "HECC, Northern temperate, July (avg 2021-2025)",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})
    s = avg_djf("Northern temperate")
    rows.append({"quantity": "HECC, Northern temperate, DJF winter (avg 2021-2025)",
                 "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})
    for m, name in [(5, "May"), (6, "June"), (7, "July"), (8, "August")]:
        s = avg_month("Tropical", m)
        rows.append({"quantity": f"HECC, Tropical, {name} (avg 2021-2025)",
                     "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    # --- HDI 2025 group estimates (Model 3) ---
    hecc_hdi = az.from_netcdf(RESULTS / "trace_health_effects_of_climate_change_hdi_category.nc")
    pg = hecc_hdi.posterior["prevalence_group"]
    hdi_df = pd.read_csv(RESULTS / "prevalence_group_health_effects_of_climate_change_hdi_category.csv", index_col=0)
    hdi_groups = list(hdi_df.index)   # ['High', 'Low', 'Medium', 'Very High']
    pg_arr = pg.values   # chain × draw × group

    label_map = {"Very High": "Very High HDI countries", "High": "High HDI countries",
                 "Medium": "Medium HDI countries", "Low": "Low HDI countries"}
    for hdi in ["Very High", "High", "Medium", "Low"]:
        gi = hdi_groups.index(hdi)
        s = summarize(pg_arr[:, :, gi])
        rows.append({"quantity": f"HECC, {label_map[hdi]}",
                     "estimate_pct": round(s[0], 1), "cri_2.5_pct": round(s[1], 1), "cri_97.5_pct": round(s[2], 1)})

    out = pd.DataFrame(rows, columns=["quantity", "estimate_pct", "cri_2.5_pct", "cri_97.5_pct"])
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
