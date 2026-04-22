"""Lattice plot of monthly climate-change coverage and health prevalence.

Panel A: monthly count of climate-change articles (time series, with COP peaks
         labelled).
Panel B: monthly stacked proportion of CC articles — HECC / CC (orange) and
         (Health - HECC) / CC (grey). Top of stack = Health / CC.

Reads the pre-aggregated monthly CSV produced by `build_plot_inputs.py`, so
the full article-level parquet isn't needed at plot time.

Inputs
------
- analysis/plot_inputs/yearly_trends_monthly.csv

Outputs
-------
- analysis/figures/yearly_trends.pdf
- analysis/figures/yearly_trends.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
DATA = ANALYSIS_DIR / "plot_inputs" / "yearly_trends_monthly.csv"
OUTDIR = ANALYSIS_DIR / "figures"
OUTDIR.mkdir(exist_ok=True)


def load_monthly() -> pd.DataFrame:
    m = pd.read_csv(DATA, parse_dates=["month"]).set_index("month").sort_index()
    m["health_only"] = m["n_cc_health"] - m["n_cc_hecc"]
    m["pct_hecc"] = 100 * m["n_cc_hecc"] / m["n_cc"]
    m["pct_health_only"] = 100 * m["health_only"] / m["n_cc"]
    return m


def main() -> None:
    m = load_monthly()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )

    # Panel A: monthly CC counts, y-axis not from zero
    ax_top.plot(m.index, m["n_cc"] / 1000, color="#3B3B3B", lw=1.6, marker="o", markersize=3)
    ax_top.set_ylabel("Climate-change articles\n(thousands / month)")
    ax_top.set_title("A. Monthly volume of climate-change coverage", loc="left", fontsize=11)
    ax_top.spines[["top", "right"]].set_visible(False)
    lo, hi = m["n_cc"].min() / 1000, m["n_cc"].max() / 1000
    pad = 0.08 * (hi - lo)
    ax_top.set_ylim(lo - pad, hi + pad * 4)

    # Label COP peaks with location
    cop_peaks = [
        ("COP26\n(Glasgow)", "2021-11-01"),
        ("COP27\n(Sharm El-Sheikh)", "2022-11-01"),
        ("COP28\n(Dubai)", "2023-12-01"),
        ("COP29\n(Baku)", "2024-11-01"),
        ("COP30\n(Belém)", "2025-11-01"),
    ]
    for label, date in cop_peaks:
        ts = pd.Timestamp(date)
        y = m.loc[ts, "n_cc"] / 1000
        ax_top.annotate(
            label,
            xy=(ts, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#C26A1E",
            fontweight="bold",
        )

    # Panel B: stacked area — HECC/CC (grey bottom) and (Health - HECC)/CC (orange top)
    ax_bot.stackplot(
        m.index,
        m["pct_hecc"],
        m["pct_health_only"],
        labels=["HECC / CC", "Health (excl. HECC) / CC"],
        colors=["#BFBFBF", "#E68A2E"],
        alpha=0.9,
        linewidth=0,
    )
    ax_bot.set_ylabel("Proportion of CC articles (%)")
    ax_bot.set_xlabel("Month")
    ax_bot.set_title(
        "B. Health framing as a share of climate-change coverage", loc="left", fontsize=11
    )
    ax_bot.legend(frameon=False, loc="upper left")
    ax_bot.spines[["top", "right"]].set_visible(False)
    stack_max = (m["pct_hecc"] + m["pct_health_only"]).max()
    ax_bot.set_ylim(0, stack_max + 6)

    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"yearly_trends.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUTDIR / 'yearly_trends.pdf'} and .png")


if __name__ == "__main__":
    main()
