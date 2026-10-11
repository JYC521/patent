#!/usr/bin/env python3
"""
src/03_summarize_representation.py

Computes representation statistics for the 2010-2013 Fortune 500 utility patent cohort.

Outputs:
  - data/interim/summary_industry_representation.csv
  - data/interim/summary_company_representation.csv
  - Formatted markdown tables in console logging for:
      * Top Sectors by patent volume and company representation
      * Top Industries by patent volume and company representation
      * Underrepresented Industries (< 3 companies or < 50 patents)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def setup_logger() -> logging.Logger:
    """Configures structured console logging."""
    logger = logging.getLogger("03_summarize_representation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def format_markdown_table(df: pd.DataFrame) -> str:
    """Generates a clean GitHub-flavored markdown table from a DataFrame."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, (int, float)):
                if isinstance(val, float):
                    row_str.append(f"{val:,.1f}")
                else:
                    row_str.append(f"{val:,}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def generate_representation_summaries(
    cohort_assignees_path: Path,
    cohort_patents_path: Path,
    output_industry_csv: Path,
    output_company_csv: Path,
    logger: logging.Logger,
) -> None:
    logger.info("=" * 60)
    logger.info("COMPUTING REPRESENTATION STATISTICS & DESCRIPTIVE SUMMARIES")
    logger.info("=" * 60)

    # 1. Load Parquet tables
    logger.info(f"Loading {cohort_assignees_path}...")
    df_assignees = pq.read_table(cohort_assignees_path).to_pandas()
    logger.info(f"Loaded {len(df_assignees):,} patent-assignee links.")

    logger.info(f"Loading {cohort_patents_path}...")
    df_patents = pq.read_table(cohort_patents_path, columns=["patent_id", "patent_date"]).to_pandas()
    df_patents["grant_year"] = df_patents["patent_date"].astype(str).str[:4]
    logger.info(f"Loaded {len(df_patents):,} unique cohort patents.")

    # Merge patent dates onto assignee pairs
    merged = pd.merge(df_assignees, df_patents[["patent_id", "grant_year"]], on="patent_id", how="inner")
    logger.info(f"Merged cohort shape: {merged.shape}")

    # ==================================================================
    # 1. Sector & Industry Overview
    # ==================================================================
    ind_grp = merged.groupby(["sector", "industry"])
    industry_summary = ind_grp.agg(
        num_companies=("f500_company_name", "nunique"),
        total_patents=("patent_id", "nunique"),
    ).reset_index()

    industry_summary["avg_patents_per_company"] = (
        industry_summary["total_patents"] / industry_summary["num_companies"]
    ).round(2)

    industry_summary = industry_summary.sort_values(
        by=["total_patents", "num_companies"], ascending=[False, False]
    ).reset_index(drop=True)

    output_industry_csv.parent.mkdir(parents=True, exist_ok=True)
    industry_summary.to_csv(output_industry_csv, index=False)
    logger.info(f"Saved Industry Overview to: {output_industry_csv}")

    # Sector aggregation
    sec_grp = merged.groupby("sector")
    sector_summary = sec_grp.agg(
        num_companies=("f500_company_name", "nunique"),
        total_patents=("patent_id", "nunique"),
    ).reset_index()
    sector_summary["avg_patents_per_company"] = (
        sector_summary["total_patents"] / sector_summary["num_companies"]
    ).round(2)
    sector_summary = sector_summary.sort_values(by="total_patents", ascending=False).reset_index(drop=True)

    # ==================================================================
    # 2. Company-Level Breakdown by Grant Year
    # ==================================================================
    comp_year = merged.groupby(["f500_company_name", "industry", "sector", "grant_year"])["patent_id"].nunique().unstack(fill_value=0)
    for yr in ["2010", "2011", "2012", "2013"]:
        if yr not in comp_year.columns:
            comp_year[yr] = 0

    comp_year["Total"] = comp_year[["2010", "2011", "2012", "2013"]].sum(axis=1)
    company_summary = comp_year.reset_index()[
        ["f500_company_name", "industry", "sector", "2010", "2011", "2012", "2013", "Total"]
    ]
    company_summary = company_summary.sort_values(by="Total", ascending=False).reset_index(drop=True)

    output_company_csv.parent.mkdir(parents=True, exist_ok=True)
    company_summary.to_csv(output_company_csv, index=False)
    logger.info(f"Saved Company Breakdown to: {output_company_csv}")

    # ==================================================================
    # 3. Console Logging & Markdown Tables
    # ==================================================================
    print("\n" + "=" * 70)
    print("### TOP SECTORS (by Utility Patent Volume 2010–2013)")
    print("=" * 70)
    print(format_markdown_table(sector_summary))
    print()

    print("=" * 70)
    print("### TOP 15 INDUSTRIES (by Utility Patent Volume 2010–2013)")
    print("=" * 70)
    print(format_markdown_table(industry_summary.head(15)))
    print()

    # Underrepresented Industries
    underrepresented = industry_summary[
        (industry_summary["num_companies"] < 3) | (industry_summary["total_patents"] < 50)
    ].sort_values(by="total_patents", ascending=True)

    print("=" * 70)
    print("### UNDERREPRESENTED INDUSTRIES (< 3 Companies or < 50 Total Patents)")
    print("=" * 70)
    print(f"Total underrepresented industries identified: {len(underrepresented)}")
    print(format_markdown_table(underrepresented))
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Representation Statistics for F500 Patent Cohort")
    parser.add_argument(
        "--cohort-assignees",
        type=Path,
        default=Path("data/interim/cohort_patent_assignees.parquet"),
        help="Path to cohort_patent_assignees.parquet",
    )
    parser.add_argument(
        "--cohort-patents",
        type=Path,
        default=Path("data/interim/cohort_patents.parquet"),
        help="Path to cohort_patents.parquet",
    )
    parser.add_argument(
        "--output-industry-csv",
        type=Path,
        default=Path("data/interim/summary_industry_representation.csv"),
        help="Path for summary_industry_representation.csv",
    )
    parser.add_argument(
        "--output-company-csv",
        type=Path,
        default=Path("data/interim/summary_company_representation.csv"),
        help="Path for summary_company_representation.csv",
    )

    args = parser.parse_args()
    logger = setup_logger()
    generate_representation_summaries(
        cohort_assignees_path=args.cohort_assignees,
        cohort_patents_path=args.cohort_patents,
        output_industry_csv=args.output_industry_csv,
        output_company_csv=args.output_company_csv,
        logger=logger,
    )


if __name__ == "__main__":
    main()
