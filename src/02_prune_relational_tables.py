#!/usr/bin/env python3
"""
src/02_prune_relational_tables.py

Downstream Relational Pruning Pipeline.
Uses data/interim/cohort_patent_ids.parquet as a fast hash lookup filter to stream
and prune raw USPTO / PatentsView bulk data tables down to the target Fortune 500 analysis cohort.

Target Relational Tables:
  1. g_application.tsv.zip          -> data/interim/filtered_application.parquet
  2. g_cpc_current.tsv.zip          -> data/interim/filtered_cpc.parquet
  3. patent_document_stats.csv      -> data/interim/filtered_claim_stats.parquet
  4. g_foreign_citation.tsv.zip     -> data/interim/filtered_foreign_citations.parquet
  5. g_us_patent_citation.tsv.zip   -> data/interim/filtered_us_citations.parquet
  (optional: g_assignee_disambiguated.tsv.zip -> data/interim/filtered_assignees.parquet)
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def setup_logger() -> logging.Logger:
    """Configures structured console logging."""
    logger = logging.getLogger("02_prune_relational")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def resolve_file_stream(file_path: Path) -> Tuple[object, Optional[zipfile.ZipFile], int]:
    """
    Opens a file stream from either an uncompressed file or directly inside a .zip archive
    without extracting the archive to disk.
    """
    candidates = [
        file_path,
        file_path.with_suffix(""),
        file_path.with_name(file_path.name + ".zip"),
        file_path.parent / file_path.name.replace("_", " "),
        file_path.parent / (file_path.name.replace("_", " ") + ".zip"),
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".zip":
                zf = zipfile.ZipFile(p, "r")
                inner_names = [n for n in zf.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
                if not inner_names:
                    raise FileNotFoundError(f"No valid file found inside archive: {p}")
                info = zf.getinfo(inner_names[0])
                return zf.open(inner_names[0], "r"), zf, info.file_size
            else:
                return open(p, "r", encoding="utf-8", errors="replace"), None, p.stat().st_size

    raise FileNotFoundError(f"Neither '{file_path}' nor its zip/uncompressed alternative exists.")


def load_cohort_ids(cohort_ids_path: Path, logger: logging.Logger) -> Set[str]:
    """Loads cohort patent IDs into a high-speed Python set for O(1) membership lookups."""
    if not cohort_ids_path.exists():
        raise FileNotFoundError(f"Cohort patent IDs file not found: {cohort_ids_path}. Run 01_filter_cohort.py first.")
    
    logger.info(f"Loading cohort patent IDs from {cohort_ids_path}...")
    t0 = time.time()
    table = pq.read_table(cohort_ids_path, columns=["patent_id"])
    cohort_set = set(table.column("patent_id").to_pylist())
    logger.info(f"Loaded {len(cohort_set):,} unique cohort IDs in {time.time() - t0:.2f}s.")
    return cohort_set


# ----------------------------------------------------------------------
# Chunk Processing Handlers for Each Table
# ----------------------------------------------------------------------

def process_application_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    mask = chunk["patent_id"].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    filtered["patent_id"] = filtered["patent_id"].astype("string")
    filtered["filing_date"] = filtered["filing_date"].astype("string")
    return filtered[["patent_id", "filing_date"]]


def process_cpc_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    mask = chunk["patent_id"].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    for col in ["patent_id", "cpc_section", "cpc_class", "cpc_subclass", "cpc_group"]:
        filtered[col] = filtered[col].astype("string")
    return filtered[["patent_id", "cpc_section", "cpc_class", "cpc_subclass", "cpc_group"]]


def process_claim_stats_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    chunk["pat_no"] = chunk["pat_no"].astype(str)
    mask = chunk["pat_no"].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    filtered["patent_id"] = filtered["pat_no"].astype("string")
    filtered["pat_no"] = filtered["pat_no"].astype("string")
    if "appl_id" in filtered.columns:
        filtered["appl_id"] = filtered["appl_id"].astype("string")
    
    int_cols = ["pat_clm_ct", "pat_wrd_ct", "pat_wrd_min", "pat_dep_clm_ct", "pat_dep_wrd_ct"]
    for col in int_cols:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce").astype("Int64")
            
    float_cols = ["pat_dep_wrd_min", "pat_wrd_avg", "pat_dep_wrd_avg"]
    for col in float_cols:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce").astype("float64")
            
    cols = ["patent_id", "pat_no"] + [c for c in filtered.columns if c not in ["patent_id", "pat_no"]]
    return filtered[cols]


def process_foreign_citation_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    mask = chunk["patent_id"].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    for col in filtered.columns:
        filtered[col] = filtered[col].astype("string")
    return filtered


def process_us_citation_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    cite_col = "citation_patent_id" if "citation_patent_id" in chunk.columns else "citation_id"
    mask = chunk["patent_id"].isin(cohort_ids) | chunk[cite_col].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    
    if "citation_id" not in filtered.columns:
        filtered["citation_id"] = filtered[cite_col]
        
    for col in filtered.columns:
        filtered[col] = filtered[col].astype("string")
        
    target_cols = [c for c in [
        "patent_id",
        "citation_sequence",
        "citation_patent_id",
        "citation_id",
        "citation_date",
        "record_name",
        "wipo_kind",
        "citation_category"
    ] if c in filtered.columns]
    
    return filtered[target_cols]


def process_assignee_chunk(chunk: pd.DataFrame, cohort_ids: Set[str]) -> Optional[pd.DataFrame]:
    mask = chunk["patent_id"].isin(cohort_ids)
    if not mask.any():
        return None
    filtered = chunk[mask].copy()
    if "disambiguated_assignee_name" not in filtered.columns:
        org = filtered["disambig_assignee_organization"].fillna("")
        first = filtered.get("disambig_assignee_individual_name_first", pd.Series("", index=filtered.index)).fillna("")
        last = filtered.get("disambig_assignee_individual_name_last", pd.Series("", index=filtered.index)).fillna("")
        ind = (first + " " + last).str.strip()
        filtered["disambiguated_assignee_name"] = org.where(org != "", ind)
        
    filtered["patent_id"] = filtered["patent_id"].astype("string")
    filtered["assignee_id"] = filtered["assignee_id"].astype("string")
    filtered["disambiguated_assignee_name"] = filtered["disambiguated_assignee_name"].astype("string")
    filtered["assignee_type"] = filtered["assignee_type"].astype("string")
    
    return filtered[["patent_id", "assignee_id", "disambiguated_assignee_name", "assignee_type"]]


# ----------------------------------------------------------------------
# Generic Stream-and-Prune Execution Function
# ----------------------------------------------------------------------

def prune_table(
    table_name: str,
    input_file: Path,
    output_file: Path,
    sep: str,
    cohort_ids: Set[str],
    process_fn: Callable[[pd.DataFrame, Set[str]], Optional[pd.DataFrame]],
    chunksize: int = 250_000,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = setup_logger()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"PROCESSING: {table_name}")
    logger.info(f"  Source file:  {input_file}")
    logger.info(f"  Target file:  {output_file}")
    logger.info(f"  Delimiter:    {repr(sep)}")
    logger.info(f"  Chunk size:   {chunksize:,} rows")
    logger.info("=" * 60)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stream, zf_obj, file_bytes = resolve_file_stream(input_file)
    logger.info(f"Opened stream successfully ({file_bytes / (1024**2):.2f} MB)")

    total_read = 0
    total_matched = 0
    writer: Optional[pq.ParquetWriter] = None

    try:
        reader = pd.read_csv(
            stream,
            sep=sep,
            chunksize=chunksize,
            dtype=str,
            quoting=1 if sep == "\t" else 0,
            keep_default_na=False,
            na_values=["", "NULL", "null", "NaN"],
            low_memory=False,
        )

        pbar = tqdm(desc=f"Pruning {table_name}", unit="chunk")

        for chunk in reader:
            chunk_len = len(chunk)
            total_read += chunk_len

            filtered_df = process_fn(chunk, cohort_ids)
            if filtered_df is not None and len(filtered_df) > 0:
                matched_len = len(filtered_df)
                total_matched += matched_len

                arrow_table = pa.Table.from_pandas(filtered_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, schema=arrow_table.schema, compression="snappy")
                writer.write_table(arrow_table)

            pbar.update(1)
            pbar.set_postfix({"scanned": f"{total_read:,}", "matched": f"{total_matched:,}"})

        pbar.close()

    finally:
        if writer is not None:
            writer.close()
        if hasattr(stream, "close"):
            stream.close()
        if zf_obj is not None:
            zf_obj.close()

    elapsed = time.time() - start_time
    retention_pct = (total_matched / total_read * 100) if total_read > 0 else 0.0
    out_size_mb = (output_file.stat().st_size / (1024**2)) if output_file.exists() else 0.0

    logger.info("-" * 60)
    logger.info(f"Finished {table_name}:")
    logger.info(f"  Rows scanned:  {total_read:,}")
    logger.info(f"  Rows retained: {total_matched:,} ({retention_pct:.2f}%)")
    logger.info(f"  Output size:   {out_size_mb:.2f} MB")
    logger.info(f"  Runtime:       {elapsed:.2f} seconds")
    logger.info("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prune USPTO Relational Tables to Target Cohort")
    parser.add_argument("--asset-dir", type=Path, default=Path("asset"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--cohort-ids", type=Path, default=Path("data/interim/cohort_patent_ids.parquet"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["all", "application", "cpc", "claim_stats", "foreign_citations", "us_citations", "assignees"],
        help="Specific table to process (default: all)",
    )

    args = parser.parse_args()
    logger = setup_logger()

    cohort_ids = load_cohort_ids(args.cohort_ids, logger)

    table_configs = [
        {
            "id": "application",
            "name": "g_application",
            "source": args.asset_dir / "g_application.tsv.zip",
            "target": args.interim_dir / "filtered_application.parquet",
            "sep": "\t",
            "func": process_application_chunk,
        },
        {
            "id": "cpc",
            "name": "g_cpc_current",
            "source": args.asset_dir / "g_cpc_current.tsv.zip",
            "target": args.interim_dir / "filtered_cpc.parquet",
            "sep": "\t",
            "func": process_cpc_chunk,
        },
        {
            "id": "claim_stats",
            "name": "patent_document_stats",
            "source": args.asset_dir / "patent_document_stats.csv",
            "target": args.interim_dir / "filtered_claim_stats.parquet",
            "sep": ",",
            "func": process_claim_stats_chunk,
        },
        {
            "id": "foreign_citations",
            "name": "g_foreign_citation",
            "source": args.asset_dir / "g_foreign_citation.tsv.zip",
            "target": args.interim_dir / "filtered_foreign_citations.parquet",
            "sep": "\t",
            "func": process_foreign_citation_chunk,
        },
        {
            "id": "us_citations",
            "name": "g_us_patent_citation",
            "source": args.asset_dir / "g_us_patent_citation.tsv.zip",
            "target": args.interim_dir / "filtered_us_citations.parquet",
            "sep": "\t",
            "func": process_us_citation_chunk,
        },
        {
            "id": "assignees",
            "name": "g_assignee_disambiguated",
            "source": args.asset_dir / "g_assignee_disambiguated.tsv.zip",
            "target": args.interim_dir / "filtered_assignees.parquet",
            "sep": "\t",
            "func": process_assignee_chunk,
        },
    ]

    total_start = time.time()
    logger.info("Starting relational pruning pipeline for Fortune 500 cohort...")

    for cfg in table_configs:
        if args.table != "all" and args.table != cfg["id"]:
            continue
        prune_table(
            table_name=cfg["name"],
            input_file=cfg["source"],
            output_file=cfg["target"],
            sep=cfg["sep"],
            cohort_ids=cohort_ids,
            process_fn=cfg["func"],
            chunksize=args.chunksize,
            logger=logger,
        )

    logger.info(f"All requested tables processed successfully in {time.time() - total_start:.2f} seconds.")


if __name__ == "__main__":
    main()
