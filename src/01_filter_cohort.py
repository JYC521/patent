#!/usr/bin/env python3
"""
src/01_filter_cohort.py

Extracts the target Fortune 500 utility patent cohort (2010-2013).
Restricted exclusively to utility patents belonging to Fortune 500 companies listed in 2012.

Pipeline Stages:
  Stage 1: Load Fortune 500 (2012) and PatentsView corporate assignees.
  Stage 2: Standardize company names and execute Tier 1 (Exact) & Tier 2 (Fuzzy) entity matching.
  Stage 3: Save crosswalk and matched assignee IDs.
  Stage 4: Stream g_assignee_disambiguated to map patent_id to matched Fortune 500 assignees.
  Stage 5: Stream g_patent to filter 2010-2013 utility patents belonging to the Fortune 500 cohort.
  Stage 6: Overwrite cohort Parquet files:
           - data/interim/cohort_patents.parquet
           - data/interim/cohort_patent_ids.parquet
           - data/interim/cohort_patent_assignees.parquet
"""

import argparse
import logging
import os
import re
import string
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from tqdm import tqdm


def setup_logger() -> logging.Logger:
    """Configures structured console logging."""
    logger = logging.getLogger("01_filter_cohort")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def clean_name(name: str) -> str:
    """
    Standardizes company names:
      - Converts to lowercase.
      - Removes punctuation and replaces with space.
      - Strips whitespace.
      - Removes corporate entity suffixes using regex:
        \b(inc|corp|corporation|co|company|ltd|limited|llc|plc|group|holdings|sa|nv|ag|lp)\b
    """
    if not isinstance(name, str):
        return ""
    text = name.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    suffix_pattern = r"\b(inc|corp|corporation|co|company|ltd|limited|llc|plc|group|holdings|sa|nv|ag|lp)\b"
    prev = None
    while text != prev:
        prev = text
        text = re.sub(suffix_pattern, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text.strip()


def resolve_file_stream(file_path: Path) -> Tuple[object, Optional[zipfile.ZipFile], int]:
    """
    Opens a file stream from an uncompressed file or directly inside a .zip archive
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

    raise FileNotFoundError(f"Could not locate '{file_path}' or any zipped/uncompressed alternative.")


def load_fortune_500_2012(f500_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Loads Fortune 500 dataset, filters for 2012, and imputes sector from official industry taxonomy."""
    stream, zf, size_bytes = resolve_file_stream(f500_path)
    try:
        df = pd.read_csv(stream)
    finally:
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    # Build industry-to-sector lookup from years where both are recorded
    with_sec = df[df["sector"].notnull() & df["industry"].notnull()]
    ind_to_sec = with_sec.groupby("industry")["sector"].agg(lambda s: s.mode().iloc[0]).to_dict()

    year_col = "year" if "year" in df.columns else "Year"
    df_2012 = df[df[year_col] == 2012].copy()
    if "sector" not in df_2012.columns or df_2012["sector"].isnull().all():
        df_2012["sector"] = df_2012["industry"].map(ind_to_sec)
    else:
        df_2012["sector"] = df_2012["sector"].fillna(df_2012["industry"].map(ind_to_sec))

    name_col = "name" if "name" in df_2012.columns else "Company"
    df_2012["f500_company_name"] = df_2012[name_col].astype(str)
    df_2012["cleaned_name"] = df_2012["f500_company_name"].apply(clean_name)
    logger.info(f"Loaded {len(df_2012):,} Fortune 500 companies listed in 2012.")
    return df_2012


def extract_unique_corporate_assignees(
    assignee_path: Path,
    chunksize: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Streams g_assignee_disambiguated to extract unique corporate assignees (types 2 and 3)."""
    logger.info("Extracting unique corporate assignees from g_assignee_disambiguated...")
    stream, zf, _ = resolve_file_stream(assignee_path)
    corp_chunks = []
    total_scanned = 0

    try:
        reader = pd.read_csv(
            stream,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            quoting=1,
            keep_default_na=False,
            na_values=["", "NULL", "null", "NaN"],
        )
        pbar = tqdm(desc="Scanning corporate assignees", unit="chunk")
        for chunk in reader:
            total_scanned += len(chunk)
            mask = chunk["assignee_type"].isin(["2", "3"])
            sub = chunk[mask][["assignee_id", "disambig_assignee_organization"]].dropna()
            sub = sub[sub["disambig_assignee_organization"].str.strip() != ""]
            if len(sub) > 0:
                corp_chunks.append(sub)
            pbar.update(1)
        pbar.close()
    finally:
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    all_corp = pd.concat(corp_chunks, ignore_index=True).drop_duplicates(subset=["assignee_id"])
    all_corp["disambiguated_assignee_name"] = all_corp["disambig_assignee_organization"].str.strip()
    all_corp["cleaned_name"] = all_corp["disambiguated_assignee_name"].apply(clean_name)
    all_corp = all_corp[all_corp["cleaned_name"] != ""].copy()
    logger.info(f"Scanned {total_scanned:,} assignee rows; found {len(all_corp):,} unique corporate assignees.")
    return all_corp


def perform_entity_matching(
    df_2012: pd.DataFrame,
    assignee_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Executes Tier 1 (Exact Match) and Tier 2 (High-Confidence Fuzzy Match via rapidfuzz).
    Returns audit crosswalk DataFrame.
    """
    logger.info("Executing Tier 1 (Exact Name Match)...")
    t1_merged = pd.merge(
        df_2012[["f500_company_name", "industry", "sector", "cleaned_name"]],
        assignee_df[["assignee_id", "disambiguated_assignee_name", "cleaned_name"]],
        on="cleaned_name",
        how="inner",
    )
    t1_records = []
    for _, r in t1_merged.iterrows():
        t1_records.append({
            "f500_company_name": r["f500_company_name"],
            "industry": r["industry"],
            "sector": r["sector"],
            "assignee_id": r["assignee_id"],
            "disambiguated_assignee_name": r["disambiguated_assignee_name"],
            "match_tier": "Tier 1 (Exact)",
            "match_score": 100.0,
        })

    matched_f500_names = set(t1_merged["f500_company_name"])
    logger.info(f"Tier 1 matched {len(matched_f500_names)} Fortune 500 companies ({len(t1_records)} crosswalk rows).")

    unmatched = df_2012[~df_2012["f500_company_name"].isin(matched_f500_names)].copy()
    logger.info(f"Executing Tier 2 (High-Confidence Fuzzy Match) for {len(unmatched)} unmatched firms...")

    t2_records = []
    pbar = tqdm(total=len(unmatched), desc="Tier 2 Fuzzy Matching", unit="firm")

    for _, row in unmatched.iterrows():
        f_name = row["f500_company_name"]
        f_clean = row["cleaned_name"]
        f_tokens = set(f_clean.split())
        if not f_tokens:
            pbar.update(1)
            continue

        first_token = f_clean.split()[0]
        token_query = first_token if len(first_token) >= 3 else f_clean

        # Filter candidate pool containing the primary corporate identifier token
        cand_subset = assignee_df[
            assignee_df["cleaned_name"].str.contains(r"\b" + re.escape(token_query) + r"\b", regex=True, na=False)
        ]

        for _, c_row in cand_subset.iterrows():
            c_clean = c_row["cleaned_name"]
            score = fuzz.token_set_ratio(f_clean, c_clean)
            if score >= 90.0:
                c_tokens = set(c_clean.split())
                # High-confidence guardrail: either F tokens are subset of candidate tokens (division/subsidiary),
                # or token_sort_ratio >= 85 (near-identical typographical match)
                if f_tokens.issubset(c_tokens) or fuzz.token_sort_ratio(f_clean, c_clean) >= 85.0:
                    t2_records.append({
                        "f500_company_name": f_name,
                        "industry": row["industry"],
                        "sector": row["sector"],
                        "assignee_id": c_row["assignee_id"],
                        "disambiguated_assignee_name": c_row["disambiguated_assignee_name"],
                        "match_tier": "Tier 2 (Fuzzy)",
                        "match_score": round(score, 1),
                    })
        pbar.update(1)

    pbar.close()

    t2_df = pd.DataFrame(t2_records)
    t2_f500_count = t2_df["f500_company_name"].nunique() if len(t2_df) > 0 else 0
    logger.info(f"Tier 2 matched {t2_f500_count} additional firms ({len(t2_records)} crosswalk rows).")

    all_records = t1_records + t2_records
    crosswalk = pd.DataFrame(all_records).drop_duplicates(subset=["assignee_id"]).reset_index(drop=True)
    total_matched_firms = crosswalk["f500_company_name"].nunique()
    logger.info(f"Total Fortune 500 firms matched: {total_matched_firms} / {len(df_2012)} ({total_matched_firms/len(df_2012)*100:.1f}%)")
    logger.info(f"Total unique assignee IDs linked: {len(crosswalk):,}")
    return crosswalk


def stream_f500_patents(
    assignee_path: Path,
    crosswalk: pd.DataFrame,
    chunksize: int,
    logger: logging.Logger,
) -> Dict[str, List[dict]]:
    """
    Streams g_assignee_disambiguated to identify all patent_ids associated with matched assignee_ids.
    Returns mapping: patent_id -> list of assignee metadata dicts.
    """
    matched_aid_set = set(crosswalk["assignee_id"])
    aid_to_meta = crosswalk.set_index("assignee_id")[
        ["f500_company_name", "industry", "sector"]
    ].to_dict(orient="index")

    logger.info("Streaming g_assignee_disambiguated to map patent_ids to matched Fortune 500 assignees...")
    stream, zf, _ = resolve_file_stream(assignee_path)
    patent_to_assignees: Dict[str, List[dict]] = {}
    total_scanned = 0

    try:
        reader = pd.read_csv(
            stream,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            quoting=1,
            keep_default_na=False,
            na_values=["", "NULL", "null", "NaN"],
        )
        pbar = tqdm(desc="Mapping F500 patent_ids", unit="chunk")
        for chunk in reader:
            total_scanned += len(chunk)
            sub = chunk[chunk["assignee_id"].isin(matched_aid_set)]
            if len(sub) > 0:
                for _, r in sub.iterrows():
                    pid = r["patent_id"]
                    aid = r["assignee_id"]
                    meta = aid_to_meta[aid]
                    entry = {
                        "patent_id": pid,
                        "assignee_id": aid,
                        "f500_company_name": meta["f500_company_name"],
                        "industry": meta["industry"],
                        "sector": meta["sector"],
                    }
                    if pid not in patent_to_assignees:
                        patent_to_assignees[pid] = [entry]
                    else:
                        patent_to_assignees[pid].append(entry)
            pbar.update(1)
        pbar.close()
    finally:
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    logger.info(f"Mapped {len(patent_to_assignees):,} distinct patent IDs to Fortune 500 assignees across all grant history.")
    return patent_to_assignees


def filter_and_save_cohort(
    patent_path: Path,
    patent_to_assignees: Dict[str, List[dict]],
    output_patents: Path,
    output_ids: Path,
    output_assignees: Path,
    start_date: str,
    end_date: str,
    chunksize: int,
    logger: logging.Logger,
) -> None:
    """
    Streams g_patent and writes the filtered Fortune 500 cohort tables to data/interim/.
    """
    logger.info("=" * 60)
    logger.info("Filtering 2010-2013 Utility Patents belonging to Fortune 500 Cohort")
    logger.info("=" * 60)

    f500_patent_set = set(patent_to_assignees.keys())
    stream, zf, file_size = resolve_file_stream(patent_path)

    output_patents.parent.mkdir(parents=True, exist_ok=True)
    patents_writer: Optional[pq.ParquetWriter] = None

    patents_schema = pa.schema([
        ("patent_id", pa.string()),
        ("patent_date", pa.string()),
        ("patent_title", pa.string()),
        ("patent_abstract", pa.string()),
    ])

    total_scanned = 0
    matched_cohort_pids = set()
    cohort_assignee_records = []

    try:
        reader = pd.read_csv(
            stream,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            quoting=1,
            keep_default_na=False,
            na_values=["", "NULL", "null", "NaN"],
        )
        pbar = tqdm(desc="Filtering g_patent chunks", unit="chunk")

        for chunk in reader:
            total_scanned += len(chunk)
            mask = (
                (chunk["patent_type"].str.lower() == "utility")
                & (chunk["patent_date"] >= start_date)
                & (chunk["patent_date"] <= end_date)
                & (chunk["patent_id"].isin(f500_patent_set))
            )
            filtered = chunk[mask].copy()

            if len(filtered) > 0:
                filtered["patent_id"] = filtered["patent_id"].astype("string")
                filtered["patent_date"] = filtered["patent_date"].astype("string")
                filtered["patent_title"] = filtered["patent_title"].astype("string")

                if "patent_abstract" in filtered.columns:
                    filtered["patent_abstract"] = filtered["patent_abstract"].astype("string")
                else:
                    filtered["patent_abstract"] = pd.Series([None] * len(filtered), dtype="string")

                batch_patents = filtered[["patent_id", "patent_date", "patent_title", "patent_abstract"]]
                arrow_table = pa.Table.from_pandas(batch_patents, schema=patents_schema, preserve_index=False)

                if patents_writer is None:
                    patents_writer = pq.ParquetWriter(output_patents, schema=patents_schema, compression="snappy")
                patents_writer.write_table(arrow_table)

                batch_pids = batch_patents["patent_id"].tolist()
                matched_cohort_pids.update(batch_pids)

                for pid in batch_pids:
                    for entry in patent_to_assignees[pid]:
                        cohort_assignee_records.append(entry)

            pbar.update(1)
            pbar.set_postfix({"scanned": f"{total_scanned:,}", "f500_patents": f"{len(matched_cohort_pids):,}"})
        pbar.close()

    finally:
        if patents_writer is not None:
            patents_writer.close()
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    # Save cohort_patent_ids.parquet
    logger.info("Writing isolated cohort_patent_ids.parquet...")
    sorted_pids = sorted(matched_cohort_pids)
    ids_table = pa.Table.from_pydict(
        {"patent_id": sorted_pids},
        schema=pa.schema([("patent_id", pa.string())]),
    )
    pq.write_table(ids_table, output_ids, compression="snappy")

    # Save cohort_patent_assignees.parquet
    logger.info("Writing cohort_patent_assignees.parquet...")
    assignees_df = pd.DataFrame(cohort_assignee_records).drop_duplicates()
    for col in ["patent_id", "assignee_id", "f500_company_name", "industry", "sector"]:
        assignees_df[col] = assignees_df[col].astype("string")
    assignees_table = pa.Table.from_pandas(assignees_df, preserve_index=False)
    pq.write_table(assignees_table, output_assignees, compression="snappy")

    logger.info("-" * 60)
    logger.info("Cohort Generation Complete:")
    logger.info(f"  Total patents scanned:      {total_scanned:,}")
    logger.info(f"  F500 utility cohort matched:{len(matched_cohort_pids):,}")
    logger.info(f"  Cohort assignee links:      {len(assignees_df):,}")
    logger.info(f"  Output cohort patents:      {output_patents} ({output_patents.stat().st_size / (1024**2):.2f} MB)")
    logger.info(f"  Output cohort IDs:          {output_ids} ({output_ids.stat().st_size / (1024**2):.2f} MB)")
    logger.info(f"  Output cohort assignees:    {output_assignees} ({output_assignees.stat().st_size / (1024**2):.2f} MB)")
    logger.info("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract Fortune 500 Utility Patent Cohort (2010-2013)")
    parser.add_argument("--f500-csv", type=Path, default=Path("asset/Fortune_500_Companies.csv"))
    parser.add_argument("--assignee-tsv", type=Path, default=Path("asset/g_assignee_disambiguated.tsv.zip"))
    parser.add_argument("--patent-tsv", type=Path, default=Path("asset/g_patent.tsv.zip"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default="2013-12-31")
    parser.add_argument("--chunksize", type=int, default=250_000)

    args = parser.parse_args()
    logger = setup_logger()
    args.interim_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Step 1 & 2: Load F500 and Assignees, execute matching
    df_2012 = load_fortune_500_2012(args.f500_csv, logger)
    assignee_df = extract_unique_corporate_assignees(args.assignee_tsv, args.chunksize, logger)
    crosswalk = perform_entity_matching(df_2012, assignee_df, logger)

    # Step 3: Save Crosswalk & Matched Assignee IDs
    crosswalk_csv = args.interim_dir / "f500_2012_assignee_crosswalk.csv"
    crosswalk.to_csv(crosswalk_csv, index=False)
    logger.info(f"Saved audit crosswalk to {crosswalk_csv}")

    matched_aids = sorted(crosswalk["assignee_id"].unique())
    matched_aids_parquet = args.interim_dir / "f500_matched_assignee_ids.parquet"
    aids_table = pa.Table.from_pydict(
        {"assignee_id": matched_aids},
        schema=pa.schema([("assignee_id", pa.string())]),
    )
    pq.write_table(aids_table, matched_aids_parquet, compression="snappy")
    logger.info(f"Saved matched assignee IDs to {matched_aids_parquet}")

    # Step 4: Stream g_assignee_disambiguated to map patent_id to F500 assignees
    patent_to_assignees = stream_f500_patents(args.assignee_tsv, crosswalk, args.chunksize, logger)

    # Step 5 & 6: Stream g_patent to filter cohort and overwrite Parquet tables
    output_patents = args.interim_dir / "cohort_patents.parquet"
    output_ids = args.interim_dir / "cohort_patent_ids.parquet"
    output_assignees = args.interim_dir / "cohort_patent_assignees.parquet"

    filter_and_save_cohort(
        patent_path=args.patent_tsv,
        patent_to_assignees=patent_to_assignees,
        output_patents=output_patents,
        output_ids=output_ids,
        output_assignees=output_assignees,
        start_date=args.start_date,
        end_date=args.end_date,
        chunksize=args.chunksize,
        logger=logger,
    )

    logger.info(f"All cohort extraction tasks finished in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    main()
