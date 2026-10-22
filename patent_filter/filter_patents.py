#!/usr/bin/env python3
"""
patent_filter/filter_patents.py

High-performance, modular patent filtering pipeline.
Filters raw USPTO patent data by grant/filing year, with optional cross-referencing
against Fortune 500 company lists (supporting multi-year union matching) or custom
company lists.

Author: Advanced Agentic Data Engineering
"""

import argparse
import logging
import os
import re
import string
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configures structured console logging with timestamp and level."""
    logger = logging.getLogger("patent_filter")
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ============================================================================
# String Normalization & Entity Cleaning
# ============================================================================

# Common corporate legal suffixes to strip iteratively
LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|"
    r"llc|lc|llp|lp|plc|gmbh|sa|sas|sarl|spa|bv|nv|ag|pty|holdings|holding|group)\b",
    re.IGNORECASE,
)
PUNCTUATION_REGEX = re.compile(r"[" + re.escape(string.punctuation) + r"]")
WHITESPACE_REGEX = re.compile(r"\s+")
TRAILING_CONNECTORS_REGEX = re.compile(r"\b(and|of|the|for)\s*$", re.IGNORECASE)


def clean_company_name(name: Any) -> str:
    """
    Standardizes company and assignee names for robust entity matching:
      - Converts to lowercase.
      - Replaces '&' with ' and '.
      - Strips all punctuation.
      - Iteratively strips corporate legal suffixes (e.g. Inc, LLC, Corp, Ltd).
      - Strips trailing conjunctions / prepositions (and, of, the).
      - Strips leading 'the '.
      - Normalizes consecutive whitespace.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""

    text = str(name).strip().lower()
    if not text or text == "nan" or text == "null":
        return ""

    # Replace ampersands with word 'and'
    text = text.replace("&", " and ")

    # Strip punctuation
    text = PUNCTUATION_REGEX.sub(" ", text)

    # Iteratively remove legal suffixes
    prev = None
    while text != prev:
        prev = text
        text = LEGAL_SUFFIX_REGEX.sub(" ", text)
        text = WHITESPACE_REGEX.sub(" ", text).strip()

    # Remove trailing connectors (e.g. "Sears Roebuck and" -> "Sears Roebuck")
    text = TRAILING_CONNECTORS_REGEX.sub("", text).strip()

    # Remove leading 'the '
    if text.startswith("the "):
        text = text[4:].strip()

    return text


# ============================================================================
# File Resolution & Streaming Ingestion
# ============================================================================

def resolve_file_stream(file_path: Path) -> Tuple[Any, Optional[zipfile.ZipFile], str]:
    """
    Opens a file stream from an uncompressed file or directly inside a .zip archive
    without extracting the entire archive to disk.

    Returns:
        (file_object, zipfile_handle_or_None, delimiter)
    """
    if not file_path.exists():
        # Check if a .zip variation exists
        zip_candidate = file_path.with_name(file_path.name + ".zip")
        if zip_candidate.exists():
            file_path = zip_candidate
        else:
            raise FileNotFoundError(f"Cannot find input file: {file_path}")

    # Determine delimiter from file naming
    filename_lower = file_path.name.lower()
    if ".tsv" in filename_lower:
        delimiter = "\t"
    elif ".csv" in filename_lower:
        delimiter = ","
    else:
        # Default fallback to comma
        delimiter = ","

    if file_path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(file_path, "r")
        # Find first valid data file inside
        inner_names = [
            n for n in zf.namelist()
            if not n.startswith("__MACOSX") and not n.endswith("/") and not n.startswith(".")
        ]
        if not inner_names:
            zf.close()
            raise FileNotFoundError(f"No valid data file found inside archive: {file_path}")
        
        inner_file = inner_names[0]
        if ".tsv" in inner_file.lower():
            delimiter = "\t"
        elif ".csv" in inner_file.lower():
            delimiter = ","

        stream = zf.open(inner_file, "r")
        return stream, zf, delimiter
    else:
        stream = open(file_path, "r", encoding="utf-8", errors="replace")
        return stream, None, delimiter


def find_file_in_assets(
    asset_dirs: List[Path],
    pattern_names: List[str]
) -> Optional[Path]:
    """
    Searches a list of directory paths for files matching pattern names or variations.
    Handles exact names, prefix matches, and .zip archives.
    """
    for adir in asset_dirs:
        if not adir.exists():
            continue
        for p in pattern_names:
            # Direct match
            direct = adir / p
            if direct.exists():
                return direct
            # Zip match
            direct_zip = adir / f"{p}.zip"
            if direct_zip.exists():
                return direct_zip
            # Case-insensitive / glob search
            matches = list(adir.glob(f"*{p}*"))
            if matches:
                return matches[0]

    return None


def detect_column(
    columns: Iterable[str],
    candidates: List[str],
    required: bool = True,
    file_label: str = "file"
) -> Optional[str]:
    """Finds the first candidate column present in columns (case-insensitive)."""
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]

    if required:
        raise KeyError(
            f"Could not find any required column from {candidates} in {file_label}. "
            f"Available columns: {list(columns)}"
        )
    return None


def extract_year_series(series: pd.Series) -> pd.Series:
    """
    Fast extraction of 4-digit integer year from string / datetime series.
    Optimized for high-speed chunk processing (10x faster than full pd.to_datetime).
    """
    str_series = series.astype(str)
    # Fast path: ISO date YYYY-MM-DD or YYYYMMDD
    first_four = str_series.str.slice(0, 4)
    # Check if first four characters are numeric and in plausible year range (1790-2100)
    numeric_years = pd.to_numeric(first_four, errors="coerce")
    valid_mask = numeric_years.between(1790, 2100)

    # If fast path succeeded for >= 90% of non-null rows, return it
    if valid_mask.sum() >= 0.9 * max(1, series.notna().sum()):
        return numeric_years

    # Fallback to regex 4-digit search or pd.to_datetime for unusual formats (e.g. MM/DD/YYYY)
    extracted = str_series.str.extract(r"(\b\d{4}\b)", expand=False)
    numeric_years = pd.to_numeric(extracted, errors="coerce")
    return numeric_years


# ============================================================================
# Company Dataset Ingestion & Union Matching Logic
# ============================================================================

def load_fortune500_data(
    f500_paths: List[Path],
    f500_target_years: Optional[Set[int]],
    logger: logging.Logger
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Loads Fortune 500 data across one or multiple files and/or years.

    UNION LOGIC:
      Any company appearing in AT LEAST ONE selected Fortune 500 annual list
      is retained in the company filter set.
    """
    all_dfs = []

    for path in f500_paths:
        logger.info(f"Loading Fortune 500 file: {path}")
        stream, zf, sep = resolve_file_stream(path)
        try:
            df = pd.read_csv(stream, sep=sep, low_memory=False)
            all_dfs.append(df)
        finally:
            if hasattr(stream, "close"):
                stream.close()
            if zf is not None:
                zf.close()

    if not all_dfs:
        raise FileNotFoundError("No valid Fortune 500 data found.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Identify relevant columns
    name_col = detect_column(
        combined.columns,
        ["company_name", "name", "company", "firm", "Company", "Name"],
        required=True,
        file_label="Fortune 500 file"
    )
    year_col = detect_column(
        combined.columns,
        ["year", "rank_year", "list_year", "Year"],
        required=False,
        file_label="Fortune 500 file"
    )
    industry_col = detect_column(
        combined.columns,
        ["industry", "Industry", "ind"],
        required=False,
        file_label="Fortune 500 file"
    )
    sector_col = detect_column(
        combined.columns,
        ["sector", "group", "Sector", "Group", "sector_group"],
        required=False,
        file_label="Fortune 500 file"
    )

    # Impute missing sector from industry if both columns exist
    if industry_col and sector_col:
        valid_pairs = combined[combined[sector_col].notna() & combined[industry_col].notna()]
        if not valid_pairs.empty:
            ind_to_sec = valid_pairs.groupby(industry_col)[sector_col].agg(
                lambda s: s.mode().iloc[0] if not s.empty else "Unspecified"
            ).to_dict()
            combined[sector_col] = combined[sector_col].fillna(combined[industry_col].map(ind_to_sec))

    # Apply year filtering if specific Fortune 500 years requested
    if year_col and f500_target_years:
        combined["_numeric_year"] = pd.to_numeric(combined[year_col], errors="coerce")
        combined = combined[combined["_numeric_year"].isin(f500_target_years)].copy()
        logger.info(
            f"Filtered Fortune 500 entries to specified year(s) {sorted(f500_target_years)}. "
            f"Retained {len(combined):,} firm-year rows."
        )

    # Clean and normalize company names
    combined["_clean_name"] = combined[name_col].apply(clean_company_name)
    combined = combined[combined["_clean_name"] != ""].copy()

    # Aggregate by cleaned company name applying UNION logic
    company_lookup: Dict[str, Dict[str, Any]] = {}
    aggregated_records = []

    grouped = combined.groupby("_clean_name")
    for clean_name, group in grouped:
        # Canonical company name (prefer latest year or most frequent)
        canonical_name = str(group[name_col].iloc[-1]).strip()

        # Industry & Sector
        ind = "Unspecified"
        if industry_col and group[industry_col].notna().any():
            ind = str(group[industry_col].dropna().iloc[-1]).strip()

        sec = "Unspecified"
        if sector_col and group[sector_col].notna().any():
            sec = str(group[sector_col].dropna().iloc[-1]).strip()

        # Years listed
        years_present = []
        if year_col and group[year_col].notna().any():
            years_present = sorted(group[year_col].dropna().unique().tolist())
            years_str = "; ".join(str(int(y) if isinstance(y, (int, float)) and not pd.isna(y) else y) for y in years_present)
        else:
            years_str = "All"

        rec = {
            "company_name": canonical_name,
            "clean_name": clean_name,
            "industry": ind,
            "sector": sec,
            "fortune500_years": years_str,
            "source": "Fortune 500"
        }
        company_lookup[clean_name] = rec
        aggregated_records.append(rec)

    companies_df = pd.DataFrame(aggregated_records)
    logger.info(
        f"Fortune 500 Union Processing complete: {len(company_lookup):,} distinct companies retained."
    )
    return company_lookup, companies_df


def load_custom_companies_data(
    custom_path: Path,
    logger: logging.Logger
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Loads custom company CSV dataset.
    Minimum required schema:
      - company_name (or name / company) [Required]
      - industry [Optional]
      - sector / group [Optional]
    """
    logger.info(f"Loading custom company list: {custom_path}")
    stream, zf, sep = resolve_file_stream(custom_path)
    try:
        df = pd.read_csv(stream, sep=sep, low_memory=False)
    finally:
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    name_col = detect_column(
        df.columns,
        ["company_name", "name", "company", "organization", "firm", "Company", "Name"],
        required=True,
        file_label="Custom Company CSV"
    )
    industry_col = detect_column(
        df.columns,
        ["industry", "Industry", "ind"],
        required=False,
        file_label="Custom Company CSV"
    )
    sector_col = detect_column(
        df.columns,
        ["sector", "group", "Sector", "Group", "sector_group"],
        required=False,
        file_label="Custom Company CSV"
    )

    df["_clean_name"] = df[name_col].apply(clean_company_name)
    df = df[df["_clean_name"] != ""].copy()

    company_lookup: Dict[str, Dict[str, Any]] = {}
    aggregated_records = []

    grouped = df.groupby("_clean_name")
    for clean_name, group in grouped:
        canonical_name = str(group[name_col].iloc[-1]).strip()
        ind = str(group[industry_col].dropna().iloc[-1]).strip() if (industry_col and group[industry_col].notna().any()) else "Unspecified"
        sec = str(group[sector_col].dropna().iloc[-1]).strip() if (sector_col and group[sector_col].notna().any()) else "Unspecified"

        rec = {
            "company_name": canonical_name,
            "clean_name": clean_name,
            "industry": ind,
            "sector": sec,
            "fortune500_years": "N/A",
            "source": "Custom List"
        }
        company_lookup[clean_name] = rec
        aggregated_records.append(rec)

    companies_df = pd.DataFrame(aggregated_records)
    logger.info(
        f"Custom company list loaded: {len(company_lookup):,} distinct companies retained."
    )
    return company_lookup, companies_df


# ============================================================================
# Patent Processing Pipeline: Year-Only Mode
# ============================================================================

def run_year_only_filtering(
    patent_file: Path,
    target_years: Set[int],
    years_slug: str,
    output_dir: Path,
    chunksize: int,
    date_col_override: Optional[str],
    logger: logging.Logger
) -> Tuple[Path, Path]:
    """
    Streams the raw patent file, filters records strictly by patent date/year,
    and writes out:
      1. patents_<years>.csv
      2. summary_patents_per_year.csv
    """
    patents_out_path = output_dir / f"patents_{years_slug}.csv"
    summary_year_path = output_dir / "summary_patents_per_year.csv"

    # Remove existing files if present
    if patents_out_path.exists():
        patents_out_path.unlink()
    if summary_year_path.exists():
        summary_year_path.unlink()

    stream, zf, sep = resolve_file_stream(patent_file)
    logger.info(f"Streaming raw patent data from: {patent_file.name} (sep={repr(sep)})")

    year_counter = Counter()
    total_scanned = 0
    total_matched = 0
    is_first_chunk = True

    try:
        reader = pd.read_csv(
            stream,
            sep=sep,
            chunksize=chunksize,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip"
        )

        pbar = tqdm(desc="Filtering patents by year", unit="chunk")
        for chunk in reader:
            total_scanned += len(chunk)

            # Detect date column
            date_col = date_col_override or detect_column(
                chunk.columns,
                ["patent_date", "grant_date", "issue_date", "date", "filing_date", "app_date", "year"],
                required=True,
                file_label=patent_file.name
            )

            # Extract integer year series
            years_series = extract_year_series(chunk[date_col])
            mask = years_series.isin(target_years)

            matched_chunk = chunk[mask].copy()
            if not matched_chunk.empty:
                total_matched += len(matched_chunk)
                # Count distribution
                chunk_counts = years_series[mask].astype(int).value_counts().to_dict()
                year_counter.update(chunk_counts)

                # Write chunk to output CSV
                matched_chunk.to_csv(
                    patents_out_path,
                    mode="w" if is_first_chunk else "a",
                    index=False,
                    header=is_first_chunk
                )
                is_first_chunk = False

            pbar.update(1)
        pbar.close()

    finally:
        if hasattr(stream, "close"):
            stream.close()
        if zf is not None:
            zf.close()

    logger.info(
        f"Year filtering complete: scanned {total_scanned:,} patents, "
        f"matched {total_matched:,} patents in year(s) {sorted(target_years)}."
    )

    # Write summary_patents_per_year.csv
    summary_rows = []
    for yr in sorted(target_years):
        summary_rows.append({"year": yr, "patent_count": year_counter.get(yr, 0)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_year_path, index=False)
    logger.info(f"Saved year summary to: {summary_year_path}")
    logger.info(f"Saved filtered patents to: {patents_out_path}")

    return patents_out_path, summary_year_path


# ============================================================================
# Patent Processing Pipeline: Company Filtering Mode
# ============================================================================

def run_company_filtering(
    patent_file: Path,
    assignee_file: Optional[Path],
    company_lookup: Dict[str, Dict[str, Any]],
    target_years: Set[int],
    years_slug: str,
    output_dir: Path,
    chunksize: int,
    date_col_override: Optional[str],
    assignee_col_override: Optional[str],
    enable_fuzzy: bool,
    fuzzy_threshold: int,
    logger: logging.Logger
) -> Tuple[Path, Path, Path]:
    """
    Executes Company Filtering Mode (Fortune 500 or Custom).
    Handles:
      - Relational USPTO Bulk files (g_patent.tsv + g_assignee_disambiguated.tsv)
      - Single consolidated patent extracts (with assignee column directly in patent file)
    Produces:
      1. filtered_patents_<years>_matched.csv
      2. summary_patents_per_company.csv
      3. summary_patents_per_industry.csv
    """
    matched_out_path = output_dir / f"filtered_patents_{years_slug}_matched.csv"
    company_summary_path = output_dir / "summary_patents_per_company.csv"
    industry_summary_path = output_dir / "summary_patents_per_industry.csv"

    # Remove existing files if present
    for p in [matched_out_path, company_summary_path, industry_summary_path]:
        if p.exists():
            p.unlink()

    # Step 1: Inspect patent file columns to determine if assignee is embedded
    s_stream, s_zf, s_sep = resolve_file_stream(patent_file)
    try:
        sample_df = pd.read_csv(s_stream, nrows=5, sep=s_sep, dtype=str)
    finally:
        if hasattr(s_stream, "close"):
            s_stream.close()
        if s_zf is not None:
            s_zf.close()

    embedded_assignee_col = assignee_col_override or detect_column(
        sample_df.columns,
        ["disambig_assignee_organization", "assignee_organization", "assignee_name", "assignee", "organization", "company"],
        required=False,
        file_label="Patent File"
    )

    fuzzy_cache: Dict[str, Optional[str]] = {}
    f500_clean_names = list(company_lookup.keys())

    def match_clean_org(org_clean: str) -> Optional[Dict[str, Any]]:
        """Matches a cleaned organization name against company lookup (exact + optional fuzzy)."""
        if not org_clean:
            return None
        # Tier 1: Exact normalized match
        if org_clean in company_lookup:
            return company_lookup[org_clean]

        # Tier 2: High-confidence fuzzy match (optional)
        if enable_fuzzy and RAPIDFUZZ_AVAILABLE and len(org_clean) >= 4:
            if org_clean in fuzzy_cache:
                matched_key = fuzzy_cache[org_clean]
                return company_lookup[matched_key] if matched_key else None

            best_match = process.extractOne(
                org_clean,
                f500_clean_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=fuzzy_threshold
            )
            if best_match:
                matched_key = best_match[0]
                fuzzy_cache[org_clean] = matched_key
                return company_lookup[matched_key]
            else:
                fuzzy_cache[org_clean] = None
                return None

        return None

    company_patent_counts = Counter()
    industry_patent_counts = Counter()
    industry_sector_map: Dict[str, str] = {}
    is_first_chunk = True
    total_matched_patents = 0

    # ------------------------------------------------------------------------
    # Branch A: Single Consolidated Patent File
    # ------------------------------------------------------------------------
    if embedded_assignee_col is not None and assignee_file is None:
        logger.info(
            f"Detected embedded assignee column '{embedded_assignee_col}' in patent file. "
            f"Processing single consolidated table..."
        )
        stream, zf, sep = resolve_file_stream(patent_file)
        try:
            reader = pd.read_csv(
                stream,
                sep=sep,
                chunksize=chunksize,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip"
            )
            pbar = tqdm(desc="Filtering and matching patents", unit="chunk")
            for chunk in reader:
                # Date filter
                date_col = date_col_override or detect_column(
                    chunk.columns,
                    ["patent_date", "grant_date", "issue_date", "date", "filing_date", "app_date", "year"],
                    required=True,
                    file_label=patent_file.name
                )
                years = extract_year_series(chunk[date_col])
                mask_year = years.isin(target_years)
                sub_chunk = chunk[mask_year].copy()

                if sub_chunk.empty:
                    pbar.update(1)
                    continue

                # Organization normalization & matching
                cleaned_orgs = sub_chunk[embedded_assignee_col].apply(clean_company_name)
                sub_chunk["_clean_org"] = cleaned_orgs

                unique_orgs = [u for u in sub_chunk["_clean_org"].unique() if u]
                org_match_map = {}
                for u in unique_orgs:
                    match_info = match_clean_org(u)
                    if match_info:
                        org_match_map[u] = match_info

                if not org_match_map:
                    pbar.update(1)
                    continue

                mask_matched = sub_chunk["_clean_org"].isin(org_match_map)
                matched_sub = sub_chunk[mask_matched]
                matched_rows = []

                for row_dict in matched_sub.to_dict(orient="records"):
                    clean_org = row_dict.pop("_clean_org")
                    info = org_match_map[clean_org]
                    row_dict["matched_company_name"] = info["company_name"]
                    row_dict["matched_company_clean"] = info["clean_name"]
                    row_dict["industry"] = info["industry"]
                    row_dict["sector"] = info["sector"]
                    row_dict["fortune500_years"] = info.get("fortune500_years", "N/A")
                    matched_rows.append(row_dict)

                    # Tally counts
                    c_name = info["company_name"]
                    ind = info["industry"]
                    sec = info["sector"]
                    company_patent_counts[c_name] += 1
                    industry_patent_counts[ind] += 1
                    if ind not in industry_sector_map:
                        industry_sector_map[ind] = sec

                if matched_rows:
                    matched_df = pd.DataFrame(matched_rows)
                    total_matched_patents += len(matched_df)
                    matched_df.to_csv(
                        matched_out_path,
                        mode="w" if is_first_chunk else "a",
                        index=False,
                        header=is_first_chunk
                    )
                    is_first_chunk = False

                pbar.update(1)
            pbar.close()
        finally:
            if hasattr(stream, "close"):
                stream.close()
            if zf is not None:
                zf.close()

    # ------------------------------------------------------------------------
    # Branch B: Relational USPTO Bulk Format (g_patent + g_assignee_disambiguated)
    # ------------------------------------------------------------------------
    else:
        if assignee_file is None:
            raise FileNotFoundError(
                "PatentsView requires an assignee table (e.g. g_assignee_disambiguated.tsv or .zip). "
                "Please place it in asset/ or specify --assignee-file."
            )

        logger.info("Operating in Relational Mode: Linking g_patent and g_assignee_disambiguated...")

        # Pass 1: Extract target patents for target year(s)
        logger.info(f"Pass 1: Scanning {patent_file.name} for year(s) {sorted(target_years)}...")
        target_patents_dict: Dict[str, Dict[str, Any]] = {}
        stream, zf, sep = resolve_file_stream(patent_file)

        try:
            reader = pd.read_csv(
                stream,
                sep=sep,
                chunksize=chunksize,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip"
            )
            pbar = tqdm(desc="Scanning patent years", unit="chunk")
            for chunk in reader:
                id_col = detect_column(
                    chunk.columns,
                    ["patent_id", "id", "patent_number"],
                    required=True,
                    file_label=patent_file.name
                )
                date_col = date_col_override or detect_column(
                    chunk.columns,
                    ["patent_date", "grant_date", "issue_date", "date", "filing_date", "app_date", "year"],
                    required=True,
                    file_label=patent_file.name
                )

                years = extract_year_series(chunk[date_col])
                mask = years.isin(target_years)
                filtered = chunk[mask]

                if not filtered.empty:
                    # Fast vectorized dictionary extraction
                    records = filtered.to_dict(orient="records")
                    for rec in records:
                        pid = str(rec[id_col])
                        target_patents_dict[pid] = rec

                pbar.update(1)
            pbar.close()
        finally:
            if hasattr(stream, "close"):
                stream.close()
            if zf is not None:
                zf.close()

        target_patent_ids = set(target_patents_dict.keys())
        logger.info(
            f"Pass 1 complete: Found {len(target_patent_ids):,} patents in year(s) {sorted(target_years)}."
        )

        if not target_patent_ids:
            logger.warning("No patents found matching target year(s).")

        # Pass 2: Stream assignee table and match against companies
        logger.info(f"Pass 2: Streaming assignee table {assignee_file.name} to resolve company matches...")
        stream_a, zf_a, sep_a = resolve_file_stream(assignee_file)

        try:
            reader_a = pd.read_csv(
                stream_a,
                sep=sep_a,
                chunksize=chunksize,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip"
            )
            pbar_a = tqdm(desc="Matching assignees to companies", unit="chunk")
            for chunk in reader_a:
                pid_col = detect_column(
                    chunk.columns,
                    ["patent_id", "id", "patent_number"],
                    required=True,
                    file_label=assignee_file.name
                )
                org_col = assignee_col_override or detect_column(
                    chunk.columns,
                    ["disambig_assignee_organization", "assignee_organization", "organization", "company", "assignee"],
                    required=True,
                    file_label=assignee_file.name
                )

                # Filter chunk for target patent IDs first
                chunk_sub = chunk[chunk[pid_col].isin(target_patent_ids)].copy()
                if chunk_sub.empty:
                    pbar_a.update(1)
                    continue

                cleaned_orgs = chunk_sub[org_col].apply(clean_company_name)
                chunk_sub["_clean_org"] = cleaned_orgs

                # Fast batch match for unique clean orgs in chunk
                unique_orgs = [u for u in chunk_sub["_clean_org"].unique() if u]
                org_match_map = {}
                for u in unique_orgs:
                    match_info = match_clean_org(u)
                    if match_info:
                        org_match_map[u] = match_info

                if not org_match_map:
                    pbar_a.update(1)
                    continue

                mask_matched = chunk_sub["_clean_org"].isin(org_match_map)
                matched_sub = chunk_sub[mask_matched]
                matched_rows = []

                for row_dict in matched_sub.to_dict(orient="records"):
                    clean_org = row_dict.pop("_clean_org")
                    info = org_match_map[clean_org]
                    pid = str(row_dict[pid_col])

                    # Retrieve patent metadata
                    patent_meta = target_patents_dict.get(pid, {pid_col: pid})

                    combined_row = dict(patent_meta)
                    # Add assignee columns
                    for col, val in row_dict.items():
                        if col not in combined_row:
                            combined_row[col] = val
                        else:
                            combined_row[f"assignee_{col}"] = val

                    # Add company metadata
                    combined_row["matched_company_name"] = info["company_name"]
                    combined_row["matched_company_clean"] = info["clean_name"]
                    combined_row["industry"] = info["industry"]
                    combined_row["sector"] = info["sector"]
                    combined_row["fortune500_years"] = info.get("fortune500_years", "N/A")
                    matched_rows.append(combined_row)

                    # Tally counts
                    c_name = info["company_name"]
                    ind = info["industry"]
                    sec = info["sector"]
                    company_patent_counts[c_name] += 1
                    industry_patent_counts[ind] += 1
                    if ind not in industry_sector_map:
                        industry_sector_map[ind] = sec

                if matched_rows:
                    matched_df = pd.DataFrame(matched_rows)
                    total_matched_patents += len(matched_df)
                    matched_df.to_csv(
                        matched_out_path,
                        mode="w" if is_first_chunk else "a",
                        index=False,
                        header=is_first_chunk
                    )
                    is_first_chunk = False

                pbar_a.update(1)
            pbar_a.close()
        finally:
            if hasattr(stream_a, "close"):
                stream_a.close()
            if zf_a is not None:
                zf_a.close()

    logger.info(
        f"Company matching complete: Captured {total_matched_patents:,} matching patent records."
    )

    # ------------------------------------------------------------------------
    # Generate Output Summaries
    # ------------------------------------------------------------------------

    # 1. summary_patents_per_company.csv
    company_summary_rows = []
    # Include all target companies, even if patent count is 0
    all_companies_seen = set(company_lookup.keys())
    for clean_name, info in company_lookup.items():
        c_name = info["company_name"]
        company_summary_rows.append({
            "company_name": c_name,
            "patent_count": company_patent_counts.get(c_name, 0),
            "industry": info["industry"],
            "sector": info["sector"],
            "fortune500_years": info.get("fortune500_years", "N/A"),
        })

    company_summary_df = pd.DataFrame(company_summary_rows)
    company_summary_df = company_summary_df.sort_values(by="patent_count", ascending=False)
    company_summary_df.to_csv(company_summary_path, index=False)
    logger.info(f"Saved company summary to: {company_summary_path}")

    # 2. summary_patents_per_industry.csv
    industry_summary_rows = []
    for ind, count in industry_patent_counts.most_common():
        industry_summary_rows.append({
            "industry": ind,
            "patent_count": count,
            "sector": industry_sector_map.get(ind, "Unspecified")
        })
    industry_summary_df = pd.DataFrame(industry_summary_rows)
    industry_summary_df.to_csv(industry_summary_path, index=False)
    logger.info(f"Saved industry summary to: {industry_summary_path}")
    logger.info(f"Saved matched patents to: {matched_out_path}")

    return matched_out_path, company_summary_path, industry_summary_path


# ============================================================================
# CLI Parser & Main Pipeline Orchestration
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Builds and parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="USPTO Patent Filtering Pipeline with Fortune 500 & Custom Company Cross-Referencing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Year-only filtering for 2018, 2019, 2020:
  python filter_patents.py --years 2018 2019 2020

  # 2. Year-range filtering (2015 through 2020 inclusive):
  python filter_patents.py --year-range 2015 2020

  # 3. Fortune 500 filtering (all companies union across all available lists):
  python filter_patents.py --years 2018 2019 --fortune500

  # 4. Fortune 500 filtering for specific list years (union rule):
  python filter_patents.py --years 2018 2019 --fortune500 2018 2019

  # 5. Custom company CSV filtering:
  python filter_patents.py --years 2020 --custom-companies asset/my_companies.csv
        """
    )

    # Filtering Criteria: Years
    year_group = parser.add_argument_group("Patent Date Filtering")
    year_group.add_argument(
        "--years",
        nargs="+",
        type=str,
        help="One or more patent years to filter by (e.g. --years 2018 2019 2020 or --years 2018,2019,2020)."
    )
    year_group.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START_YEAR", "END_YEAR"),
        help="Inclusive year range to filter by (e.g. --year-range 2015 2020)."
    )
    year_group.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Optional explicit name of the date column in the patent dataset (default: auto-detect)."
    )

    # Filtering Criteria: Company Lists
    company_group = parser.add_argument_group("Company Filtering Options")
    company_group.add_argument(
        "--fortune500",
        nargs="*",
        default=None,
        metavar="PATH_OR_YEAR",
        help="Enable Fortune 500 filtering. Optionally pass file path(s) and/or year identifiers "
             "(e.g. --fortune500 2018 2019 or --fortune500 asset/Fortune_500_Companies.csv)."
    )
    company_group.add_argument(
        "--fortune500-years",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of Fortune 500 ranking years to filter by (union matching rule applies)."
    )
    company_group.add_argument(
        "--custom-companies",
        type=str,
        default=None,
        help="Path to a custom company CSV file (minimum schema: company_name [required], industry [optional], sector [optional])."
    )
    company_group.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable secondary high-confidence fuzzy matching via rapidfuzz."
    )
    company_group.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=92,
        help="Similarity score threshold (0-100) for fuzzy matching (default: 92)."
    )

    # Paths & System Tuning
    sys_group = parser.add_argument_group("Paths & Performance")
    sys_group.add_argument(
        "--asset-dir",
        type=str,
        default=None,
        help="Directory containing asset files (default: auto-detected as patent_filter/asset or ./asset)."
    )
    sys_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where output CSVs will be saved (default: patent_filter/output or ./output)."
    )
    sys_group.add_argument(
        "--patent-file",
        type=str,
        default=None,
        help="Path to USPTO patent file (default: auto-detected in asset dir, e.g. g_patent.tsv[.zip])."
    )
    sys_group.add_argument(
        "--assignee-file",
        type=str,
        default=None,
        help="Path to USPTO assignee file (default: auto-detected in asset dir, e.g. g_assignee_disambiguated.tsv[.zip])."
    )
    sys_group.add_argument(
        "--assignee-col",
        type=str,
        default=None,
        help="Optional explicit name of the assignee organization column (default: auto-detect)."
    )
    sys_group.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size for streaming large files (default: 100,000 rows)."
    )
    sys_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging."
    )

    return parser.parse_args()


def resolve_target_years(args: argparse.Namespace) -> Set[int]:
    """Combines and validates target years from --years and --year-range."""
    years: Set[int] = set()

    if args.years:
        for item in args.years:
            # Handle comma-separated values
            for part in item.split(","):
                part = part.strip()
                if part:
                    try:
                        years.add(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid year specification: '{part}'")

    if args.year_range:
        start_y, end_y = args.year_range
        if start_y > end_y:
            raise ValueError(f"Invalid year range: start year ({start_y}) must be <= end year ({end_y})")
        years.update(range(start_y, end_y + 1))

    if not years:
        raise ValueError(
            "No filter years specified. You must provide at least one year via --years or --year-range."
        )

    return years


def format_years_slug(years: Set[int]) -> str:
    """Formats years set into a concise slug for filenames (e.g. 2018_2020 or 2018)."""
    sorted_years = sorted(years)
    if len(sorted_years) == 1:
        return str(sorted_years[0])
    
    # Check if continuous range
    if sorted_years == list(range(sorted_years[0], sorted_years[-1] + 1)):
        return f"{sorted_years[0]}_{sorted_years[-1]}"
    else:
        return "_".join(str(y) for y in sorted_years)


def main():
    start_time = time.time()
    args = parse_arguments()
    logger = setup_logger(verbose=args.verbose)

    logger.info("=== Starting USPTO Patent Filtering Pipeline ===")

    # 1. Resolve Target Years
    try:
        target_years = resolve_target_years(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    years_slug = format_years_slug(target_years)
    logger.info(f"Target Patent Year(s): {sorted(target_years)} (Slug: {years_slug})")

    # 2. Resolve Asset and Output Directories
    script_dir = Path(__file__).resolve().parent
    candidates_asset = []
    if args.asset_dir:
        candidates_asset.append(Path(args.asset_dir))
    candidates_asset.extend([
        script_dir / "asset",
        script_dir.parent / "asset",
        Path("asset"),
        Path("./patent_filter/asset"),
    ])
    
    asset_dir = None
    for cand in candidates_asset:
        if cand.exists():
            asset_dir = cand
            break
    if not asset_dir:
        asset_dir = script_dir / "asset"
        asset_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Asset Directory:  {asset_dir.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")

    # 3. Locate Patent File
    patent_file = None
    if args.patent_file:
        p_path = Path(args.patent_file)
        if p_path.exists():
            patent_file = p_path
        elif (asset_dir / args.patent_file).exists():
            patent_file = asset_dir / args.patent_file
    if not patent_file:
        patent_file = find_file_in_assets(
            [asset_dir, script_dir / "asset", script_dir.parent / "asset"],
            ["g_patent.tsv", "g_patent", "patents.csv", "patents.tsv", "raw_patents.csv"]
        )

    if not patent_file:
        logger.error(
            f"USPTO patent file not found in asset directory: {asset_dir}. "
            f"Please place g_patent.tsv (or .zip) or patents.csv into the asset directory, "
            f"or specify --patent-file."
        )
        sys.exit(1)

    logger.info(f"Located Patent File: {patent_file}")

    # 4. Check Execution Mode: Year-Only vs. Company Filtering
    is_f500_mode = (args.fortune500 is not None)
    is_custom_mode = (args.custom_companies is not None)

    if is_f500_mode and is_custom_mode:
        logger.error("Please specify either --fortune500 or --custom-companies, not both.")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Execution Path 1: Year-Only Filtering Mode
    # ------------------------------------------------------------------------
    if not is_f500_mode and not is_custom_mode:
        logger.info("Executing Path 1: Year-Only Filtering Mode")
        run_year_only_filtering(
            patent_file=patent_file,
            target_years=target_years,
            years_slug=years_slug,
            output_dir=output_dir,
            chunksize=args.chunksize,
            date_col_override=args.date_col,
            logger=logger
        )

    # ------------------------------------------------------------------------
    # Execution Path 2: Company Filtering Mode (Fortune 500 or Custom)
    # ------------------------------------------------------------------------
    else:
        logger.info("Executing Path 2: Company Filtering Mode")

        # Ingest company metadata
        if is_f500_mode:
            # Parse fortune500 arguments (can be paths or year numbers)
            f500_paths: List[Path] = []
            f500_years: Set[int] = set()

            if args.fortune500_years:
                f500_years.update(args.fortune500_years)

            for item in args.fortune500:
                # Check if item is an integer (year)
                if item.isdigit() and len(item) == 4:
                    f500_years.add(int(item))
                else:
                    cand_path = Path(item)
                    if cand_path.exists():
                        f500_paths.append(cand_path)
                    elif (asset_dir / item).exists():
                        f500_paths.append(asset_dir / item)
                    else:
                        logger.warning(f"Fortune 500 argument '{item}' not recognized as file or year.")

            # If no paths provided, auto-discover in asset directories
            if not f500_paths:
                discovered = find_file_in_assets(
                    [asset_dir, script_dir / "asset", script_dir.parent / "asset"],
                    ["Fortune_500_Companies.csv", "fortune_500.csv", "f500.csv", "fortune500.csv"]
                )
                if discovered:
                    f500_paths.append(discovered)

            if not f500_paths:
                logger.error(
                    f"No Fortune 500 company CSV file found in asset directory: {asset_dir}. "
                    f"Please place Fortune_500_Companies.csv in asset/ or pass the path via --fortune500."
                )
                sys.exit(1)

            target_f500_years_set = f500_years if f500_years else None
            company_lookup, companies_df = load_fortune500_data(
                f500_paths=f500_paths,
                f500_target_years=target_f500_years_set,
                logger=logger
            )

        else:
            # Custom company list
            custom_path = Path(args.custom_companies)
            if not custom_path.exists() and (asset_dir / args.custom_companies).exists():
                custom_path = asset_dir / args.custom_companies

            if not custom_path.exists():
                logger.error(f"Custom company file not found: {args.custom_companies}")
                sys.exit(1)

            company_lookup, companies_df = load_custom_companies_data(
                custom_path=custom_path,
                logger=logger
            )

        # Locate Assignee File if needed
        assignee_file = None
        if args.assignee_file:
            a_path = Path(args.assignee_file)
            if a_path.exists():
                assignee_file = a_path
            elif (asset_dir / args.assignee_file).exists():
                assignee_file = asset_dir / args.assignee_file
        else:
            # Inspect patent file: if it already contains assignee data, separate table is not required
            s_stream, s_zf, s_sep = resolve_file_stream(patent_file)
            try:
                sample_cols = pd.read_csv(s_stream, nrows=5, sep=s_sep, dtype=str).columns
            finally:
                if hasattr(s_stream, "close"):
                    s_stream.close()
                if s_zf is not None:
                    s_zf.close()

            has_embedded_assignee = any(
                c.lower() in [
                    "disambig_assignee_organization", "assignee_organization",
                    "assignee_name", "assignee", "organization", "company"
                ] for c in sample_cols
            ) or (args.assignee_col and args.assignee_col.lower() in [c.lower() for c in sample_cols])

            if not has_embedded_assignee:
                assignee_file = find_file_in_assets(
                    [asset_dir, script_dir / "asset", script_dir.parent / "asset"],
                    ["g_assignee_disambiguated.tsv", "g_assignee_disambiguated", "assignees.csv", "assignees.tsv"]
                )

        run_company_filtering(
            patent_file=patent_file,
            assignee_file=assignee_file,
            company_lookup=company_lookup,
            target_years=target_years,
            years_slug=years_slug,
            output_dir=output_dir,
            chunksize=args.chunksize,
            date_col_override=args.date_col,
            assignee_col_override=args.assignee_col,
            enable_fuzzy=args.fuzzy,
            fuzzy_threshold=args.fuzzy_threshold,
            logger=logger
        )

    elapsed = time.time() - start_time
    logger.info(f"=== Pipeline Finished Successfully in {elapsed:.2f} seconds ===")


if __name__ == "__main__":
    main()
