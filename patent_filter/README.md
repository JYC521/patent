# Patent Filter: Modular USPTO Data Pipeline

A high-performance, memory-efficient Python pipeline designed to ingest, filter, and cross-reference bulk United States Patent and Trademark Office (USPTO) data.

The pipeline filters raw USPTO patents by grant or filing year, and optionally cross-references assignees against **Fortune 500** corporate rankings (supporting multi-year union matching) or user-provided **Custom Company** datasets.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Directory Structure](#directory-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [USPTO Bulk Data Download Guide](#uspto-bulk-data-download-guide)
5. [Company Dataset Schema Specifications](#company-dataset-schema-specifications)
   - [Fortune 500 CSV Schema](#fortune-500-csv-schema)
   - [Custom Company CSV Schema](#custom-company-csv-schema)
6. [Multi-Year Fortune 500 Union Matching Rule](#multi-year-fortune-500-union-matching-rule)
7. [CLI Usage & Configuration Guide](#cli-usage--configuration-guide)
   - [CLI Parameters Reference](#cli-parameters-reference)
   - [Execution Examples](#execution-examples)
8. [Output Deliverables](#output-deliverables)
   - [Year-Only Mode Deliverables](#year-only-mode-deliverables)
   - [Company Filtering Mode Deliverables](#company-filtering-mode-deliverables)
9. [Entity Resolution & Normalization Logic](#entity-resolution--normalization-logic)

---

## Key Features

- **Chunked Streaming & Low Memory Footprint**: Reads gigabyte-scale USPTO files chunk-by-chunk with constant memory usage $O(\text{chunksize})$.
- **Native ZIP Streaming**: Reads compressed archives (e.g. `g_patent.tsv.zip`, `g_assignee_disambiguated.tsv.zip`) directly on-the-fly without unzipping them to disk, saving tens of gigabytes of disk space.
- **Dual Processing Paths**:
  1. *Year-Only Mode*: Filters raw patents by target years and produces annual volume summaries.
  2. *Company Filtering Mode*: Cross-references patent assignees against Fortune 500 or custom company sets, attaching rich industry and sector classifications.
- **Multi-Year Fortune 500 Union Matching**: Implements a union rule across multiple Fortune 500 list years—any firm ranked in *at least one* selected list is tracked across all target patent years.
- **Robust Entity Normalization**: Strips legal suffixes (`Inc`, `LLC`, `Corp`, `GmbH`, `PLC`, etc.), cleans punctuation, and normalizes casing to maximize match recall against USPTO assignees.
- **Optional High-Confidence Fuzzy Matching**: Integrates secondary token-set fuzzy matching via RapidFuzz for slight typographical variations.

---

## Directory Structure

The project strictly follows the modular layout below:

```text
patent_filter/
├── asset/                   # Place raw USPTO bulk files (.tsv/.zip) and company CSVs here
├── output/                  # Generated filtered CSVs and summary reports
├── filter_patents.py        # Core processing script & pipeline engine
├── requirements.txt         # Required Python dependencies
└── README.md                # Comprehensive documentation & user guide
```

Sample fixtures are provided in `asset/` so you can run and test the pipeline.

---

## Prerequisites & Installation

### 1. Requirements
- Python 3.8 or higher
- `pip` package manager

### 2. Setup Virtual Environment (Recommended)

```bash
cd patent_filter

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Dependencies in `requirements.txt`:
- `pandas>=2.0.0`: Chunked data manipulation and streaming I/O.
- `pyarrow>=14.0.0`: High-speed tabular processing engine.
- `rapidfuzz>=3.0.0`: High-performance C++ accelerated string normalization and fuzzy matching.
- `tqdm>=4.65.0`: Real-time progress bars for chunk streaming.

---

## USPTO Bulk Data Download Guide

The pipeline is pre-configured to ingest official bulk tabular data provided by **PatentsView** (the official USPTO data platform).

### Step-by-Step Download Instructions:

1. Visit the **PatentsView Bulk Data Portal**:  
   [https://patentsview.org/download/data-download-tables](https://patentsview.org/download/data-download-tables)

2. Download the following required tables:

| File Name | Description | Why It's Needed |
| :--- | :--- | :--- |
| **`g_patent.tsv.zip`** | Master patent table containing `patent_id`, `patent_date`, `patent_title`, `num_claims`, etc. | **Required** for all modes to filter patents by grant date. |
| **`g_assignee_disambiguated.tsv.zip`** | Disambiguated assignee records containing `patent_id`, `assignee_id`, `disambig_assignee_organization`, `assignee_type`. | **Required** for Company Filtering Mode when using PatentsView tables. |
| **`g_application.tsv.zip`** *(Optional)* | Patent application details containing `patent_id` and `filing_date`. | Optional, only if you wish to filter by filing year instead of grant year. |

3. Place the downloaded files directly into the `patent_filter/asset/` directory:

```bash
patent_filter/asset/
├── g_patent.tsv.zip
├── g_assignee_disambiguated.tsv.zip
└── Fortune_500_Companies.csv
```

> [!NOTE]
> **Do not unzip the downloaded `.zip` files!**  
> `filter_patents.py` automatically detects and streams data directly from inside the `.tsv.zip` archives, conserving your storage.

> [!TIP]
> **Custom / Unified Patent CSVs Supported**: If you already have an exported patent file (e.g., `patents.csv` or `raw_patents.tsv`) that already contains both the patent date and assignee organization in a single file, simply drop it into `patent_filter/asset/`. The pipeline automatically detects embedded assignee columns and processes the file in a single high-speed pass.

---

## Company Dataset Schema Specifications

### Fortune 500 CSV Schema

The pipeline supports official Fortune 500 datasets (such as `Fortune_500_Companies.csv` covering 1996–2023 or annual extract CSVs).

#### Expected Columns:
| Column Name | Required | Data Type | Description & Example |
| :--- | :---: | :--- | :--- |
| `company_name` (or `name`) | **Yes** | String | Official corporate name. E.g. `"Apple"`, `"General Motors Corporation"`, `"Microsoft Corp."` |
| `year` | Recommended | Integer | Ranking list year (e.g. `2018`, `2019`, `2020`). |
| `industry` | Recommended | String | Industry category. E.g. `"Computers, Office Equipment"`, `"Pharmaceuticals"`. |
| `sector` | Optional | String | Sector grouping. E.g. `"Technology"`, `"Health Care"`. If missing, automatically imputed from the industry taxonomy. |

#### Sample Fortune 500 Rows:
```csv
company_name,year,industry,sector
"Apple",2018,"Computers, Office Equipment","Technology"
"Exxon Mobil",2018,"Petroleum Refining","Energy"
"Amazon.com",2019,"Internet Services and Retailing","Technology"
"Tesla",2019,"Motor Vehicles & Parts","Motor Vehicles & Parts"
```

---

### Custom Company CSV Schema

Users can supply any arbitrary corporate target list via the `--custom-companies` argument.

#### Minimum Required Schema:
| Column Name | Required | Data Type | Description |
| :--- | :---: | :--- | :--- |
| `company_name` (or `name`, `company`) | **Yes** | String | Name of the firm to track. |
| `industry` | Optional | String | Industry label (defaults to `"Unspecified"` if omitted). |
| `sector` (or `group`) | Optional | String | Broader group or sector (defaults to `"Unspecified"` if omitted). |

#### Sample Custom Company CSV (`asset/my_companies.csv`):
```csv
company_name,industry,sector
"Apple Inc.","Consumer Electronics","Technology"
"Tesla, Inc.","Automotive","Clean Tech"
"Pfizer Inc.","Biopharmaceuticals","Healthcare"
"Samsung Electronics","Electronics","Technology"
"Moderna Therapeutics","Biotechnology","Healthcare"
```

---

## Multi-Year Fortune 500 Union Matching Rule

When multiple Fortune 500 years or files are selected, the pipeline applies a **Union Rule**:

$$\text{Target Companies} = \bigcup_{y \in Y} \text{Companies}(y)$$

### How the Rule Works:
1. **Inclusion Criteria**: A firm is included in the company filter set if it appeared in **at least one** of the specified Fortune 500 annual lists.
2. **Comprehensive Patent Capture**: Once a company is qualified via the union rule, **all** of its matching patents within the target patent years are captured.
3. **Firm Metadata Preservation**: The output records the specific Fortune 500 ranking years in which the firm was present (e.g. `"2018; 2019; 2020"`).

### Real-World Example:
- Suppose you filter patent years **2018 through 2020** (`--years 2018 2019 2020`) against Fortune 500 lists for **2019 and 2020** (`--fortune500 2019 2020`).
- *Tesla* entered the Fortune 500 in 2019 (not ranked in 2018).
- Under the union rule, Tesla is retained in the filter set because it appeared in at least one list (2019, 2020).
- Consequently, **all of Tesla's patents granted in 2018, 2019, and 2020 are captured**, providing a complete view of corporate intellectual property leading up to and during their Fortune 500 listing.

---

## CLI Usage & Configuration Guide

### CLI Parameters Reference

Run `python filter_patents.py --help` to view all available CLI options:

```text
Patent Date Filtering:
  --years YEARS [YEARS ...]
                        One or more patent years to filter by (e.g. --years 2018 2019 2020 or --years 2018,2019,2020).
  --year-range START_YEAR END_YEAR
                        Inclusive year range to filter by (e.g. --year-range 2015 2020).
  --date-col DATE_COL   Explicit name of the date column in the patent table (default: auto-detected).

Company Filtering Options:
  --fortune500 [PATH_OR_YEAR ...]
                        Enable Fortune 500 filtering. Accepts file path(s) and/or list years
                        (e.g. --fortune500 2018 2019 or --fortune500 asset/Fortune_500_Companies.csv).
  --fortune500-years YEARS [YEARS ...]
                        Explicit list of Fortune 500 ranking years to filter by (union matching rule applies).
  --custom-companies PATH
                        Path to a custom company CSV file.
  --fuzzy               Enable secondary high-confidence fuzzy matching via RapidFuzz.
  --fuzzy-threshold INT Similarity score threshold (0-100) for fuzzy matching (default: 92).

Paths & Performance:
  --asset-dir DIR       Directory containing asset files (default: auto-detected as patent_filter/asset).
  --output-dir DIR      Directory where output CSVs will be saved (default: patent_filter/output).
  --patent-file FILE    Path to USPTO patent table (default: auto-detected in asset dir).
  --assignee-file FILE  Path to USPTO assignee table (default: auto-detected in asset dir).
  --assignee-col COL    Explicit name of the assignee organization column (default: auto-detected).
  --chunksize INT       Chunk size for streaming large files (default: 100,000 rows).
  --verbose             Enable detailed debug logging.
```

---

### Execution Examples

#### 1. Year-Only Filtering Mode
Filters patents purely by the specified year(s) without company restrictions.

```bash
# Filter for specific patent years (2018, 2019, 2020)
python filter_patents.py --years 2018 2019 2020

# Filter for an inclusive year range (2015 through 2020)
python filter_patents.py --year-range 2015 2020
```

#### 2. Multi-Year Fortune 500 Filtering (Union Rule)
Filters patents for target years and cross-references them against Fortune 500 firms:

```bash
# Filter patents from 2018-2020 matching firms listed in 2018 or 2019 Fortune 500 lists
python filter_patents.py \
  --year-range 2018 2020 \
  --fortune500 2018 2019

# Provide explicit Fortune 500 CSV files
python filter_patents.py \
  --year-range 2018 2020 \
  --fortune500 asset/sample_fortune500_2018.csv asset/sample_fortune500_2019.csv

# Auto-detect combined Fortune 500 CSV and union all available years
python filter_patents.py \
  --years 2020 \
  --fortune500
```

#### 3. Custom Company List Filtering
Filters patents matching an arbitrary company dataset:

```bash
python filter_patents.py \
  --year-range 2018 2020 \
  --custom-companies asset/sample_custom_companies.csv
```

#### 4. High-Confidence Fuzzy Matching
Enables secondary token-set fuzzy matching for minor typographical variations:

```bash
python filter_patents.py \
  --years 2020 \
  --custom-companies asset/sample_custom_companies.csv \
  --fuzzy \
  --fuzzy-threshold 92
```

---

## Output Deliverables

All deliverables are written to `patent_filter/output/`:

### Year-Only Mode Deliverables

| Deliverable File | Description | Schema / Columns |
| :--- | :--- | :--- |
| **`patents_<years>.csv`** | All patents matching the target year criteria. | `patent_id`, `patent_date`, `patent_title`, `num_claims`, etc. |
| **`summary_patents_per_year.csv`** | Patent count aggregated by year. | `year`, `patent_count` |

#### Sample `summary_patents_per_year.csv`:
```csv
year,patent_count
2018,308253
2019,354430
2020,352015
```

---

### Company Filtering Mode Deliverables

| Deliverable File | Description | Schema / Columns |
| :--- | :--- | :--- |
| **`filtered_patents_<years>_matched.csv`** | Matched patents with joined assignee, company, industry, and sector metadata. | Original patent columns + `matched_company_name`, `matched_company_clean`, `industry`, `sector`, `fortune500_years`. |
| **`summary_patents_per_company.csv`** | Total patent count per company, sorted descending by count. | `company_name`, `patent_count`, `industry`, `sector`, `fortune500_years` |
| **`summary_patents_per_industry.csv`** | Total patent count aggregated by industry and group/sector. | `industry`, `patent_count`, `sector` |

#### Sample `summary_patents_per_company.csv`:
```csv
company_name,patent_count,industry,sector,fortune500_years
Apple,3160,"Computers, Office Equipment",Technology,2018; 2019; 2020
Intel,2882,Semiconductors and Other Electronic Components,Technology,2018; 2019; 2020
Qualcomm,2285,Semiconductors and Other Electronic Components,Technology,2018; 2019; 2020
General Electric,1785,Industrial Machinery,Industrials,2018; 2019; 2020
Tesla,1420,Motor Vehicles & Parts,Motor Vehicles & Parts,2019; 2020
```

#### Sample `summary_patents_per_industry.csv`:
```csv
industry,patent_count,sector
Semiconductors and Other Electronic Components,9263,Technology
"Computers, Office Equipment",3652,Technology
Aerospace & Defense,2540,Aerospace & Defense
Industrial Machinery,2374,Industrials
Motor Vehicles & Parts,1890,Motor Vehicles & Parts
```

---

## Entity Resolution & Normalization Logic

Corporate names in USPTO patent grants often differ from official commercial names due to legal suffixes, punctuation, or variations in entity structuring (e.g., `"Apple Inc."` vs `"Apple, Inc."` vs `"APPLE CORP"`).

To guarantee high match recall without false positives:

1. **Case Folding**: All text is converted to lowercase.
2. **Ampersand Expansion**: Characters like `&` are standardized to `" and "` (`"AT&T"` $\rightarrow$ `"at and t"`).
3. **Punctuation Stripping**: Commas, periods, hyphens, and slashes are replaced with whitespace.
4. **Iterative Legal Suffix Elimination**: Legal corporate entity designators are iteratively stripped using regex pattern matching:
   $$\text{Suffixes: } \text{inc, incorporated, corp, corporation, co, company, companies, ltd, limited, llc, lc, llp, lp, plc, gmbh, sa, sas, sarl, spa, bv, nv, ag, pty, holdings, holding, group}$$
   *Iterative stripping* ensures nested suffixes like `"Holdings, Inc."` are completely resolved.
5. **Trailing Connector Removal**: Trailing prepositions or conjunctions (`and`, `of`, `the`, `for`) left behind are stripped.
6. **Leading Article Normalization**: Leading `"The "` is removed (`"The Procter & Gamble Company"` $\rightarrow$ `"procter and gamble"`).
7. **Whitespace Standardization**: Internal whitespace is collapsed to single spaces and trimmed.

### Normalization Examples:
- `"General Motors Corporation"` $\rightarrow$ `"general motors"`
- `"The Procter & Gamble Company"` $\rightarrow$ `"procter and gamble"`
- `"International Business Machines Corporation"` $\rightarrow$ `"international business machines"`
- `"Alphabet Inc."` $\rightarrow$ `"alphabet"`
- `"E.I. Du Pont de Nemours and Company, Inc."` $\rightarrow$ `"e i du pont de nemours"`
