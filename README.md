# Multi-Stage Patent Renewal Prediction with Tabular & Graph Learning

Predicting utility patent maintenance fee renewals across the three statutory USPTO stages using historical bibliometric records, tabular baseline models, and Heterogeneous Information Networks (HIN).

---

## 1. Project Overview & Research Context

Utility patents granted by the United States Patent and Trademark Office (USPTO) require maintenance fee payments to remain in force across three statutory windows:
- **Stage 1 (4th Year)**: Window opens at 3.5 years, deadline at 4.0 years.
- **Stage 2 (8th Year)**: Window opens at 7.5 years, deadline at 8.0 years.
- **Stage 3 (12th Year)**: Window opens at 11.5 years, deadline at 12.0 years.

Failure to pay results in patent abandonment and expiration.

### Core Objectives
1. **Replicate & Evaluate Tabular Baselines**: Implement 23 bibliometric feature variables across strategic, defensiveness, citation velocity, and technological scope dimensions based on Ching-Tun Chang (2010) and Pei-Yu Lin (2015).
2. **Heterogeneous Graph Neural Networks (HIN-GNN)**: Formulate the patent ecosystem as a Heterogeneous Information Network with nodes (*Patents*, *Assignees*, *CPC Technology Classes*) and multi-relational edges (*Cites*, *Assigned-To*, *Classified-In*) to predict multi-stage renewal decisions.
3. **Fortune 500 Corporate Cohort**: Restrict analysis to utility patents granted between **January 1, 2010 and December 31, 2013** belonging to **Fortune 500 companies listed in 2012**. All patents in this window have reached 12+ years of age, ensuring complete ground-truth outcomes without administrative truncation.
4. **Leakage Prevention**: Strictly enforce time-slicing at each stage cutoff (3.5y, 7.5y, 11.5y) so no downstream forward citations or events leak into feature matrices.

---

## 2. Repository Architecture

```
patent/
│
├── asset/                                   # Raw USPTO / PatentsView bulk archives (read-only, gitignored)
│   ├── Fortune_500_Companies.csv            # Corporate rankings, sectors, and industries (1996–2023)
│   ├── g_patent.tsv.zip                     # Patent grant dates, types, titles, abstracts
│   ├── g_assignee_disambiguated.tsv.zip     # Assignee IDs, organizations, and entity types
│   ├── g_application.tsv.zip                # Application filing dates (for approval lag)
│   ├── g_cpc_current.tsv.zip                # CPC technology classification hierarchy
│   ├── patent_document_stats.csv            # Precomputed independent/dependent claim statistics
│   ├── g_foreign_citation.tsv.zip           # Non-US prior art citation records
│   ├── g_us_patent_citation.tsv.zip         # US patent citation pairs and citation velocities
│   ├── MaintFeeEvents_20261005.txt          # USPTO maintenance fee event histories
│   └── MaintFeeEventsDesc_20261005.txt      # Maintenance fee event codes and descriptions
│
├── data/
│   ├── interim/                             # Filtered Parquet tables & crosswalks (gitignored)
│   │   ├── f500_2012_assignee_crosswalk.csv # Audit crosswalk connecting F500 firms to PatentsView IDs
│   │   ├── f500_matched_assignee_ids.parquet# Matched assignee ID lookup filter
│   │   ├── cohort_patents.parquet           # Core F500 utility patents table
│   │   ├── cohort_patent_ids.parquet        # Fast hash lookup of cohort patent IDs
│   │   ├── cohort_patent_assignees.parquet  # Patent-to-company relational links with sector & industry
│   │   ├── filtered_application.parquet     # Cohort application filing dates
│   │   ├── filtered_cpc.parquet             # Cohort CPC classification hierarchies
│   │   ├── filtered_claim_stats.parquet     # Cohort document and claim metrics
│   │   ├── filtered_foreign_citations.parquet# Cohort non-US citations
│   │   ├── filtered_us_citations.parquet    # Forward & backward citation edges connected to cohort
│   │   ├── filtered_assignees.parquet       # Disambiguated assignee records for cohort patents
│   │   ├── summary_industry_representation.csv # Sector and industry representation metrics
│   │   └── summary_company_representation.csv  # Company-level breakdown by grant year (2010–2013)
│   └── processed/                           # Final feature matrices & graph edge lists (planned)
│
├── src/                                     # Data processing & feature engineering pipeline
│   ├── 01_filter_cohort.py                  # F500 entity matching & cohort extraction
│   ├── 02_prune_relational_tables.py        # Stream-filters downstream tables against cohort IDs
│   └── 03_summarize_representation.py       # Computes representation summaries & underrepresented groups
│
├── doc/                                     # Research literature & methodology reference PDFs
│   ├── 運用資料探勘技術協助專利維護決策 - 張景敦.pdf (Chang, 2010)
│   └── 以分類強化方法協助專利維護決策之制定 - 林蓓妤.pdf (Lin, 2015)
│
├── requirements.txt                         # Python environment dependencies
└── README.md                                # Project documentation
```

---

## 3. Data Engineering & Processing Pipeline

All scripts operate under strict memory-safe constraints:
- **Zero Full-Disk Archive Extraction**: Read directly from compressed `.tsv.zip` streams in memory using `zipfile` combined with chunked `pandas` (`chunksize=250_000`).
- **Standardized Compression**: All intermediate outputs are saved in Apache Parquet format using `pyarrow` with `snappy` compression.
- **Progress Tracking**: Real-time progress feedback via `tqdm` on every streaming loop.

### Pipeline Stages

#### Stage 1: Fortune 500 Entity Matching (`src/01_filter_cohort.py`)
1. **Source Loading**: Filters `asset/Fortune_500_Companies.csv` for `year == 2012` (499 companies). Maps missing 2012 `sector` values from the official industry-to-sector taxonomy observed across adjacent dataset years.
2. **Name Standardization**: Standardizes company names via `clean_name()`:
   - Lowercases, removes punctuation, and normalizes whitespace.
   - Strips legal entity suffixes via regex: `\b(inc|corp|corporation|co|company|ltd|limited|llc|plc|group|holdings|sa|nv|ag|lp)\b`.
3. **Assignee Extraction**: Streams `asset/g_assignee_disambiguated.tsv.zip` to extract unique corporate assignees (`assignee_type` in `[2, 3]`).
4. **Hierarchical Matching**:
   - **Tier 1 (Exact Match)**: Matches cleaned Fortune 500 names directly against cleaned assignee names (325 companies matched).
   - **Tier 2 (High-Confidence Fuzzy Match)**: Matches remaining unmatched firms using `rapidfuzz.fuzz.token_set_ratio >= 90.0` with token-subset and sort-ratio guardrails to capture corporate subsidiaries (e.g., *Home Depot Product Authority, LLC*, *Target Brands, Inc.*, *ExxonMobil Chemical Patents*) while preventing false positives.
   - **Result**: Matched **389 Fortune 500 companies** (78.0% of all listed firms, capturing virtually 100% of patent-active firms) across **1,521 distinct PatentsView assignee IDs**.

#### Stage 2: Cohort Extraction (`src/01_filter_cohort.py`)
1. Streams `g_assignee_disambiguated.tsv.zip` to index all patent IDs associated with matched Fortune 500 assignees (1,014,129 historical patents).
2. Streams `g_patent.tsv.zip` (9,454,161 rows) and applies vectorized filtering:
   - `patent_type == 'utility'`
   - `patent_date >= '2010-01-01'` AND `patent_date <= '2013-12-31'`
   - `patent_id` exists in the Fortune 500 patent ID index.
3. Generates [`cohort_patents.parquet`](file:///home/jyc521/Codes/patent/data/interim/cohort_patents.parquet), [`cohort_patent_ids.parquet`](file:///home/jyc521/Codes/patent/data/interim/cohort_patent_ids.parquet), and [`cohort_patent_assignees.parquet`](file:///home/jyc521/Codes/patent/data/interim/cohort_patent_assignees.parquet).

#### Stage 3: Downstream Relational Pruning (`src/02_prune_relational_tables.py`)
Loads `cohort_patent_ids.parquet` into an in-memory hash set for $O(1)$ lookups and filters all auxiliary USPTO tables:
- `g_application.tsv.zip`: Retains filing dates for the exact cohort (129,515 rows).
- `g_cpc_current.tsv.zip`: Retains classification hierarchy (726,102 rows).
- `patent_document_stats.csv`: Retains claim counts, word counts, and claim structure statistics (129,357 rows).
- `g_foreign_citation.tsv.zip`: Retains non-US prior art citations (490,750 rows).
- `g_us_patent_citation.tsv.zip`: Streams 152.6M citations and keeps rows where `patent_id` is in the cohort **OR** `citation_patent_id` is in the cohort (4,223,646 citation edges, preserving both backward and forward graph edges).
- `g_assignee_disambiguated.tsv.zip`: Retains disambiguated assignee details for cohort patents (131,314 rows).

#### Stage 4: Representation Statistics (`src/03_summarize_representation.py`)
Aggregates patenting distributions across sectors, industries, and companies, generating summary reports and identifying underrepresented categories.

---

## 4. Current Dataset Inventory

All processed tables are stored in [`data/interim/`](file:///home/jyc521/Codes/patent/data/interim):

| Table Name | Row Count | Columns | File Size | Description |
| :--- | :---: | :---: | :---: | :--- |
| `cohort_patents.parquet` | **129,515** | 4 | 5.37 MB | Core cohort: `patent_id`, `patent_date`, `patent_title`, `patent_abstract` |
| `cohort_patent_ids.parquet` | **129,515** | 1 | 0.75 MB | Isolated unique patent IDs for lookup |
| `cohort_patent_assignees.parquet` | **129,573** | 5 | 1.16 MB | Relational links: `patent_id`, `assignee_id`, `f500_company_name`, `industry`, `sector` |
| `f500_2012_assignee_crosswalk.csv` | **1,521** | 7 | 0.21 MB | Audit crosswalk connecting 389 F500 firms to PatentsView assignees |
| `f500_matched_assignee_ids.parquet` | **1,521** | 1 | 0.05 MB | Unique matched PatentsView assignee IDs |
| `filtered_application.parquet` | **129,515** | 2 | 1.06 MB | Filing dates (`patent_id`, `filing_date`) |
| `filtered_cpc.parquet` | **726,102** | 5 | 7.24 MB | CPC classifications (`patent_id`, `cpc_section`, `cpc_class`, `cpc_subclass`, `cpc_group`) |
| `filtered_claim_stats.parquet` | **129,357** | 11 | 4.23 MB | Precomputed claim metrics (`pat_clm_ct`, `pat_wrd_ct`, `pat_dep_clm_ct`, etc.) |
| `filtered_foreign_citations.parquet` | **490,750** | 6 | 10.51 MB | Non-US prior art citations |
| `filtered_us_citations.parquet` | **4,223,646** | 8 | 100.76 MB | Forward & backward US citation graph edges |
| `filtered_assignees.parquet` | **131,314** | 4 | 1.68 MB | Disambiguated assignee metadata for cohort patents |
| `summary_industry_representation.csv` | **63** | 4 | 0.01 MB | Aggregation by sector and industry |
| `summary_company_representation.csv` | **256** | 8 | 0.02 MB | Company patent counts by grant year (2010–2013) |

---

## 5. Descriptive Representation & Bibliometric Findings

### Top Sectors by Utility Patent Volume (2010–2013)

| Sector | Unique F500 Companies | Total Utility Patents | Avg Patents / Company |
| :--- | :---: | :---: | :---: |
| **Technology** | 36 | 89,005 | 2,472.4 |
| **Aerospace & Defense** | 13 | 10,553 | 811.8 |
| **Financials** | 28 | 7,824 | 279.4 |
| **Health Care** | 28 | 6,461 | 230.8 |
| **Industrials** | 14 | 4,365 | 311.8 |
| **Chemicals** | 12 | 2,998 | 249.8 |
| **Household Products** | 9 | 2,159 | 239.9 |
| **Motor Vehicles & Parts** | 10 | 1,462 | 146.2 |
| **Telecommunications** | 9 | 825 | 91.7 |
| **Apparel** | 3 | 799 | 266.3 |
| **Retailing** | 7 | 753 | 107.6 |
| **Business Services** | 6 | 686 | 114.3 |
| **Materials** | 10 | 572 | 57.2 |
| **Energy** | 29 | 523 | 18.0 |
| **Food, Beverages & Tobacco** | 17 | 345 | 20.3 |
| **Food & Drug Stores** | 4 | 83 | 20.8 |
| **Transportation** | 6 | 31 | 5.2 |
| **Wholesalers** | 7 | 29 | 4.1 |
| **Media** | 3 | 27 | 9.0 |
| **Hotels, Restaurants & Leisure** | 4 | 17 | 4.2 |
| **Engineering & Construction** | 1 | 6 | 6.0 |

### Top 10 Patenting Corporations (2010–2013 Annual Breakdown)

| Company Name | Primary Industry | 2010 | 2011 | 2012 | 2013 | Total Patents |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **International Business Machines Corp.** | Information Technology Services | 5,892 | 6,177 | 6,479 | 6,813 | **25,361** |
| **Microsoft Corporation** | Computer Software | 3,091 | 2,307 | 2,619 | 2,663 | **10,680** |
| **General Electric Company** | Diversified Financials | 1,223 | 1,450 | 1,652 | 1,738 | **6,063** |
| **Intel Corporation** | Semiconductors & Components | 1,653 | 1,244 | 1,293 | 1,460 | **5,650** |
| **Hewlett-Packard Company** | Computers, Office Equipment | 1,493 | 1,336 | 1,407 | 1,375 | **5,611** |
| **Qualcomm Incorporated** | Communications Equipment | 656 | 928 | 1,291 | 2,105 | **4,980** |
| **Broadcom Corporation** | Semiconductors & Components | 958 | 1,164 | 1,157 | 1,084 | **4,363** |
| **Apple, Inc.** | Computers, Office Equipment | 562 | 675 | 1,137 | 1,779 | **4,153** |
| **Xerox Corporation** | Computers, Office Equipment | 857 | 880 | 1,052 | 1,013 | **3,802** |
| **Micron Technology, Inc.** | Semiconductors & Components | 910 | 945 | 916 | 1,023 | **3,794** |

### Underrepresented Groups & Filter Impact

Excluding the 32 sparse industries (`num_companies < 3` OR `total_patents < 50`) preserves the vast majority of patent volume while eliminating severe categorical imbalance:

| Cohort Window | Dataset Version | Companies | Industries | Utility Patents | Patents Dropped |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **2010–2013** | **Full Baseline** | 256 | 63 | 129,515 | — |
| **2010–2013** | **Cleaned (Underrepresented Excluded)** | **189** | **31** | **125,914** | **3,601 (2.8%)** |
| **2011–2013** | **Full Baseline** | 245 | 63 | 100,360 | — |
| **2011–2013** | **Cleaned (Underrepresented Excluded)** | **181** | **31** | **97,563** | **2,797 (2.8%)** |

---

## 6. Execution & Reproducibility

### Installation
```bash
pip install -r requirements.txt
```

### Running the Pipeline End-to-End

1. **Cohort Extraction & Entity Crosswalk**:
   ```bash
   python3 src/01_filter_cohort.py \
     --f500-csv asset/Fortune_500_Companies.csv \
     --assignee-tsv asset/g_assignee_disambiguated.tsv.zip \
     --patent-tsv asset/g_patent.tsv.zip \
     --interim-dir data/interim \
     --chunksize 250000
   ```

2. **Downstream Relational Table Pruning**:
   ```bash
   python3 src/02_prune_relational_tables.py \
     --asset-dir asset \
     --interim-dir data/interim \
     --cohort-ids data/interim/cohort_patent_ids.parquet \
     --chunksize 250000 \
     --table all
   ```

3. **Representation Summaries & Diagnostics**:
   ```bash
   python3 src/03_summarize_representation.py \
     --cohort-assignees data/interim/cohort_patent_assignees.parquet \
     --cohort-patents data/interim/cohort_patents.parquet \
     --output-industry-csv data/interim/summary_industry_representation.csv \
     --output-company-csv data/interim/summary_company_representation.csv
   ```

---

## 7. Upcoming Roadmap

- **Step 3 (`src/03_extract_labels.py`)**: Parse USPTO maintenance fee payment events from `asset/MaintFeeEvents_20261005.txt` into binary renewal targets ($Y_1, Y_2, Y_3 \in \{0, 1\}$) across Stages 1, 2, and 3.
- **Step 4 (`src/04_engineer_features.py`)**: Compute time-sliced Chang (2010) and Lin (2015) tabular features (23 dimensions across technological scope, citation velocities, claim complexities, and defensiveness metrics) enforced at cutoffs $t \in \{3.5\text{y}, 7.5\text{y}, 11.5\text{y}\}$.
- **Step 5 (`src/05_build_graph.py`)**: Assemble heterogeneous graph edge lists and node feature matrices for PyTorch Geometric (PyG).