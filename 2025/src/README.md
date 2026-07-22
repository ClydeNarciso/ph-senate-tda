# 2025 Electoral Topology / Socioeconomic Analysis

Modularized TDA + BallMapper pipeline for the **2025 Philippine midterm election**.

## Structure

```
config.py           Paths, constants, year-specific data flags
preprocessing.py    Precinct cleaning + Isolation Forest outlier removal
topology.py         H0 persistence intervals (Vietoris-Rips) + 4D TDA features
socioeconomic.py    Province-name standardization + data loaders & merge
ballmapper.py       BallMapper graph construction, visualization, intersection matrix
visualization.py    Topology-vs-covariate scatterplots (plotly), correlation heatmap
reporting.py        Styled summary tables + LaTeX export
robustness.py       Landscape convergence + epsilon-robustness diagnostics
main.py             Orchestrates the full pipeline end-to-end
```

## Directory layout

```
data/
  processed/
    2025_4features.csv
  socioeconomic data/
    ARI Dependency Rate/
      By-LGU-ARI-and-Dependencies-2025.xlsx
    Ethnicity/
      provincial_dominant_ethnicity_psa2020.csv
    Poverty Incidence among Families/
      PSA_Full_Year_Poverty_Statistics_-_2018_2021_2023.csv
    Dynasty Proxy Variables/
      DynastyProxyVariables.csv
    FLEMMS/
      2024_FLEMMS_Statistical_Tables_Press_Release_2.xlsx
results/
  figures/                  <- all PNG/JPG plots
  persistence_cache/        <- cached intermediate CSVs (resumable)
    comparative_robustness/
  tables/                   <- final summary tables (CSV/LaTeX)
```

## Data source decisions

| Dataset | Source year used | Reason |
|---------|-----------------|--------|
| Precinct features | 2025 | Actual 2025 election data |
| NTA dependency | FY2025 (XLSX) | Actual 2025 BLGF data; column renamed from IRA DEPENDENCY to NTA DEPENDENCY in this release |
| Poverty incidence | **2023** (primary) | Closest PSA survey year to 2025; 2021 used only as fallback for Maguindanao del Norte/Sur which have no separate 2023 entries |
| Functional literacy | **2024** (FLEMMS) | New dimension added for 2025; no equivalent source exists in prior pipeline years |
| Dynasty HHI / GINI | **2019** (proxy) | Last available year in the dynasty dataset; no 2022 or 2025 data exists |
| Ethnicity | 2020 PSA census | No newer subnational census available |

## Key differences from the 2022 pipeline

1. **NTA replaces IRA.** The 2025 BLGF XLSX uses `NTA DEPENDENCY` (National Tax Allotment) in place of the earlier `IRA DEPENDENCY` column. The sheet name also changed from `FY2022` to `ARI 2025`. All other parsing logic — header row index, decimal fraction format, NCR city-to-district redistribution — is identical.

2. **New covariate: FLEMMS 2024 functional literacy.** The 2024 Functional Literacy, Education and Mass Media Survey provides province-level functional literacy rates (Table 3, both sexes, %). Provinces with Highly Urbanized Cities (HUCs) are listed as separate rows (e.g. `Benguet (Excluding City of Baguio)` + `City of Baguio`); these are averaged back to a single province figure in `socioeconomic.py`. A dedicated `Literacy` BallMapper coloring mode and comparative robustness run are added in `main.py`.

3. **Non-province artefact rows.** The 2025 feature file contains two rows that do not correspond to geographic provinces and are dropped in `preprocessing.py`:
   - `LAV` (single-precinct artefact under Region `LAV`)
   - `SPECIAL GEOGRAPHIC AREA` (59-precinct BARMM administrative zone)

4. **Updated OAV pseudo-province names.** The 2025 set is `ASIA PACIFIC`, `EUROPE`, `MIDDLE EAST AND AFRICAS`, `NORTH AND LATIN AMERICAS` (note: `NORTH AND LATIN AMERICAS` with trailing `S`, and `ASIA PACIFIC` instead of bare `ASIA`).

5. **NCR label format.** The 2025 feature file uses the full `NATIONAL CAPITAL REGION - X` format (same as 2019 and 2022). There is no `TAGUIG - PATEROS` row — Taguig and Pateros are folded into the NCR Fourth District in the 2025 file.

6. **Hierarchical poverty CSV.** The PSA source file uses a dot-indented hierarchy (`....Province`, `......Sub-province`). Province rows are extracted by matching four or more leading dots, then names are cleaned and standardised. Maguindanao del Norte and del Sur appear at the six-dot level and have no separate 2023 estimates; their 2021 values are used as fallback.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Controls in `config.py`

| Setting | Default | Notes |
|---------|---------|-------|
| `RUN_STEPWISE_STABILITY` | `True` | Set `False` to skip landscape convergence (can be slow; results cached after first run) |
| `NUM_REPETITIONS` | `1000` | Bootstrap reps for epsilon-robustness; use `100` for testing |
| `BM_EPSILON` | `0.6` | BallMapper radius |
| `POVERTY_PRIMARY_YEAR` | `"2023 Poverty Incidence"` | Primary poverty column name |
| `POVERTY_FALLBACK_YEAR` | `"2021 Poverty Incidence"` | Used when 2023 value is missing (Maguindanao del Norte/Sur) |
| `DYNASTY_YEAR` | `2019` | Last available year in dynasty dataset |
| `FLEMMS_COL_LABEL` | `"Functional_Literacy_Rate"` | Column name after loading FLEMMS data |

## Outputs

**Figures** (`results/figures/`):
- `bm_eps_{eps}_{mode}.jpg` — BallMapper network per covariate
- `scatter_{mode}_{metric}.png` — node scatterplots vs. topological metrics
- `feature_correlation_heatmap.png` — H0 feature correlation matrix
- `stepwise_stability.png` — landscape convergence (if enabled)
- `ballmapper_num_balls_robustness.jpg` / `ballmapper_ball_size_robustness.jpg`
- `{col}_numnodes.jpg` / `{col}_avgnodesize.jpg` — comparative robustness

**Tables** (`results/tables/`):
- `2025_outlier_report.csv` / `.tex`
- `summary_table_{mode}.csv` / `.tex` (one per covariate mode: Admin_Share, Poverty, Literacy, Ethnicity, Dynasty_HHI, Inequality_GINI)
- `summary_table_master.csv` — all metrics combined
- `2025_intersection_matrix_master.csv`
- `provincial_topological_summaries_all.csv`
