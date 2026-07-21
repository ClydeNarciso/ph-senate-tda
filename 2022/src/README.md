# 2022 Electoral Topology / Socioeconomic Analysis

Modularized TDA + BallMapper pipeline for the **2022 Philippine midterm election**.

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
run.py              CLI entry-point with --fast / --reps / --epsilon flags
```

## Directory layout

```
data/
  processed/
    2022_4features.csv
  socioeconomic data/
    ARI Dependency Rate/
      By-LGU-ARI-and-Dependencies-2022.xlsx
    Ethnicity/
      provincial_dominant_ethnicity_psa2020.csv
    Poverty Incidence among Families/
      Poverty_Incidence_among_Families_2018_to_2023.csv
    Dynasty Proxy Variables/
      DynastyProxyVariables.csv
results/
  figures/                  <- all PNG/JPG plots
  persistence_cache/        <- cached intermediate CSVs (resumable)
    comparative_robustness/
  tables/                   <- final summary tables (CSV/LaTeX)
```

## Data source decisions

| Dataset | Source year used | Reason |
|---------|-----------------|--------|
| Precinct features | 2022 | Actual 2022 election data |
| IRA dependency | FY2022 (XLSX) | Actual 2022 BLGF data |
| Poverty incidence | **2021** (primary) | Closest PSA survey year before 2022; 2023 used only as fallback for Maguindanao del Norte/Sur which had no 2021 estimate |
| Dynasty HHI / GINI | **2019** (proxy) | Last available year in the dynasty dataset; no 2022 data exists |
| Ethnicity | 2020 PSA census | No newer subnational census available |

## Key differences from the 2019 pipeline

1. **OAV exclusion is broader.** The 2022 feature file contains both a
   `Region == 'OAV'` block *and* overseas pseudo-province rows (AMERICAS,
   ASIA PACIFIC, EUROPE, MIDDLE EAST AND AFRICAS) scattered under
   non-OAV regions. Both are removed in `preprocessing.py`.

2. **IRA source is XLSX, not CSV.** `By-LGU-ARI-and-Dependencies-2022.xlsx`
   has a 7-row preamble before the header; IRA DEPENDENCY values are
   stored as decimal fractions (0–1) rather than percent strings.
   NCR cities are all listed under PROVINCE = "METRO MANILA" and are
   redistributed to the four NCR district labels + TAGUIG-PATEROS by
   LGU NAME in `socioeconomic.py`.

3. **Maguindanao split.** The 2022 election separated MAGUINDANAO into
   MAGUINDANAO DEL NORTE and MAGUINDANAO DEL SUR.  Socioeconomic
   datasets (dynasty, ethnicity) still list consolidated MAGUINDANAO;
   both sub-provinces receive the parent value via replication.

4. **Province label format changed for NCR.** The 2022 feature file uses
   `NCR - MANILA`, `NCR - SECOND DISTRICT`, etc. (hyphen, no "NATIONAL
   CAPITAL REGION" prefix), whereas the 2019 file used
   `NATIONAL CAPITAL REGION - MANILA` etc.  All standardisation maps
   in `socioeconomic.py` target the 2022 format.

## Usage

### Quick test
```bash
pip install -r requirements.txt
python run.py --fast --reps 100
```

### Full run
```bash
python run.py
# or: python main.py
```

## Controls in `config.py`

| Setting | Default | Notes |
|---------|---------|-------|
| `RUN_STEPWISE_STABILITY` | `True` | Set `False` to skip landscape convergence (can be slow; results cached after first run) |
| `NUM_REPETITIONS` | `1000` | Bootstrap reps for epsilon-robustness; use `100` for testing |
| `BM_EPSILON` | `0.6` | BallMapper radius |
| `POVERTY_PRIMARY_YEAR` | `"2021 Poverty Incidence"` | Primary poverty column name in CSV |
| `POVERTY_FALLBACK_YEAR` | `"2023 Poverty Incidence"` | Used when 2021 value is missing |
| `DYNASTY_YEAR` | `2019` | Last available year in dynasty dataset |

## Outputs

**Figures** (`results/figures/`):
- `bm_eps_{eps}_{mode}.jpg` — BallMapper network per covariate
- `scatter_{mode}_{metric}.png` — node scatterplots vs. topological metrics
- `feature_correlation_heatmap.png` — H0 feature correlation matrix
- `stepwise_stability.png` — landscape convergence (if enabled)
- `ballmapper_num_balls_robustness.jpg` / `ballmapper_ball_size_robustness.jpg`
- `{col}_numnodes.jpg` / `{col}_avgnodesize.jpg` — comparative robustness

**Tables** (`results/tables/`):
- `2022_outlier_report.csv` / `.tex`
- `summary_table_{mode}.csv` / `.tex` (one per covariate mode)
- `summary_table_master.csv` — all metrics combined
- `2022_intersection_matrix_master.csv`
- `provincial_topological_summaries_all.csv`
