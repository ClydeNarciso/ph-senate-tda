# 2016 Electoral Topology / Socioeconomic Analysis

Modularized TDA + BallMapper pipeline for the **2016 Philippine general election**.

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
    2016_4features.csv
  socioeconomic data/
    ARI Dependency Rate/
      By-LGU-ARI-and-Dependencies-2016.xlsx
    Ethnicity/
      provincial_dominant_ethnicity_psa2020.csv
    Poverty Incidence among Families/
      Poverty_Incidence_among_Families_2015_to_2018.csv
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
| Precinct features | 2016 | Actual 2016 election data |
| IRA dependency | FY2016 (XLSX) | Actual 2016 BLGF data |
| Poverty incidence | **2015** (primary) | Closest PSA survey year before 2016; 2018 used only as fallback for Davao Occidental, which was created from Davao del Sur in 2016 and has no 2015 estimate |
| Dynasty HHI / GINI | **2016** (actual) | The only pipeline year with contemporaneous dynasty data; 2016 data exists in the dataset |
| Ethnicity | 2020 PSA census | No subnational census closer to 2016 is available |

## Key differences from the 2019 / 2022 pipelines

1. **ARMM, not BARMM.** The 2016 data predates the Bangsamoro transition; the Muslim Mindanao region is labelled `ARMM` throughout.

2. **Pre-split province names.** Several provinces used different names in 2016 and are stored as such in the feature file. The socioeconomic loaders map between these and the names each source dataset uses:

   | Feature file name | Later / ARI name | Standardised to |
   |---|---|---|
   | `COMPOSTELA VALLEY` | `DAVAO DE ORO` (renamed 2019) | kept as `COMPOSTELA VALLEY` |
   | `COTABATO (NORTH COT.)` | `NORTH COTABATO` | feature-file form used as canonical |
   | `DAVAO (DAVAO DEL NORTE)` | `DAVAO DEL NORTE` | feature-file form used as canonical |
   | `MAGUINDANAO` | split into del Norte / del Sur in 2022 | kept as `MAGUINDANAO` |
   | `SAMAR (WESTERN SAMAR)` | `SAMAR` / `WESTERN SAMAR (SAMAR)` | feature-file form used as canonical |

3. **OAV pseudo-province names differ.** The 2016 set is `ASIA`, `EUROPE`, `MIDDLE EAST AND AFRICAS`, `NORTH AND LATIN AMERICA` (note: `ASIA` not `ASIA PACIFIC`; `NORTH AND LATIN AMERICA` without trailing `S`). Both the OAV region check and the pseudo-province name set in `config.py` reflect these 2016-specific labels.

4. **Wide-format poverty CSV.** Unlike the semicolon-delimited or hierarchical CSVs used in later pipelines, the 2015–2018 source file is a wide-format table with footnote-heavy province names (`Apayao* c/`, `Guimarasa/, c/`). The parser skips 7 header rows, extracts columns 0, 3, and 4, and uses a three-step regex to strip all footnote variants without clipping province names.

5. **2016 dynasty data is actual, not a proxy.** The dynasty dataset includes 2016 rows, unlike every other pipeline year which uses 2019 as a proxy. Five provinces are absent from the 2016 dynasty data (Compostela Valley, Davao Occidental, Dinagat Islands, Sulu, Tawi-Tawi) and are imputed from regional means.

6. **TAGUIG - PATEROS present.** The 2016 NCR layout matches 2022: four district labels plus a separate `TAGUIG - PATEROS` row. All socioeconomic loaders replicate the NCR 4th district value for `TAGUIG - PATEROS`.

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
| `POVERTY_PRIMARY_YEAR` | `"2015 Poverty Incidence"` | Primary poverty column name |
| `POVERTY_FALLBACK_YEAR` | `"2018 Poverty Incidence"` | Used when 2015 value is missing (Davao Occidental) |
| `DYNASTY_YEAR` | `2016` | Actual election-year dynasty data |

## Outputs

**Figures** (`results/figures/`):
- `bm_eps_{eps}_{mode}.jpg` — BallMapper network per covariate
- `scatter_{mode}_{metric}.png` — node scatterplots vs. topological metrics
- `feature_correlation_heatmap.png` — H0 feature correlation matrix
- `stepwise_stability.png` — landscape convergence (if enabled)
- `ballmapper_num_balls_robustness.jpg` / `ballmapper_ball_size_robustness.jpg`
- `{col}_numnodes.jpg` / `{col}_avgnodesize.jpg` — comparative robustness

**Tables** (`results/tables/`):
- `2016_outlier_report.csv` / `.tex`
- `summary_table_{mode}.csv` / `.tex` (one per covariate mode)
- `summary_table_master.csv` — all metrics combined
- `2016_intersection_matrix_master.csv`
- `provincial_topological_summaries_all.csv`
