# ZZPH — Zigzag Persistent Homology Pipeline

Longitudinal TDA pipeline for the **2016–2025 Philippine senatorial elections**.

Computes province-level **zigzag persistent homology** across four election cycles and clusters the resulting topological features with **BallMapper**, coloured by cross-cycle averages of socioeconomic covariates.

---

## Project layout

```
ph-senate-tda/
    run.py                      <- top-level runner (all years + ZZPH)
    2016/src/  2019/src/  2022/src/  2025/src/   <- individual year pipelines
    ZZPH/
        src/
            config.py           <- all paths, constants, colour configs
            preprocessing.py    <- province harmonisation + outlier removal
            topology.py         <- zigzag filtration + H0 feature extraction
            socioeconomic.py    <- longitudinal socioeconomic data loaders
            ballmapper.py       <- BallMapper graph, tables, scatter plots
            robustness.py       <- epsilon-robustness diagnostics
            visualization.py    <- synthetic zigzag demo figure (cell 7)
            main.py             <- master execution controller
        data/
            processed/          <- 2016_4features.csv … 2025_4features.csv
            socioeconomic data/
                ARI Dependency Rate/
                Literacy Rate/
                Poverty Incidence among Families/
        results/
            1_preprocessed_data/
            2_knn_radius_optimization/
            3_zigzag_results/
            4_ballmapper_clusters/
                robustness_caches/
```

---

## Quickstart

```bash
# From ph-senate-tda/ — run everything
python3 run.py

# Quick smoke-test (100 reps, skip stepwise stability, year pipelines only)
python3 run.py --reps 100 --fast --skip-zzph

# ZZPH pipeline only
python3 run.py --zzph-only

# Specific years only, then ZZPH
python3 run.py --years 2019 2022 --reps 100

# ZZPH module only (from ZZPH/src/)
cd ZZPH/src && python main.py
```

---

## Data sources

| Layer | Source files | Years averaged | Column produced |
|---|---|---|---|
| Poverty incidence | `Poverty Incidence among Families 2015 to 2018.csv` + `… 2018 to 2023.csv` | 2015, 2018, 2021, 2023 | `Mean_Poverty_Incidence` |
| Functional literacy | `2024 FLEMMS Statistical Tables Press Release 2.xlsx` (Table 3) | 2024 (single year) | `Basic_Literacy_Rate` |
| IRA / NTA dependency | `By-LGU-ARI-and-Dependencies-{2016,2019,2022,2025}.xlsx` | 2016, 2019, 2022, 2025 | `Mean_IRA_Dependency` |
| Admin vote share | Computed from the cleaned precinct DataFrames | 2016, 2019, 2022, 2025 | `Mean_Admin_Vote` |

All four dimensions are **averaged across available years** rather than pinned to a single survey date, reflecting the longitudinal nature of the zigzag topology.

---

## Methodology

### Province harmonisation

Before any computation the four election CSVs are normalised to a shared canonical province set:

| Original name | Canonical name | Reason |
|---|---|---|
| `COMPOSTELA VALLEY` | `DAVAO DE ORO` | Administrative rename (2019) |
| `COTABATO (NORTH COT.)` / `COTABATO` | `NORTH COTABATO` | Abbreviation variants |
| `DAVAO (DAVAO DEL NORTE)` | `DAVAO DEL NORTE` | Abbreviation variant |
| `SAMAR` / `SAMAR (WESTERN SAMAR)` | `WESTERN SAMAR` | Rename |
| `MAGUINDANAO DEL NORTE` / `DEL SUR` | `MAGUINDANAO` | Reunify 2022 split for time-series continuity |
| `SPECIAL GEOGRAPHIC AREA` | `NORTH COTABATO` | 63-barangay BARMM zone absorbed into parent |
| `TAGUIG - PATEROS` | `NCR - FOURTH DISTRICT` | Consistent NCR labelling |
| `NCR - MANILA` (short form) | `NATIONAL CAPITAL REGION - MANILA` | Consistent NCR labelling |

Only provinces present in **all four** election years enter the zigzag computation (intersection, not union).

### Zigzag filtration

For each province, the filtration follows seven steps:

```
X₁₆  →  X₁₆∪X₁₉  ←  X₁₉  →  X₁₉∪X₂₂  ←  X₂₂  →  X₂₂∪X₂₅  ←  X₂₅
 t=0       t=1        t=2       t=3        t=4       t=5        t=6
```

Each `Xᵢ` is a Vietoris-Rips complex built from all CLR simplex vectors (Admin, Opposition, Independent, Non-Aligned) recorded for that province in election year `i`.  Union complexes are built on the joint point cloud.

Per-province radii are set automatically via a **k-NN distance percentile** (`k=5`, 90th percentile of the k-th nearest-neighbour distance across all four years combined), with a minimum floor of 0.10.  Radii are cached in `2_knn_radius_optimization/optimal_radii_provincial.json`.

Zigzag persistence is computed with **Dionysus 2** (`zigzag_homology_persistence`).  H0 (connected components) is the sole homology dimension used.

### H0 features extracted per province

| Feature | Description |
|---|---|
| `H0_PersistentEntropy` | Normalised Shannon entropy of H0 bar lifetimes — measures distributional complexity of the connected-component structure across the four cycles |
| `H0_L2_Norm` | L2 norm of the persistence landscape vector (10 landscapes, resolution 700, range [0, 7]) — measures evolutionary rigidity; large values indicate topology that barely changes |
| `H0_Total_Persistence` | Sum of all H0 bar lifetimes — overall fragmentation across the four cycles |

All three features are z-score standardised before being passed to BallMapper.

### BallMapper

A single BallMapper graph is built from the standardised 3-D feature space at `ε = BM_EPSILON` (default 0.8).  Each node is coloured by the mean value of one socioeconomic covariate across member provinces.

---

## Outputs

### `results/1_preprocessed_data/`

| File | Description |
|---|---|
| `df_16_prov_clean.csv` … `df_25_prov_clean.csv` | Cleaned precinct-level DataFrames per year |
| `outlier_removal_report.csv` / `.tex` | Province-level Isolation Forest removal statistics (accumulated across all four years) |

### `results/2_knn_radius_optimization/`

| File | Description |
|---|---|
| `optimal_radii_provincial.json` | Per-province VR radius (float, one entry per common province) |

### `results/3_zigzag_results/`

| File | Description |
|---|---|
| `provincial_zigzag_features.csv` | One row per province: `H0_PersistentEntropy`, `H0_L2_Norm`, `H0_Total_Persistence` |

### `results/4_ballmapper_clusters/`

| File | Description |
|---|---|
| `unified_bm_summary_eps_{ε}.csv` | Master node table: all covariates + all TDA metrics + province and region lists |
| `unified_bm_summary_eps_{ε}.tex` | LaTeX version of the master table |
| `unified_intersection_matrix_eps_{ε}.csv` | Node × node matrix of shared province names |
| `bm_summary_{covariate}_eps_{ε}.csv` | Compact 4-column table (Node ID, Size, covariate mean, Provinces) |
| `bm_summary_{covariate}_eps_{ε}.tex` | Two-column parallel LaTeX layout of the compact table |
| `bm_graph_eps_{ε}_{mode}.png` | Coloured spring-layout network graph |
| `scatter_{mode}_{metric}.jpg` | Plotly scatter: covariate vs. scaled TDA metric with Pearson r annotation |
| `feature_correlation_heatmap.png` | 3×3 Pearson correlation heatmap of raw TDA features |
| `ballmapper_num_balls_robustness.jpg` | ε-sweep: mean ± 95% CI for number of nodes |
| `ballmapper_ball_size_robustness.jpg` | ε-sweep: mean ± 95% CI for average node size |

### `results/4_ballmapper_clusters/robustness_caches/`

| File | Description |
|---|---|
| `epsilon_robustness.csv` | Raw global robustness sweep data |
| `robustness_cache_{col}.csv` | Per-covariate comparative robustness sweep data |
| `{col}_numnodes.jpg` | Comparative robustness: number of nodes (vulnerable vs baseline) |
| `{col}_avgnodesize.jpg` | Comparative robustness: average node size (vulnerable vs baseline) |

---

## Module reference

### `config.py`

All file paths, algorithm hyperparameters, and `COLOR_CONFIGS` dictionary.  Edit this file to adjust `BM_EPSILON`, `NUM_REPETITIONS`, `KNN_K`, or input file locations.

Key constants:

| Constant | Default | Notes |
|---|---|---|
| `BM_EPSILON` | `0.8` | BallMapper radius |
| `NUM_REPETITIONS` | `10000` | Bootstrap reps for robustness; use `100` for testing |
| `KNN_K` | `5` | k for per-province radius optimisation |
| `KNN_PERCENTILE` | `90` | Distance percentile used as the VR radius |
| `OUTLIER_CONTAMINATION` | `0.01` | IsolationForest contamination rate |
| `LANDSCAPE_NUM` | `10` | Number of persistence landscapes |
| `LANDSCAPE_RES` | `700` | Landscape discretisation resolution |

### `preprocessing.py`

`process_dataframe(df)` — harmonises province names and drops OAV rows.
`remove_outliers_provincial(df_list, …)` — joint IsolationForest across all four years.
`log_province_coverage(…)` — two-stage drop audit printed to stdout.

### `topology.py`

`build_radius_cache(…)` — load or compute per-province VR radii.
`compute_zigzag_features(…)` — parallelised zigzag computation with caching.
`run_provincial_zigzag(…)` — single-province seven-step filtration.
`extract_topological_metrics(dgms)` — convert H0 diagrams to three scalars.

### `socioeconomic.py`

`load_poverty_data()` — merge two PSA CSV files, average 2015/2018/2021/2023.
`load_literacy_data()` — extract province-level functional literacy from FLEMMS XLSX.
`load_ira_data()` — load four ARI/NTA XLSX files and compute cross-year means.
`append_admin_vote_share(…)` — population-weighted admin vote share per province.
`load_and_merge_socioeconomic(df_topo)` — full merge with strict NaN filtering.

### `ballmapper.py`

`build_unified_ballmapper_structure(…)` — builds BallMapper, writes all tables and intersection matrix.
`visualize_ballmapper_coloring(…)` — coloured network graph (matplotlib).
`generate_topology_scatter_plots(…)` — Pearson-annotated scatter plots (plotly).
`generate_correlation_heatmap(…)` — seaborn TDA feature correlation heatmap.

### `robustness.py`

`run_epsilon_robustness(df_features)` — global ε sweep.
`run_epsilon_comparative_robustness(…)` — per-covariate vulnerable vs. baseline ε sweep.

### `visualization.py`

`plot_zigzag_demo(…)` — two-row publication figure: five Vietoris-Rips complexes (row 1) + H0 barcode (row 2).  Self-contained; does not require real election data.

---

## `run.py` (project root)

```
usage: run.py [-h] [--years {2016,2019,2022,2025} ...] [--skip-zzph]
              [--zzph-only] [--fast] [--reps N] [--epsilon E]

  --years       Which year pipelines to run (default: all four)
  --skip-zzph   Skip the ZZPH longitudinal pipeline
  --zzph-only   Run only ZZPH (skip year pipelines)
  --fast        Set RUN_STEPWISE_STABILITY=False in each year config
  --reps N      Override NUM_REPETITIONS in every pipeline
  --epsilon E   Override BM_EPSILON in every pipeline
```

Each year's `src/` directory is added to `sys.path` for the duration of its run and removed afterwards; all loaded modules are evicted so the next year's `config.py` does not shadow the previous one.

---

## Dependencies

```
numpy  pandas  scipy  scikit-learn  tqdm  joblib
gudhi  dionysus  persim  pyballmapper
networkx  matplotlib  plotly  kaleido  seaborn
openpyxl  ipython
```

Install with:

```bash
pip install -r requirements.txt
```
