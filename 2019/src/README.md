# 2019 Electoral Topology / Socioeconomic Analysis

Modularized version of `2019_analysis.ipynb`.

## Structure

```
config.py           Paths & shared constants (single source of truth)
preprocessing.py     Precinct cleaning + Isolation Forest outlier removal
topology.py          H0 persistence intervals (Vietoris-Rips) + 4D TDA features
socioeconomic.py     Province-name standardization + poverty/IRA/ethnicity/dynasty loaders & merge
ballmapper.py         BallMapper graph construction, node visualization, intersection matrix
visualization.py     Topology-vs-covariate scatterplots (plotly), correlation heatmap
reporting.py          Styled summary tables + LaTeX export
robustness.py         Landscape convergence + epsilon-robustness diagnostics (single & comparative)
main.py               Orchestrates the full pipeline end-to-end
```

## Directory layout

```
data/
  processed/                     <- put 2019_4features.csv here
  socioeconomic data/
    Poverty Incidence among Families/
    ARI Dependency Rate/
    Ethnicity/
    Dynasty Proxy Variables/
results/
  figures/                       <- all PNG/JPG plots
  persistence_cache/             <- cached intermediate CSVs (resumable)
    comparative_robustness/
  tables/                        <- final summary tables (CSV/LaTeX)
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Re-running `main.py` will transparently reuse anything already cached in
`results/persistence_cache/` (cleaned data, H0 intervals, topological
features, convergence and robustness runs) — delete the relevant cache
file to force recomputation.

## Notes

- `config.py` creates all output directories on import.
- Update the file names under `data/socioeconomic data/` if your source
  filenames differ; only the paths in `config.py` need to change.
- `NUM_REPETITIONS` (robustness bootstrap repeats) and `BM_EPSILON`
  (chosen BallMapper radius) are configured in `config.py`.
