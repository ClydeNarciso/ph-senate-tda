"""
Central configuration for the 2022 electoral-topology / socioeconomic
analysis pipeline: input/output paths and shared constants.

Key differences from the 2019 pipeline
----------------------------------------
- Input data: 2022_4features.csv
- ARI/IRA source: XLSX (FY2022 sheet), not CSV, with IRA DEPENDENCY as
  a decimal fraction (not a percent string). Province is in column
  "PROVINCE"; NCR cities are all listed under "METRO MANILA" and must
  be redistributed to the four NCR districts + TAGUIG-PATEROS using
  LGU NAME.
- Poverty: semicolon-delimited CSV with 2018 / 2021 / 2023 data.
  No 2022 year exists, so 2021 (closest prior year to the election)
  is used as the primary poverty column. 2023 is used to fill the two
  provinces (Maguindanao del Norte / del Sur) missing from the 2021 survey.
- Dynasty (HHI / GINI): last available year is 2019. Used as the proxy
  for 2022 dynasty concentration (no 2022 data exists).

Directory layout
-----------------
data/processed/                    -> 2022_4features.csv
data/socioeconomic data/
  ARI Dependency Rate/             -> By-LGU-ARI-and-Dependencies-2022.xlsx
  Ethnicity/                       -> provincial_dominant_ethnicity_psa2020.csv
  Poverty Incidence among Families/-> Poverty_Incidence_among_Families_2018_to_2023.csv
  Dynasty Proxy Variables/         -> DynastyProxyVariables.csv
results/figures/                   -> all plots (PNG/JPG)
results/persistence_cache/         -> cached intermediate computations (CSV)
results/tables/                    -> final summary tables (CSV/LaTeX)
"""
from pathlib import Path
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
results_dir = BASE_DIR / 'results' / 'figures'

DATA_DIR = Path(BASE_DIR / "data/processed")
SOCIOECONOMIC_DIR = Path(BASE_DIR / "data/socioeconomic data")

RESULTS_DIR = BASE_DIR / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "persistence_cache"
COMPARATIVE_CACHE_DIR = CACHE_DIR / "comparative_robustness"
TABLES_DIR = RESULTS_DIR / "tables"

for _dir in (FIGURES_DIR, CACHE_DIR, COMPARATIVE_CACHE_DIR, TABLES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# INPUT FILES
# ---------------------------------------------------------------------------

DATA_FILE = DATA_DIR / "2022_4features.csv"

POVERTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / 'Poverty_Incidence_among Families_2018_to_2023.csv'
)
# Closest available poverty year to 2022. 2021 used as primary;
# 2023 used only to fill Maguindanao del Norte / del Sur (missing in 2021).
POVERTY_PRIMARY_YEAR = "2021 Poverty Incidence"
POVERTY_FALLBACK_YEAR = "2023 Poverty Incidence"
POVERTY_COL_LABEL = "2021 Poverty Incidence"   # label used throughout

ARI_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "ARI Dependency Rate"
    / "By-LGU-ARI-and-Dependencies-2022.xlsx"
)
ARI_SHEET_NAME = "FY2022"
ARI_HEADER_ROW = 6          # 0-indexed row that contains column names
# IRA DEPENDENCY values in this XLSX are already decimal fractions (0–1),
# unlike the 2019 CSV which stored them as percent strings.
ARI_VALUE_IS_FRACTION = True

ETHNICITY_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "Ethnicity" / "provincial_dominant_ethnicity_psa2020.csv"
)
DYNASTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Dynasty Proxy Variables" / "DynastyProxyVariables.csv"
)
# Last year in dynasty dataset; used as 2022 proxy.
DYNASTY_YEAR = 2019

# ---------------------------------------------------------------------------
# CACHE FILES (persisted intermediate computations)
# ---------------------------------------------------------------------------

CLEANED_DATA_PATH      = CACHE_DIR / "2022_clean.csv"
INTERVALS_CACHE_PATH   = CACHE_DIR / "2022_h0_intervals.csv"
TOPOLOGY_RESULTS_PATH  = CACHE_DIR / "2022_provincial_topological_features.csv"
CONVERGENCE_CACHE_PATH = CACHE_DIR / "2022_cauchy_convergence.csv"
ROBUSTNESS_CACHE_PATH  = CACHE_DIR / "2022_epsilon_robustness.csv"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CLR_COLS = ['Admin', 'Opposition', 'Independent', 'Non-Aligned']

# OAV pseudo-provinces to exclude (present in 2022 data as Region == 'OAV')
OAV_PSEUDO_PROVINCES = {
    'AMERICAS', 'ASIA PACIFIC', 'EUROPE', 'MIDDLE EAST AND AFRICAS',
}

# Set to 10000 for the final run; test with a smaller value first.
NUM_REPETITIONS = 1000
BM_EPSILON = 0.8

# Set to False to skip stepwise stability (landscape convergence) analysis.
# This can be slow; results are cached after the first run.
RUN_STEPWISE_STABILITY = True

# Colormap used throughout for the admin-share diverging scale
ADMIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "SharpAdminSplit",
    [(0.0, "darkblue"), (0.499, "cyan"), (0.501, "yellow"), (1.0, "red")],
)

SHARP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "SharpDiverging",
    [(0.0, "darkblue"), (0.499, "cyan"), (0.501, "yellow"), (1.0, "red")],
)

SCALED_COLS = [
    'Mean Entropy (Scaled)',
    'Mean L2 Norm (Scaled)',
    'Mean Total Persistence (Scaled)',
]
