"""
Central configuration for the 2016 electoral-topology / socioeconomic
analysis pipeline: input/output paths and shared constants.

Key differences from the 2019 / 2022 pipelines
------------------------------------------------
- Input data: 2016_4features.csv
- Region label: ARMM (not BARMM). The Bangsamoro transition happened after
  2019, so the 2016 data still uses the old ARMM label.
- Province labels use pre-split names:
    COMPOSTELA VALLEY  (renamed Davao de Oro in 2019)
    COTABATO (NORTH COT.)  (feature file name for North Cotabato)
    DAVAO (DAVAO DEL NORTE)  (feature file name for Davao del Norte)
    MAGUINDANAO  (not yet split into del Norte / del Sur)
    SAMAR (WESTERN SAMAR)  (renamed simply Samar in later files)
  These must be normalised in socioeconomic.py so they match the names
  used in the socioeconomic source files.
- NCR: same "NATIONAL CAPITAL REGION - X" format as 2022 / 2019, plus
  TAGUIG - PATEROS as a separate province row (same as 2022).
- OAV pseudo-provinces: ASIA, EUROPE, MIDDLE EAST AND AFRICAS,
  NORTH AND LATIN AMERICA (no trailing S — differs from both 2022 and 2025).
- ARI/IRA source: By-LGU-ARI-and-Dependencies-2016.xlsx  (sheet: Data).
  Same header-row index (6) and IRA DEPENDENCY column as 2022.
  Values are decimal fractions (0–1).
  ARI uses WESTERN SAMAR (SAMAR) for Samar province.
- Poverty: Poverty_Incidence_among_Families_2015_to_2018.csv
  Wide-format CSV with 2015 and 2018 values in the same row.
  No 2016 year exists; 2015 is the closest prior survey year and is used
  as the primary column.  2018 is kept as the fallback.
  Davao Occidental has no 2015 estimate (NaN) so 2018 is used for it.
- Dynasty (HHI / GINI): 2016 actual data exists in the dataset.
  This is the only pipeline year where dynasty data is contemporaneous
  with the election year.

Directory layout
-----------------
data/processed/                    -> 2016_4features.csv
data/socioeconomic data/
  ARI Dependency Rate/             -> By-LGU-ARI-and-Dependencies-2016.xlsx
  Ethnicity/                       -> provincial_dominant_ethnicity_psa2020.csv
  Poverty Incidence among Families/-> Poverty_Incidence_among_Families_2015_to_2018.csv
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

DATA_FILE = DATA_DIR / "2016_4features.csv"

POVERTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / "Poverty_Incidence_among_Families_2015_to_2018.csv"
)
# No 2016 poverty data; 2015 is the closest prior survey year.
# 2018 is used as fallback only for Davao Occidental which has no 2015 estimate.
POVERTY_PRIMARY_YEAR  = "2015 Poverty Incidence"
POVERTY_FALLBACK_YEAR = "2018 Poverty Incidence"
POVERTY_COL_LABEL     = "2015 Poverty Incidence"

ARI_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "ARI Dependency Rate"
    / "By-LGU-ARI-and-Dependencies-2016.xlsx"
)
ARI_SHEET_NAME    = "Data"
ARI_HEADER_ROW    = 6          # 0-indexed row that contains column names
ARI_DEP_COL       = "IRA DEPENDENCY"
ARI_VALUE_IS_FRACTION = True   # values are decimal fractions (0–1)

ETHNICITY_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "Ethnicity" / "provincial_dominant_ethnicity_psa2020.csv"
)
DYNASTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Dynasty Proxy Variables" / "DynastyProxyVariables.csv"
)
# 2016 actual dynasty data exists 
DYNASTY_YEAR = 2016

# ---------------------------------------------------------------------------
# OAV PSEUDO-PROVINCE NAMES (2016-specific set)
# ---------------------------------------------------------------------------

OAV_PSEUDO_PROVINCES = {
    'ASIA',                      # 2016 label (not 'ASIA PACIFIC' like later years)
    'EUROPE',
    'MIDDLE EAST AND AFRICAS',
    'NORTH AND LATIN AMERICA',   # no trailing S (differs from 2022/2025)
}

# ---------------------------------------------------------------------------
# CACHE FILES
# ---------------------------------------------------------------------------

CLEANED_DATA_PATH      = CACHE_DIR / "2016_clean.csv"
INTERVALS_CACHE_PATH   = CACHE_DIR / "2016_h0_intervals.csv"
TOPOLOGY_RESULTS_PATH  = CACHE_DIR / "2016_provincial_topological_features.csv"
CONVERGENCE_CACHE_PATH = CACHE_DIR / "2016_cauchy_convergence.csv"
ROBUSTNESS_CACHE_PATH  = CACHE_DIR / "2016_epsilon_robustness.csv"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CLR_COLS = ['Admin', 'Opposition', 'Independent', 'Non-Aligned']

NUM_REPETITIONS = 1000
BM_EPSILON = 0.6

RUN_STEPWISE_STABILITY = True

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
