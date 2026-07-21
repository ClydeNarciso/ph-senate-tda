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

DATA_FILE = DATA_DIR / "2025_4features.csv"

POVERTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / "PSA Full Year Poverty Statistics - 2018,2021,2023.csv"
)
# No 2025 poverty data; 2023 is the closest available year.
# 2021 is used as fallback only for Maguindanao del Norte / del Sur
# which have no separate 2023 entry (only a combined Maguindanao 2023 figure).
POVERTY_PRIMARY_YEAR   = "2023 Poverty Incidence"
POVERTY_FALLBACK_YEAR  = "2021 Poverty Incidence"
POVERTY_COL_LABEL      = "2023 Poverty Incidence"   # label used throughout

ARI_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "ARI Dependency Rate"
    / "By-LGU-ARI-and-Dependencies-2025.xlsx"
)
ARI_SHEET_NAME    = "ARI 2025"
ARI_HEADER_ROW    = 6          # 0-indexed row that contains column names
# Column renamed from 'IRA DEPENDENCY' (2022) to 'NTA DEPENDENCY' (2025).
ARI_DEP_COL       = "NTA DEPENDENCY"
ARI_VALUE_IS_FRACTION = True   # values are already decimal fractions (0-1)

ETHNICITY_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "Ethnicity" / "provincial_dominant_ethnicity_psa2020.csv"
)
DYNASTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Dynasty Proxy Variables" / "DynastyProxyVariables.csv"
)
DYNASTY_YEAR = 2019            # last available year; used as 2025 proxy

FLEMMS_FILE = (
    SOCIOECONOMIC_DIR / "Literacy Rate"
    / "2024 FLEMMS Statistical Tables Press Release 2.xlsx"
)
FLEMMS_SHEET      = "Table 3"  # Functional literacy rate by province
FLEMMS_COL_LABEL  = "Functional_Literacy_Rate"   # column name after loading

# ---------------------------------------------------------------------------
# PSEUDO-PROVINCES / SINGLE-ROW NON-PROVINCES TO EXCLUDE
# ---------------------------------------------------------------------------

# OAV overseas pseudo-provinces (2025 names differ slightly from 2022)
OAV_PSEUDO_PROVINCES = {
    'ASIA PACIFIC', 'EUROPE', 'MIDDLE EAST AND AFRICAS',
    'NORTH AND LATIN AMERICAS',         # replaces 'AMERICAS' from 2022
}

# Non-geographic province labels that appear in the 2025 feature file
NON_PROVINCE_LABELS = {
    'LAV',                    # single-precinct artefact row, Region LAV
    'SPECIAL GEOGRAPHIC AREA',  # BARMM special zone; too small for TDA
}

# ---------------------------------------------------------------------------
# CACHE FILES
# ---------------------------------------------------------------------------

CLEANED_DATA_PATH      = CACHE_DIR / "2025_clean.csv"
INTERVALS_CACHE_PATH   = CACHE_DIR / "2025_h0_intervals.csv"
TOPOLOGY_RESULTS_PATH  = CACHE_DIR / "2025_provincial_topological_features.csv"
CONVERGENCE_CACHE_PATH = CACHE_DIR / "2025_cauchy_convergence.csv"
ROBUSTNESS_CACHE_PATH  = CACHE_DIR / "2025_epsilon_robustness.csv"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CLR_COLS = ['Admin', 'Opposition', 'Independent', 'Non-Aligned']

NUM_REPETITIONS = 1000
BM_EPSILON = 0.8

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
