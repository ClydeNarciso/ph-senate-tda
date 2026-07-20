"""
Central configuration for the 2019 electoral-topology / socioeconomic
analysis pipeline: input/output paths and shared constants.

Directory layout
-----------------
data/processed/                 -> primary precinct-level feature CSV
data/socioeconomic data/        -> raw socioeconomic source CSVs
results/figures/                -> all plots (PNG/JPG)
results/persistence_cache/      -> cached intermediate computations (CSV)
results/tables/                 -> final summary tables (CSV/LaTeX)
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

DATA_FILE = DATA_DIR / "2019_4features.csv"

POVERTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / "Poverty Incidence among Families 2018 to 2023.csv"
)
ARI_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "ARI Dependency Rate" / "By-LGU-ARI-and-Dependencies-2019.csv"
)
ETHNICITY_DEPENDENCY_FILE = (
    SOCIOECONOMIC_DIR / "Ethnicity" / "provincial_dominant_ethnicity_psa2020.csv"
)
DYNASTY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Dynasty Proxy Variables" / "DynastyProxyVariables.csv"
)

# ---------------------------------------------------------------------------
# CACHE FILES (persisted intermediate computations)
# ---------------------------------------------------------------------------

CLEANED_DATA_PATH      = CACHE_DIR / "2019_clean.csv"
INTERVALS_CACHE_PATH   = CACHE_DIR / "2019_h0_intervals.csv"
TOPOLOGY_RESULTS_PATH  = CACHE_DIR / "2019_provincial_topological_features.csv"
CONVERGENCE_CACHE_PATH = CACHE_DIR / "2019_cauchy_convergence.csv"
ROBUSTNESS_CACHE_PATH  = CACHE_DIR / "2019_epsilon_robustness.csv"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CLR_COLS = ['Admin', 'Opposition', 'Independent', 'Non-Aligned']

# Set to 10000 for the final run; test with a smaller value first.
NUM_REPETITIONS = 1000
BM_EPSILON = 0.6

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
