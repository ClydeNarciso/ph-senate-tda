"""
Central configuration for the Zigzag Persistent Homology (ZZPH)
longitudinal analysis pipeline.

This pipeline pools precinct-level CLR features across the 2016, 2019, 2022,
and 2025 Philippine senatorial elections and computes province-level zigzag
persistence to capture topological change across four election cycles.

Directory layout (relative to the ZZPH module root, i.e. ph-senate-tda/ZZPH)
------------------------------------------------------------------------------
data/processed/               <- raw 4-features CSVs for each year
data/socioeconomic data/
  ARI Dependency Rate/        <- By-LGU-ARI-and-Dependencies-{year}.xlsx
  Literacy Rate/              <- 2024 FLEMMS Statistical Tables Press Release 2.xlsx
  Poverty Incidence among Families/
    Poverty Incidence among Families 2015 to 2018.csv
    Poverty Incidence among Families 2018 to 2023.csv
results/
  1_preprocessed_data/        <- cleaned per-year CSVs + outlier report
  2_knn_radius_optimization/  <- per-province optimal radius JSON cache
  3_zigzag_results/           <- provincial_zigzag_features.csv
  4_ballmapper_clusters/      <- BallMapper graphs, scatter plots, tables
    robustness_caches/        <- comparative robustness CSV cache files
"""
from pathlib import Path
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data" / "processed"
SOCIOECONOMIC_DIR = BASE_DIR / "data" / "socioeconomic data"

RESULTS_DIR = BASE_DIR / "results"
PREP_DIR          = RESULTS_DIR / "1_preprocessed_data"
RADIUS_DIR        = RESULTS_DIR / "2_knn_radius_optimization"
ZIGZAG_DIR        = RESULTS_DIR / "3_zigzag_results"
BALLMAPPER_DIR    = RESULTS_DIR / "4_ballmapper_clusters"
ROBUSTNESS_DIR    = BALLMAPPER_DIR / "robustness_caches"

for _d in (PREP_DIR, RADIUS_DIR, ZIGZAG_DIR, BALLMAPPER_DIR, ROBUSTNESS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# INPUT FILES — raw election data
# ---------------------------------------------------------------------------

FILE_16_RAW = DATA_DIR / "2016_4features.csv"
FILE_19_RAW = DATA_DIR / "2019_4features.csv"
FILE_22_RAW = DATA_DIR / "2022_4features.csv"
FILE_25_RAW = DATA_DIR / "2025_4features.csv"

# ---------------------------------------------------------------------------
# INPUT FILES — socioeconomic sources
# ---------------------------------------------------------------------------

# Poverty: two files cover different survey periods
POVERTY_DATA_FILE_1 = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / "Poverty Incidence among Families 2015 to 2018.csv"
)
POVERTY_DATA_FILE_2 = (
    SOCIOECONOMIC_DIR / "Poverty Incidence among Families"
    / "Poverty Incidence among Families 2018 to 2023.csv"
)

# Literacy: 2024 FLEMMS survey
LITERACY_DATA_FILE = (
    SOCIOECONOMIC_DIR / "Literacy Rate"
    / "2024 FLEMMS Statistical Tables Press Release 2.xlsx"
)

# IRA / NTA dependency: one XLSX per election year
ARI_DEPENDENCY_DIR = SOCIOECONOMIC_DIR / "ARI Dependency Rate"

# ---------------------------------------------------------------------------
# CACHE FILES
# ---------------------------------------------------------------------------

FILE_16_CLEAN       = PREP_DIR / "df_16_prov_clean.csv"
FILE_19_CLEAN       = PREP_DIR / "df_19_prov_clean.csv"
FILE_22_CLEAN       = PREP_DIR / "df_22_prov_clean.csv"
FILE_25_CLEAN       = PREP_DIR / "df_25_prov_clean.csv"
EXPECTED_PREP_FILES = [FILE_16_CLEAN, FILE_19_CLEAN, FILE_22_CLEAN, FILE_25_CLEAN]

RADIUS_CACHE_FILE   = RADIUS_DIR / "optimal_radii_provincial.json"
FINAL_FEATURES_FILE = ZIGZAG_DIR / "provincial_zigzag_features.csv"
ROBUSTNESS_CACHE_FILE = BALLMAPPER_DIR / "epsilon_robustness.csv"

# ---------------------------------------------------------------------------
# ALGORITHM PARAMETERS
# ---------------------------------------------------------------------------

# CLR feature columns present in every 4-features CSV
FEATURE_COLS = ['Admin', 'Opposition', 'Independent', 'Non-Aligned']

# KNN parameters for per-province radius optimisation
KNN_K          = 5
KNN_PERCENTILE = 90
KNN_MIN_RADIUS = 0.10

# Maximum zigzag time-step index (7 steps: X16, X16∪X19, X19, …, X25)
ZIGZAG_MAX_TIME = 7

# Landscape parameters for H0 L2-norm extraction
LANDSCAPE_NUM   = 10
LANDSCAPE_RES   = 700
LANDSCAPE_RANGE = [0.0, 7.0]

# Isolation Forest contamination rate for outlier removal
OUTLIER_CONTAMINATION = 0.01
OUTLIER_MIN_POINTS    = 8      # minimum precinct count to apply IsoForest

# BallMapper
BM_EPSILON      = 0.8
NUM_REPETITIONS = 10000       # set lower (e.g. 100) for quick tests

# ---------------------------------------------------------------------------
# COLORMAPS
# ---------------------------------------------------------------------------

ADMIN_CMAP_MPL = mcolors.LinearSegmentedColormap.from_list(
    "AdminSplit",
    [(0.0, "darkblue"), (0.499, "cyan"), (0.501, "yellow"), (1.0, "red")],
)
ADMIN_CMAP_PLOTLY = [
    [0.0,   "darkblue"],
    [0.499, "cyan"],
    [0.501, "yellow"],
    [1.0,   "red"],
]

# ---------------------------------------------------------------------------
# COLORING CONFIGURATIONS
# ---------------------------------------------------------------------------
# Each entry maps an analysis mode to the column in df_master, display labels,
# colourmap, and axis bounds used by the BallMapper and scatter-plot engines.

COLOR_CONFIGS = {
    "Poverty": {
        "col":          "Mean_Poverty_Incidence",
        "summary_name": "Poverty Incidence (%)",
        "cmap":         "RdYlBu_r",
        "vmin":         None,
        "vmax":         None,
        "label":        "Mean Poverty Incidence (%) [2015–2023 avg]",
        "format":       lambda val, loc: f"{val:.1f}%",
        "robustness": {
            "vulnerable_label": "High Poverty",
            "baseline_label":   "Low Poverty",
            "metric_name":      "Poverty Incidence",
            "invert_colors":    False,
        },
    },
    "Literacy": {
        "col":          "Basic_Literacy_Rate",
        "summary_name": "Literacy Rate (%)",
        "cmap":         "viridis",
        "vmin":         None,
        "vmax":         None,
        "label":        "Basic Literacy Rate (%) [2024 FLEMMS]",
        "format":       lambda val, loc: f"{val:.1f}%",
        "robustness": {
            "vulnerable_label": "Low Literacy",
            "baseline_label":   "High Literacy",
            "metric_name":      "Basic Literacy Rate",
            "invert_colors":    True,
        },
    },
    "IRA_Dependency": {
        "col":          "Mean_IRA_Dependency",
        "summary_name": "IRA Dependency (%)",
        "cmap":         "plasma",
        "vmin":         None,
        "vmax":         None,
        "label":        "Mean IRA/NTA Dependency Rate (%) [2016–2025 avg]",
        "format":       lambda val, loc: f"{val:.1f}%",
        "robustness": {
            "vulnerable_label": "High Dependency",
            "baseline_label":   "Low Dependency",
            "metric_name":      "Mean IRA Dependency",
            "invert_colors":    False,
        },
    },
    "Admin_Vote": {
        "col":          "Mean_Admin_Vote",
        "summary_name": "Admin Vote Share (%)",
        "cmap":         {"mpl": ADMIN_CMAP_MPL, "plotly": ADMIN_CMAP_PLOTLY},
        "vmin":         0,
        "vmax":         100,
        "label":        "Mean Admin Vote Share (%) [2016–2025 avg]",
        "format":       lambda val, loc: f"{val:.1f}%",
        "robustness":   None,   # no comparative robustness run for vote share
    },
}
