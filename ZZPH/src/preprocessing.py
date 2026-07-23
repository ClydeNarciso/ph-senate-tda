"""
preprocessing.py — Province harmonisation, OAV exclusion, and
Isolation Forest outlier removal for the ZZPH pipeline.

The four election-year CSVs use different province naming conventions.
``process_dataframe`` maps all of them to a single canonical set so that
the four-year intersection used by the zigzag filtration is as large as
possible.  Key decisions:

- Maguindanao del Norte / del Sur (2022 split) → reunified as MAGUINDANAO
  for time-series continuity.
- COMPOSTELA VALLEY → DAVAO DE ORO (renamed in 2019).
- SPECIAL GEOGRAPHIC AREA (BARMM) → absorbed into NORTH COTABATO.
- TAGUIG - PATEROS → NATIONAL CAPITAL REGION - FOURTH DISTRICT.
- NCR short labels (NCR - MANILA etc.) → full labels.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from config import (
    FEATURE_COLS,
    OUTLIER_CONTAMINATION,
    OUTLIER_MIN_POINTS,
    PREP_DIR,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Province name standardisation map (canonical = post-2022 official names)
# ---------------------------------------------------------------------------

_PROVINCE_MAP: dict[str, str] = {
    # Renames
    "COMPOSTELA VALLEY":                          "DAVAO DE ORO",
    "COTABATO":                                   "NORTH COTABATO",
    "COTABATO (NORTH COT.)":                      "NORTH COTABATO",
    "DAVAO (DAVAO DEL NORTE)":                    "DAVAO DEL NORTE",
    "SAMAR":                                      "WESTERN SAMAR",
    "SAMAR (WESTERN SAMAR)":                      "WESTERN SAMAR",
    # 2022 Maguindanao split — reunify for longitudinal continuity
    "MAGUINDANAO DEL NORTE":                      "MAGUINDANAO",
    "MAGUINDANAO DEL SUR":                        "MAGUINDANAO",
    # BARMM special zone (63 barangays from North Cotabato)
    "SPECIAL GEOGRAPHIC AREA":                    "NORTH COTABATO",
    # NCR short-form labels (present in 2022 data)
    "NCR - MANILA":                               "NATIONAL CAPITAL REGION - MANILA",
    "NCR - SECOND DISTRICT":                      "NATIONAL CAPITAL REGION - SECOND DISTRICT",
    "NCR - THIRD DISTRICT":                       "NATIONAL CAPITAL REGION - THIRD DISTRICT",
    "NCR - FOURTH DISTRICT":                      "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
    # Taguig-Pateros (separate province row in 2016/2022 data)
    "TAGUIG - PATEROS":                           "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
}

_OAV_PATTERN = (
    r"OAV|OVERSEAS|LAV|EUROPE|MIDDLE EAST|AFRICA|AMERICA|ASIA PACIFIC"
)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise province names and drop OAV / non-provincial rows.

    Parameters
    ----------
    df:
        Raw 4-features DataFrame loaded from one of the year CSVs.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a canonical ``Province`` column and
        numeric feature columns validated.
    """
    df_clean = df.copy()

    # Drop OAV rows by Region and by Province name
    if "Region" in df_clean.columns:
        df_clean = df_clean[
            ~df_clean["Region"]
            .astype(str)
            .str.contains(_OAV_PATTERN, case=False, na=False)
        ]

    df_clean["Province"] = (
        df_clean["Province"].astype(str).str.strip().str.upper()
    )
    df_clean["Province"] = df_clean["Province"].replace(_PROVINCE_MAP)

    # Secondary pass: catch any province name that still matches OAV patterns
    df_clean = df_clean[
        ~df_clean["Province"]
        .str.contains(_OAV_PATTERN, case=False, na=False)
    ]
    df_clean = df_clean.dropna(subset=["Province"])

    # Coerce feature columns to numeric
    for col in FEATURE_COLS:
        if col in df_clean.columns:
            if df_clean[col].dtype == object:
                df_clean[col] = (
                    df_clean[col].astype(str).str.replace(",", "", regex=False)
                )
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    return df_clean.reset_index(drop=True)


def remove_outliers_provincial(
    df_list: list[pd.DataFrame],
    feature_columns: list[str],
    contamination: float = OUTLIER_CONTAMINATION,
    min_points: int = OUTLIER_MIN_POINTS,
    output_dir: Path | None = PREP_DIR,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Apply per-province Isolation Forest across all four election DataFrames.

    Outlier removal is performed jointly across all years for the same
    province so that the total-points figures in the report reflect the
    full four-cycle footprint.

    Parameters
    ----------
    df_list:
        List of four cleaned DataFrames [df_16, df_19, df_22, df_25].
    feature_columns:
        Column names to pass to the IsolationForest.
    contamination:
        Expected outlier fraction (default 0.01).
    min_points:
        Minimum precinct count required to activate IsolationForest.
        Provinces below this threshold are kept in full.
    output_dir:
        Directory to write the CSV and LaTeX outlier report.

    Returns
    -------
    cleaned_dfs : list[pd.DataFrame]
        One cleaned DataFrame per input year.
    stats_df : pd.DataFrame
        Province-level outlier removal statistics.
    """
    cleaned_dfs: list[pd.DataFrame] = []
    stats_dict: dict[str, dict] = {}

    for df in tqdm(df_list, desc="Filtering Outliers"):
        df_filtered = pd.DataFrame()

        for prov in df["Province"].unique():
            stats_dict.setdefault(
                prov, {"Original Points": 0, "Removed Points": 0, "Retained Points": 0}
            )
            subset = df[df["Province"] == prov].copy()
            n_orig = len(subset)
            stats_dict[prov]["Original Points"] += n_orig

            X = subset[feature_columns].values
            can_apply = (
                n_orig > min_points
                and np.var(X, axis=0).sum() > 1e-10
            )

            if can_apply:
                preds = IsolationForest(
                    contamination=contamination, random_state=42, n_jobs=-1
                ).fit_predict(X)
                subset_clean = subset[preds == 1].copy()
                n_ret = len(subset_clean)
                stats_dict[prov]["Retained Points"] += n_ret
                stats_dict[prov]["Removed Points"] += n_orig - n_ret
                df_filtered = pd.concat(
                    [df_filtered, subset_clean], ignore_index=True
                )
            else:
                stats_dict[prov]["Retained Points"] += n_orig
                df_filtered = pd.concat(
                    [df_filtered, subset], ignore_index=True
                )

        cleaned_dfs.append(df_filtered)

    stats_df = (
        pd.DataFrame.from_dict(stats_dict, orient="index")
        .rename_axis("Province")
        .reset_index()
    )
    stats_df["% Removed"] = (
        stats_df["Removed Points"]
        / stats_df["Original Points"].replace(0, 1)
        * 100
    )
    stats_df = stats_df.sort_values("% Removed", ascending=False).reset_index(
        drop=True
    )

    if output_dir is not None:
        out = Path(output_dir)
        stats_df.to_csv(out / "outlier_removal_report.csv", index=False)

        styled = stats_df.style.format({"% Removed": "{:.2f}\\%"})
        if hasattr(styled, "hide"):
            styled = styled.hide(axis="index")
        with open(out / "outlier_removal_report.tex", "w") as fh:
            fh.write(styled.to_latex(convert_css=True, hrules=True))

        print(f"\n-> Exported Outlier Report to {out / 'outlier_removal_report.csv'}")
        print("\n--- OUTLIER REMOVAL REPORT (TOP 5 BY % REMOVED) ---")
        try:
            from IPython.display import display
            display(
                stats_df.head(5)
                .style.format({"% Removed": "{:.2f}%"})
                .background_gradient(cmap="Reds", subset=["% Removed"])
            )
        except Exception:
            print(stats_df.head(5).to_string())

    return cleaned_dfs, stats_df


def log_province_coverage(
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
    df_master: pd.DataFrame,
) -> None:
    """Print a two-stage province-drop audit to stdout.

    Stage 1: provinces lost by the four-year intersection requirement.
    Stage 2: provinces lost by the strict socioeconomic merge.
    """
    set_16 = set(df_16["Province"])
    set_19 = set(df_19["Province"])
    set_22 = set(df_22["Province"])
    set_25 = set(df_25["Province"])

    all_provs    = set_16 | set_19 | set_22 | set_25
    common_provs = set_16 & set_19 & set_22 & set_25
    dropped_elec = all_provs - common_provs
    final_provs  = set(df_master["Province"])
    dropped_socio = common_provs - final_provs

    print("\n## --- STAGE 1: ELECTION DATA DROPS --- ##")
    print(f"Total unique provinces (union):          {len(all_provs)}")
    print(f"Provinces surviving 4-cycle intersection:{len(common_provs)}")
    print(f"Provinces dropped by election filter:    {len(dropped_elec)}")

    if dropped_elec:
        print("\nDropped provinces and the years they went missing:")
        for prov in sorted(dropped_elec):
            missing = []
            if prov not in set_16: missing.append("2016")
            if prov not in set_19: missing.append("2019")
            if prov not in set_22: missing.append("2022")
            if prov not in set_25: missing.append("2025")
            print(f"  - {prov}: missing from {', '.join(missing)}")

    print(f"\n## --- STAGE 2: SOCIOECONOMIC MERGE DROPS --- ##")
    print(f"Provinces in final df_master:            {len(final_provs)}")
    print(f"Dropped due to missing socioeconomic data: {len(dropped_socio)}")

    if dropped_socio:
        print("\nDropped during strict socioeconomic matching:")
        for prov in sorted(dropped_socio):
            print(f"  - {prov}")
