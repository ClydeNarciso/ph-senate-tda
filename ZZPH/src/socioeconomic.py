"""
socioeconomic.py — Longitudinal socioeconomic data loaders for the ZZPH pipeline.

All four socioeconomic dimensions are *averaged* across the four election
cycles rather than using a single year, because the zigzag topology already
represents longitudinal change.  Each loader therefore returns a single
province-level scalar that is the cross-year mean.

Poverty
    Two source files (2015–2018 and 2018–2023) are merged by province.
    Four survey years (2015, 2018, 2021, 2023) are averaged into a single
    ``Mean_Poverty_Incidence`` column.

Functional Literacy (FLEMMS 2024)
    Province-level functional literacy rates extracted from Table 3 of the
    2024 FLEMMS press release.  HUC cities are collapsed to their parent
    province (simple average) before the join.

IRA / NTA Dependency
    Loaded from one XLSX per election year (2016 → IRA, 2025 → NTA).
    NCR LGU-level rows are redistributed to district labels.
    Values are converted to percentages if stored as fractions.
    Cross-year mean → ``Mean_IRA_Dependency``.

Admin Vote Share
    Computed directly from the cleaned precinct DataFrames; not an
    external source.  See ``append_admin_vote_share``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ARI_DEPENDENCY_DIR,
    LITERACY_DATA_FILE,
    POVERTY_DATA_FILE_1,
    POVERTY_DATA_FILE_2,
)


# ---------------------------------------------------------------------------
# Province-name standardisation (socioeconomic sources → canonical)
# ---------------------------------------------------------------------------

def standardize_province_names(series: pd.Series) -> pd.Series:
    """Normalise province names in socioeconomic source files to the same
    canonical set used by the election feature DataFrames.

    Handles:
    - Footnote suffixes from PSA CSV files (``a/``, ``r1, c/``, etc.)
    - FLEMMS HUC parentheticals (``Excluding City of Baguio``)
    - Old ARI XLSX province spellings and abbreviations
    - NCR HUC city names → district labels
    """
    clean = series.astype(str).str.upper()
    # Strip FLEMMS "(Excluding …)" / "(Including …)" qualifiers
    clean = clean.str.replace(r"\s*\(EXCLUDING[^)]*\)", "", regex=True)
    clean = clean.str.replace(r"\s*\(INCLUDING[^)]*\)", "", regex=True)
    # Strip PSA footnote suffixes: " a/", " r1, c/", " b/, c/" …
    clean = clean.str.replace(
        r"\s+[A-Z0-9]{1,2}[/,].*", "", regex=True, flags=re.IGNORECASE
    )
    clean = clean.str.strip()

    mapping: dict[str, str] = {
        # Old province names → canonical
        "COTABATO (NORTH COT.)":          "NORTH COTABATO",
        "COTABATO":                       "NORTH COTABATO",
        "SAMAR (WESTERN SAMAR)":          "WESTERN SAMAR",
        "SAMAR":                          "WESTERN SAMAR",
        "DAVAO (DAVAO DEL NORTE)":        "DAVAO DEL NORTE",
        "COMPOSTELA VALLEY":              "DAVAO DE ORO",
        # Maguindanao split
        "MAGUINDANAO DEL NORTE":          "MAGUINDANAO",
        "MAGUINDANAO DEL SUR":            "MAGUINDANAO",
        # Misc
        "MT. PROVINCE":                   "MOUNTAIN PROVINCE",
        "DINAGAT ISLAND":                 "DINAGAT ISLANDS",
        "SARANGGANI":                     "SARANGANI",
        # NCR district labels from poverty CSV
        "1ST DISTRICT":  "NATIONAL CAPITAL REGION - MANILA",
        "2ND DISTRICT":  "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "3RD DISTRICT":  "NATIONAL CAPITAL REGION - THIRD DISTRICT",
        "4TH DISTRICT":  "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        # NCR HUC cities → districts
        "CITY OF MANILA":      "NATIONAL CAPITAL REGION - MANILA",
        "CITY OF MANDALUYONG": "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "CITY OF MARIKINA":    "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "CITY OF PASIG":       "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "QUEZON CITY":         "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "CITY OF SAN JUAN":    "NATIONAL CAPITAL REGION - SECOND DISTRICT",
        "CITY OF CALOOCAN":    "NATIONAL CAPITAL REGION - THIRD DISTRICT",
        "CITY OF MALABON":     "NATIONAL CAPITAL REGION - THIRD DISTRICT",
        "CITY OF NAVOTAS":     "NATIONAL CAPITAL REGION - THIRD DISTRICT",
        "CITY OF VALENZUELA":  "NATIONAL CAPITAL REGION - THIRD DISTRICT",
        "CITY OF LAS PIÑAS":   "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "CITY OF MAKATI":      "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "CITY OF MUNTINLUPA":  "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "CITY OF PARAÑAQUE":   "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "PASAY CITY":          "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "PATEROS":             "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        "CITY OF TAGUIG":      "NATIONAL CAPITAL REGION - FOURTH DISTRICT",
        # Province-level HUC cities
        "CITY OF BAGUIO":        "BENGUET",
        "CITY OF ANGELES":       "PAMPANGA",
        "CITY OF OLONGAPO":      "ZAMBALES",
        "CITY OF LUCENA":        "QUEZON",
        "CITY OF PUERTO PRINCESA": "PALAWAN",
        "CITY OF ILOILO":        "ILOILO",
        "CITY OF BACOLOD":       "NEGROS OCCIDENTAL",
        "CITY OF CEBU":          "CEBU",
        "CITY OF LAPU-LAPU":     "CEBU",
        "CITY OF MANDAUE":       "CEBU",
        "CITY OF TACLOBAN":      "LEYTE",
        "CITY OF ISABELA":       "BASILAN",
        "CITY OF ZAMBOANGA":     "ZAMBOANGA DEL SUR",
        "CITY OF ILIGAN":        "LANAO DEL NORTE",
        "CITY OF CAGAYAN DE ORO": "MISAMIS ORIENTAL",
        "CITY OF DAVAO":         "DAVAO DEL SUR",
        "CITY OF GENERAL SANTOS": "SOUTH COTABATO",
        "CITY OF BUTUAN":        "AGUSAN DEL NORTE",
        "COTABATO CITY":         "MAGUINDANAO",
    }
    return clean.replace(mapping)


# ---------------------------------------------------------------------------
# A. Poverty Incidence (2015 + 2018–2023 cross-file average)
# ---------------------------------------------------------------------------

def load_poverty_data(
    file_1: Path = POVERTY_DATA_FILE_1,
    file_2: Path = POVERTY_DATA_FILE_2,
) -> pd.DataFrame:
    """Return a province-level ``Mean_Poverty_Incidence`` (%) column.

    Source 1 (2015–2018 CSV): wide format; column 0 = Province,
    column 3 = 2015 value.  Header at row 2, data from row 7.

    Source 2 (2018–2023 CSV): semicolon-delimited; columns are
    Province / 2018 / 2021 / 2023.  Data starts at row 3.

    The four year-columns are averaged row-wise to produce a single
    longitudinal poverty proxy.
    """
    dfs = []

    if file_1.exists():
        df_p1 = pd.read_csv(
            file_1, sep=",", header=2, encoding="latin1"
        ).replace("..", np.nan)
        df_p1.rename(
            columns={df_p1.columns[0]: "Region/Province", df_p1.columns[3]: "POV_2015"},
            inplace=True,
        )
        df_p1.dropna(subset=["Region/Province"], inplace=True)
        df_p1["Province"] = standardize_province_names(df_p1["Region/Province"])
        df_p1["POV_2015"] = pd.to_numeric(df_p1["POV_2015"], errors="coerce")
        dfs.append(
            df_p1.dropna(subset=["POV_2015"])
            .groupby("Province", as_index=False)["POV_2015"]
            .mean()
        )

    if file_2.exists():
        df_p2 = pd.read_csv(
            file_2, sep=";", skiprows=2, encoding="latin1"
        ).replace("..", np.nan)
        df_p2.columns = [str(c).replace('"', "").strip() for c in df_p2.columns]

        if "Province" in df_p2.columns and df_p2.shape[1] >= 4:
            df_p2.dropna(subset=["Province"], inplace=True)
            df_p2["Province"] = standardize_province_names(df_p2["Province"])
            df_p2["POV_2018"] = pd.to_numeric(df_p2.iloc[:, 1], errors="coerce")
            df_p2["POV_2021"] = pd.to_numeric(df_p2.iloc[:, 2], errors="coerce")
            df_p2["POV_2023"] = pd.to_numeric(df_p2.iloc[:, 3], errors="coerce")
            dfs.append(
                df_p2.groupby("Province", as_index=False)[
                    ["POV_2018", "POV_2021", "POV_2023"]
                ].mean()
            )

    if not dfs:
        print("  [!] WARNING: Poverty data files not found. Returning empty.")
        return pd.DataFrame(columns=["Province", "Mean_Poverty_Incidence"])

    # Outer-join the two sources and compute cross-year mean
    df_merged = dfs[0] if len(dfs) == 1 else dfs[0].merge(dfs[1], on="Province", how="outer")
    pov_cols = [c for c in df_merged.columns if c.startswith("POV_")]
    df_merged["Mean_Poverty_Incidence"] = df_merged[pov_cols].mean(axis=1)

    return (
        df_merged[["Province", "Mean_Poverty_Incidence"]]
        .dropna()
        .drop_duplicates(subset="Province", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# B. Functional Literacy (FLEMMS 2024)
# ---------------------------------------------------------------------------

def load_literacy_data(file: Path = LITERACY_DATA_FILE) -> pd.DataFrame:
    """Return a province-level ``Basic_Literacy_Rate`` (%) column.

    Reads Table 3 of the 2024 FLEMMS Excel file.  Searches for the row
    containing 'Philippines' to find the data start; uses the third-to-last
    numeric column as the functional literacy rate (both sexes).
    """
    if not file.exists():
        print("  [!] WARNING: FLEMMS literacy file not found. Returning empty.")
        return pd.DataFrame(columns=["Province", "Basic_Literacy_Rate"])

    df_raw = pd.read_excel(file, header=None).dropna(axis=1, how="all")
    df_raw.columns = range(df_raw.shape[1])

    # Locate the text column that contains "Philippines"
    text_col = None
    for col in df_raw.columns:
        if df_raw[col].astype(str).str.contains("Philippines", case=False, na=False).any():
            text_col = col
            break
    if text_col is None:
        print("  [!] WARNING: Could not locate data anchor in FLEMMS file.")
        return pd.DataFrame(columns=["Province", "Basic_Literacy_Rate"])

    # Trim to data rows
    start_row = df_raw[
        df_raw[text_col].astype(str).str.contains("Philippines", case=False, na=False)
    ].index[0]
    df_raw = df_raw.iloc[start_row:].copy()
    df_raw["Province"] = standardize_province_names(df_raw[text_col])

    # Identify the functional literacy column (3rd-to-last numeric column)
    ph_row = df_raw.iloc[0]
    numeric_cols = [
        c for c in df_raw.columns
        if c != text_col and pd.notna(pd.to_numeric(ph_row[c], errors="coerce"))
    ]
    rate_col = numeric_cols[-3] if len(numeric_cols) >= 3 else (text_col + 7)

    df_raw["Basic_Literacy_Rate"] = pd.to_numeric(df_raw[rate_col], errors="coerce")

    return (
        df_raw.dropna(subset=["Province", "Basic_Literacy_Rate"])
        .groupby("Province", as_index=False)["Basic_Literacy_Rate"]
        .mean()
        .drop_duplicates(subset="Province", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# C. IRA / NTA Dependency (longitudinal 2016–2025 average)
# ---------------------------------------------------------------------------

def _reformat_ncr_lgu(val: str) -> str:
    """Reformat e.g. 'CALOOCAN CITY' → 'CITY OF CALOOCAN' for mapping."""
    v = str(val).upper().strip()
    if "CITY" in v and not v.startswith("CITY OF"):
        name = v.replace("CITY", "").strip()
        return f"CITY OF {name}"
    return v


def load_ira_data(ari_dir: Path = ARI_DEPENDENCY_DIR) -> pd.DataFrame:
    """Return a province-level ``Mean_IRA_Dependency`` (%) and ``Region`` column.

    Reads one XLSX per election year (2016, 2019, 2022, 2025), extracts the
    IRA DEPENDENCY / NTA DEPENDENCY column, converts fractions to percentages,
    redistributes NCR LGU-level rows to district labels, then averages all
    four years into a single longitudinal mean per province.
    """
    years = [2016, 2019, 2022, 2025]
    yearly_frames = []

    for yr in years:
        fp = ari_dir / f"By-LGU-ARI-and-Dependencies-{yr}.xlsx"
        if not fp.exists():
            continue

        df_yr = pd.read_excel(fp, skiprows=6)
        df_yr.columns = [str(c).upper().strip() for c in df_yr.columns]

        required = {"REGION", "PROVINCE", "LGU TYPE"}
        if not required.issubset(df_yr.columns):
            continue

        # Find the dependency column (IRA or NTA)
        dep_col = next(
            (c for c in df_yr.columns
             if any(tok in c for tok in ["IRA DEPENDENCY", "NTA DEPENDENCY",
                                          "IRA DEPENDENCE", "NTA DEPENDENCE"])),
            None,
        )
        if dep_col is None:
            continue

        df_yr = df_yr.dropna(subset=["PROVINCE", "LGU TYPE", dep_col])
        df_yr["LGU_TYPE_C"]  = df_yr["LGU TYPE"].astype(str).str.upper().str.strip()
        df_yr["REGION_C"]    = df_yr["REGION"].astype(str).str.upper().str.strip()
        is_ncr = df_yr["REGION_C"].str.contains("NCR", na=False)

        df_filtered = df_yr[
            (df_yr["LGU_TYPE_C"] == "PROVINCE") | is_ncr
        ].copy()

        # NCR: use LGU NAME; others: use PROVINCE
        df_filtered["Prov_Raw"] = np.where(
            is_ncr[df_filtered.index],
            df_filtered["LGU NAME"].astype(str),
            df_filtered["PROVINCE"].astype(str),
        )
        df_filtered["Prov_Raw"] = df_filtered["Prov_Raw"].apply(_reformat_ncr_lgu)
        df_filtered["Province"] = standardize_province_names(df_filtered["Prov_Raw"])
        df_filtered["Dep_Value"] = pd.to_numeric(df_filtered[dep_col], errors="coerce")

        # Convert fractions (0–1) to percentages
        max_val = df_filtered["Dep_Value"].max()
        if pd.notna(max_val) and max_val <= 1.0:
            df_filtered["Dep_Value"] *= 100

        agg = df_filtered.groupby("Province", as_index=False).agg(
            Dep_Value=("Dep_Value", "mean"),
            Region=("REGION_C", "last"),
        )
        yearly_frames.append(agg)

    if not yearly_frames:
        print("  [!] WARNING: No ARI XLSX files found. Returning empty.")
        return pd.DataFrame(columns=["Province", "Mean_IRA_Dependency", "Region"])

    df_combined = pd.concat(yearly_frames, ignore_index=True)
    df_ira = df_combined.groupby("Province", as_index=False).agg(
        Mean_IRA_Dependency=("Dep_Value", "mean"),
        Region=("Region", "last"),
    )
    return (
        df_ira.drop_duplicates(subset="Province", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# D. Admin vote share (computed from cleaned election DataFrames)
# ---------------------------------------------------------------------------

def append_admin_vote_share(
    df_master: pd.DataFrame,
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
) -> pd.DataFrame:
    """Compute population-weighted provincial admin vote share per year,
    then average across the four cycles.

    Population weighting: sum raw votes province-wide, then compute the
    admin fraction from those totals (not an average of precinct fractions).

    Returns
    -------
    pd.DataFrame
        ``df_master`` with an additional ``Mean_Admin_Vote`` column (%).
    """
    vote_cols = [
        "Admin_votes", "Opposition_votes",
        "Independent_votes", "Non-Aligned_votes",
    ]

    def _province_share(df: pd.DataFrame, label: str) -> pd.DataFrame:
        grp = df.groupby("Province", as_index=False)[vote_cols].sum()
        grp["Total_Votes"] = grp[vote_cols].sum(axis=1)
        grp[f"Admin_{label}"] = grp["Admin_votes"] / grp["Total_Votes"]
        print(
            f"  • {label} — provincial admin mean: "
            f"{grp[f'Admin_{label}'].mean():.4f}"
        )
        return grp[["Province", f"Admin_{label}"]]

    print("\n--- CALCULATING MEAN ADMIN VOTE SHARE ---")
    a16 = _province_share(df_16, "16")
    a19 = _province_share(df_19, "19")
    a22 = _province_share(df_22, "22")
    a25 = _province_share(df_25, "25")

    df_admin = (
        a16.merge(a19, on="Province", how="outer")
           .merge(a22, on="Province", how="outer")
           .merge(a25, on="Province", how="outer")
    )
    cycle_cols = ["Admin_16", "Admin_19", "Admin_22", "Admin_25"]
    df_admin["Mean_Admin_Vote"] = df_admin[cycle_cols].mean(axis=1) * 100

    print(
        f"  • Global mean across cycles: {df_admin['Mean_Admin_Vote'].mean():.2f}%  "
        f"(min {df_admin['Mean_Admin_Vote'].min():.2f}%  "
        f"max {df_admin['Mean_Admin_Vote'].max():.2f}%)"
    )

    df_out = (
        df_master.merge(df_admin[["Province", "Mean_Admin_Vote"]], on="Province", how="left")
                 .dropna(subset=["Mean_Admin_Vote"])
                 .reset_index(drop=True)
    )
    print(f"-> Mapped Mean_Admin_Vote to {len(df_out)} provinces.")
    return df_out


# ---------------------------------------------------------------------------
# E. Diagnostic: unmatched province check
# ---------------------------------------------------------------------------

def check_unmatched_provinces(
    df_topo: pd.DataFrame,
    df_pov: pd.DataFrame,
    df_lit: pd.DataFrame,
    df_ira: pd.DataFrame,
) -> None:
    """Print which topology provinces are absent from each socioeconomic table."""
    topo_p = set(df_topo["Province"].unique())
    pov_p  = set(df_pov["Province"].unique()) if not df_pov.empty else set()
    lit_p  = set(df_lit["Province"].unique()) if not df_lit.empty else set()
    ira_p  = set(df_ira["Province"].unique()) if not df_ira.empty else set()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: UNMATCHED PROVINCES IN SOCIOECONOMIC DATA")
    print("=" * 60)

    for label, ref in [("POVERTY", pov_p), ("LITERACY", lit_p), ("IRA/NTA", ira_p)]:
        missing = sorted(topo_p - ref)
        if missing:
            print(f"\n[{label}] Missing {len(missing)} province(s):")
            for p in missing:
                print(f"  - {p}")
        else:
            print(f"\n[{label}] All election provinces matched. \u2713")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# F. Master merge
# ---------------------------------------------------------------------------

def load_and_merge_socioeconomic(df_topo: pd.DataFrame) -> pd.DataFrame:
    """Join poverty, literacy, and IRA/NTA data onto the zigzag feature table.

    Only provinces with non-null values in **all three** socioeconomic
    dimensions are retained (strict inner filtering), matching the
    notebook's ``dropna`` step.

    Returns
    -------
    pd.DataFrame
        Province-level feature table with socioeconomic columns appended.
    """
    df = df_topo.copy()
    df["Province"] = df["Province"].str.strip().str.upper()

    print("-> Loading Poverty Data…")
    df_pov = load_poverty_data()
    print("-> Loading Literacy Data…")
    df_lit = load_literacy_data()
    print("-> Loading IRA/NTA Dependency Data…")
    df_ira = load_ira_data()

    check_unmatched_provinces(df, df_pov, df_lit, df_ira)

    if not df_pov.empty:
        df_pov["Province"] = df_pov["Province"].str.strip().str.upper()
        df = df.merge(
            df_pov[["Province", "Mean_Poverty_Incidence"]].drop_duplicates("Province"),
            on="Province", how="left",
        )
    if not df_lit.empty:
        df_lit["Province"] = df_lit["Province"].str.strip().str.upper()
        df = df.merge(
            df_lit[["Province", "Basic_Literacy_Rate"]].drop_duplicates("Province"),
            on="Province", how="left",
        )
    if not df_ira.empty:
        df_ira["Province"] = df_ira["Province"].str.strip().str.upper()
        merge_cols = ["Province", "Mean_IRA_Dependency"]
        if "Region" not in df.columns:
            merge_cols.append("Region")
        df = df.merge(
            df_ira[merge_cols].drop_duplicates("Province"),
            on="Province", how="left",
        )

    n_before = len(df)
    df = df.dropna(
        subset=["Mean_Poverty_Incidence", "Basic_Literacy_Rate", "Mean_IRA_Dependency"]
    ).reset_index(drop=True)
    n_after = len(df)

    if n_after < n_before:
        print(
            f"-> Strict filtering: dropped {n_before - n_after} province(s) "
            f"with incomplete socioeconomic data."
        )

    return df
