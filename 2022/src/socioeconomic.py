"""Socioeconomic data helpers for the 2022 pipeline.

Key differences from the 2019 pipeline
---------------------------------------

POVERTY
  Source: Poverty_Incidence_among_Families_2018_to_2023.csv
  Format: semicolon-delimited; columns are Province / 2018 / 2021 / 2023.
  Strategy: 2021 is the closest available year to the 2022 election and is
  used as the primary column.  Maguindanao del Norte and Maguindanao del
  Sur have no 2021 value (marked "..") so their 2023 figure is used as
  the fallback. Province names carry footnote suffixes ("a/", "b/", "c/",
  "r1") that must be stripped before matching.

IRA DEPENDENCY
  Source: By-LGU-ARI-and-Dependencies-2022.xlsx  (sheet: FY2022)
  Header row: index 6 (7th row in the file).
  Important differences from 2019:
    • NCR cities are all listed under PROVINCE = "Metro Manila"; they must
      be redistributed to the four NCR district labels + TAGUIG-PATEROS
      using the LGU NAME column (same city → district mapping used in 2019).
    • IRA DEPENDENCY values are already decimal fractions (0–1), not
      percent strings.  No string-cleaning step is needed.
    • The column "PROVINCE" holds the name (all-caps, title-case mix).

DYNASTY (HHI / GINI)
  Source: DynastyProxyVariables.csv  (unchanged from 2019)
  No 2022 data exists; the 2019 row (last available year) is used as a
  proxy for the 2022 election cycle.

ETHNICITY
  Source: provincial_dominant_ethnicity_psa2020.csv  (unchanged from 2019)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ETHNICITY_DEPENDENCY_FILE,
    ARI_DEPENDENCY_FILE,
    ARI_SHEET_NAME,
    ARI_HEADER_ROW,
    ARI_VALUE_IS_FRACTION,
    POVERTY_PRIMARY_YEAR,
    POVERTY_FALLBACK_YEAR,
    POVERTY_COL_LABEL,
    DYNASTY_YEAR,
)


# ---------------------------------------------------------------------------
# Province-name standardisation
# ---------------------------------------------------------------------------

def standardize_province_names(series: pd.Series) -> pd.Series:
    """Normalise province name strings to a canonical upper-case form.

    Handles:
    - Footnote suffixes from the poverty CSV ("a/", "b/", "c/", "r1", "b/,")
    - Mixed-case / extra whitespace
    - NCR district names used in the 2022 feature file
    - Province renaming / splits that occurred between the 2019 and 2022 elections
    """
    clean = (
        series.astype(str)
        # Strip PSA footnote suffixes: " a/", " b/", " r1", " a/, b/, c/" etc.
        .str.replace(r'\s+[a-z0-9]+[/,].*', '', regex=True)
        .str.upper()
        .str.strip()
    )
    mapping = {
        # ---- NCR: dynasty CSV uses old comma-based labels ----
        'NCR, CITY OF MANILA, FIRST DISTRICT': 'NCR - MANILA',
        'NCR, SECOND DISTRICT':                'NCR - SECOND DISTRICT',
        'NCR, THIRD DISTRICT':                 'NCR - THIRD DISTRICT',
        'NCR, FOURTH DISTRICT':                'NCR - FOURTH DISTRICT',
        # ---- NCR: poverty CSV uses "1ST DISTRICT" etc. ----
        '1ST DISTRICT': 'NCR - MANILA',
        '2ND DISTRICT': 'NCR - SECOND DISTRICT',
        '3RD DISTRICT': 'NCR - THIRD DISTRICT',
        '4TH DISTRICT': 'NCR - FOURTH DISTRICT',
        # ---- Poverty CSV: Isabela City is a standalone city in BARMM ----
        'ISABELA CITY': 'BASILAN',
        # ---- Mountain Province footnote variant ----
        'MT. PROVINCE': 'MOUNTAIN PROVINCE',
        # ---- Cotabato name variants ----
        'NORTH COTABATO':     'COTABATO',
        'COTABATO (NORTH COT.)': 'COTABATO',
        # ---- Samar ----
        'SAMAR (WESTERN SAMAR)': 'SAMAR',
        # ---- Davao province name changes (present in 2022 data) ----
        'COMPOSTELA VALLEY':   'DAVAO DE ORO',
        'DAVAO (DAVAO DEL NORTE)': 'DAVAO DEL NORTE',
        # ---- Dinagat ----
        'DINAGAT ISLAND': 'DINAGAT ISLANDS',
        # ---- Sarangani ----
        'SARANGGANI': 'SARANGANI',
        # ---- Tawi-Tawi footnote variant ----
        'TAWI-TAWI B/': 'TAWI-TAWI',
        'TAWI-TAWI':    'TAWI-TAWI',
        # ---- Maguindanao split (new in 2022 data) ----
        # The 2022 election had Maguindanao split into del Norte / del Sur;
        # socioeconomic data may still list consolidated "Maguindanao".
        # We keep both forms and let the merge handle them gracefully.
    }
    return clean.replace(mapping)


# ---------------------------------------------------------------------------
# POVERTY (2021 primary / 2023 fallback)
# ---------------------------------------------------------------------------

def process_poverty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    """Parse the 2018-to-2023 poverty CSV and return a province-level table.

    The CSV is semicolon-delimited and its first data row is a header
    embedded after a long title row that we skip with skiprows=1.

    Closest-year strategy
    ---------------------
    Primary  : 2021 Poverty Incidence  (survey year closest before 2022)
    Fallback : 2023 Poverty Incidence  (for the two Maguindanao provinces
               whose 2021 value was not separately published)

    The merged column is labelled POVERTY_COL_LABEL in config (defaults to
    "2021 Poverty Incidence") so downstream code treats it uniformly.
    """
    df_pov = pd.read_csv(csv_path, sep=';', skiprows=1, header=0)
    df_pov.columns = [
        'Province',
        '2018 Poverty Incidence',
        '2021 Poverty Incidence',
        '2023 Poverty Incidence',
    ]
    df_pov = df_pov.replace('..', np.nan)
    for col in ['2018 Poverty Incidence', '2021 Poverty Incidence', '2023 Poverty Incidence']:
        df_pov[col] = pd.to_numeric(df_pov[col], errors='coerce')

    df_pov['Province'] = standardize_province_names(df_pov['Province'])

    # Fill 2021 gaps (Maguindanao del Norte/Sur) using 2023 figures
    mask_missing_primary = df_pov[POVERTY_PRIMARY_YEAR].isna()
    df_pov.loc[mask_missing_primary, POVERTY_PRIMARY_YEAR] = (
        df_pov.loc[mask_missing_primary, POVERTY_FALLBACK_YEAR]
    )
    n_filled = mask_missing_primary.sum()
    if n_filled > 0:
        filled_provs = df_pov.loc[mask_missing_primary, 'Province'].tolist()
        print(f"   [i] Filled {n_filled} missing 2021 poverty values with 2023 data: "
              f"{filled_provs}")

    # Rename primary column to the standard label used downstream
    if POVERTY_PRIMARY_YEAR != POVERTY_COL_LABEL:
        df_pov = df_pov.rename(columns={POVERTY_PRIMARY_YEAR: POVERTY_COL_LABEL})

    # Replicate NCR 4th-district row under TAGUIG-PATEROS label
    ncr_4th = df_pov[df_pov['Province'] == 'NCR - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_pov = pd.concat([df_pov, extra], ignore_index=True)

    pov_cols = [POVERTY_COL_LABEL]
    return df_pov, pov_cols


# ---------------------------------------------------------------------------
# IRA DEPENDENCY (2022 XLSX)
# ---------------------------------------------------------------------------

# Map NCR LGU names to the district-level Province labels used in the
# 2022 feature file.  Keys are upper-cased substrings of LGU NAME.
_NCR_CITY_TO_DISTRICT = {
    'MANILA':       'NCR - MANILA',
    'MANDALUYONG':  'NCR - SECOND DISTRICT',
    'MARIKINA':     'NCR - SECOND DISTRICT',
    'PASIG':        'NCR - SECOND DISTRICT',
    'QUEZON':       'NCR - SECOND DISTRICT',
    'SAN JUAN':     'NCR - SECOND DISTRICT',
    'CALOOCAN':     'NCR - THIRD DISTRICT',
    'MALABON':      'NCR - THIRD DISTRICT',
    'NAVOTAS':      'NCR - THIRD DISTRICT',
    'VALENZUELA':   'NCR - THIRD DISTRICT',
    'LAS PIÑAS':    'NCR - FOURTH DISTRICT',
    'LAS PINAS':    'NCR - FOURTH DISTRICT',
    'MAKATI':       'NCR - FOURTH DISTRICT',
    'MUNTINLUPA':   'NCR - FOURTH DISTRICT',
    'PARAÑAQUE':    'NCR - FOURTH DISTRICT',
    'PARANAQUE':    'NCR - FOURTH DISTRICT',
    'PASAY':        'NCR - FOURTH DISTRICT',
    'TAGUIG':       'TAGUIG - PATEROS',
    'PATEROS':      'TAGUIG - PATEROS',
}


def process_ira_data(xlsx_path: Path) -> tuple[pd.DataFrame, list]:
    """Load IRA dependency from the 2022 XLSX and return province-level means.

    Structural notes (verified against actual file)
    -----------------------------------------------
    - Sheet: FY2022
    - Column headers are at Excel row 7 (0-indexed row 6 in pandas).
    - PROVINCE column: mixed case, no footnotes.
    - LGU NAME column: used to redistribute NCR Metro Manila entries.
    - IRA DEPENDENCY column: already a decimal fraction (0–1).
    """
    df_ari = pd.read_excel(
        xlsx_path,
        sheet_name=ARI_SHEET_NAME,
        header=ARI_HEADER_ROW,
    )

    # Keep only the columns we need; drop rows where both PROVINCE and LGU NAME are null
    df_ari = df_ari[['REGION', 'PROVINCE', 'LGU NAME', 'LGU TYPE', 'IRA DEPENDENCY']].copy()
    df_ari = df_ari.dropna(subset=['PROVINCE'])

    # Exclude summary/blank rows (header-echo rows from merged cells)
    df_ari = df_ari[df_ari['PROVINCE'].astype(str).str.strip() != 'PROVINCE']

    # Standardise string columns
    df_ari['PROVINCE'] = df_ari['PROVINCE'].astype(str).str.upper().str.strip()
    df_ari['LGU NAME'] = df_ari['LGU NAME'].astype(str).str.upper().str.strip()
    df_ari['REGION']   = df_ari['REGION'].astype(str).str.strip()

    # IRA DEPENDENCY is already a fraction; validate range
    df_ari['IRA DEPENDENCY'] = pd.to_numeric(df_ari['IRA DEPENDENCY'], errors='coerce')

    # ------------------------------------------------------------------
    # Assign Province labels:
    #   Non-NCR rows: use PROVINCE directly (after standardization)
    #   NCR rows: PROVINCE == "METRO MANILA" → redistribute by LGU NAME
    # ------------------------------------------------------------------
    df_ari['Province'] = df_ari['PROVINCE'].apply(standardize_province_names_scalar)

    ncr_mask = df_ari['REGION'].str.contains('NCR', case=False, na=False)
    for city_key, district_label in _NCR_CITY_TO_DISTRICT.items():
        lgu_mask = df_ari['LGU NAME'].str.contains(city_key, case=False, na=False)
        df_ari.loc[ncr_mask & lgu_mask, 'Province'] = district_label

    # Aggregate to province level (mean IRA dependency across LGUs)
    df_ira_prov = (
        df_ari.groupby('Province')['IRA DEPENDENCY']
        .mean()
        .reset_index()
        .rename(columns={'IRA DEPENDENCY': 'Mean_IRA_Dependency'})
    )

    # Ensure NCR 4th district entry exists alongside TAGUIG-PATEROS
    ncr_4th = df_ira_prov[df_ira_prov['Province'] == 'TAGUIG - PATEROS']
    if (not ncr_4th.empty
            and 'NCR - FOURTH DISTRICT' not in df_ira_prov['Province'].values):
        extra = ncr_4th.copy()
        extra['Province'] = 'NCR - FOURTH DISTRICT'
        df_ira_prov = pd.concat([df_ira_prov, extra], ignore_index=True)

    return df_ira_prov, ['Mean_IRA_Dependency']


def standardize_province_names_scalar(name: str) -> str:
    """Single-string wrapper around the series-based standardiser."""
    return standardize_province_names(pd.Series([name])).iloc[0]


# ---------------------------------------------------------------------------
# ETHNICITY (2020 PSA census — unchanged from 2019 pipeline)
# ---------------------------------------------------------------------------

def load_demographic_ethnicity_map(csv_path: Path, feature_provinces=None) -> dict:
    df_eth = pd.read_csv(csv_path)
    df_eth['Province'] = df_eth['Province'].astype(str).str.strip().str.upper()

    manual_map = {
        'COTABATO (NORTH COTABATO) 1':                      'COTABATO',
        'DAVAO DE ORO (COMPOSTELA VALLEY)':                 'DAVAO DE ORO',
        'MAGUINDANAO (INCLUDING THE CITY OF COTABATO)':     'MAGUINDANAO',
        'DAVAO DEL NORTE':                                  'DAVAO DEL NORTE',
        'BASILAN (EXCLUDING THE CITY OF ISABELA)':          'BASILAN',
        'SAMAR (WESTERN SAMAR)':                            'SAMAR',
        # NCR districts in ethnicity file use full "NATIONAL CAPITAL REGION" prefix
        'NATIONAL CAPITAL REGION - FOURTH DISTRICT':  'NCR - FOURTH DISTRICT',
        'NATIONAL CAPITAL REGION - MANILA':           'NCR - MANILA',
        'NATIONAL CAPITAL REGION - SECOND DISTRICT':  'NCR - SECOND DISTRICT',
        'NATIONAL CAPITAL REGION - THIRD DISTRICT':   'NCR - THIRD DISTRICT',
    }
    df_eth['Province'] = df_eth['Province'].replace(manual_map)

    # Replicate 4th district entry for TAGUIG-PATEROS
    ncr_4th = df_eth[df_eth['Province'] == 'NCR - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_eth = pd.concat([df_eth, extra], ignore_index=True)

    # Maguindanao split: replicate base Maguindanao entry for both sub-provinces
    mag_base = df_eth[df_eth['Province'] == 'MAGUINDANAO']
    for sub in ['MAGUINDANAO DEL NORTE', 'MAGUINDANAO DEL SUR']:
        if not mag_base.empty and sub not in df_eth['Province'].values:
            extra = mag_base.copy()
            extra['Province'] = sub
            df_eth = pd.concat([df_eth, extra], ignore_index=True)

    if feature_provinces is not None:
        missing = set(feature_provinces) - set(df_eth['Province'].unique())
        print("\n--- ETHNICITY SOURCE PROVINCE ALIGNMENT CHECK ---")
        if missing:
            print(f"[!] {len(missing)} provinces unmatched in Ethnicity dataset: "
                  f"{sorted(missing)}")
        else:
            print("[\u2713] All primary model provinces aligned with Ethnicity records.")

    return dict(zip(df_eth['Province'], df_eth['Dominant_Ethnicity']))


# ---------------------------------------------------------------------------
# DYNASTY (2019 proxy for 2022)
# ---------------------------------------------------------------------------

def process_dynasty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    """Load dynasty proxy variables.

    No 2022 data exists; DYNASTY_YEAR (2019) is used as the proxy.
    Province names in this file use caps with occasional NCR labels like
    'NCR, SECOND DISTRICT' that need remapping to the 2022 feature format.
    """
    df_dyn = pd.read_csv(csv_path)
    df_dyn = df_dyn[df_dyn['Year'] == DYNASTY_YEAR].copy()

    df_dyn['Province'] = standardize_province_names(df_dyn['Province'])

    dyn_cols = ['HHI', 'GINI']
    for col in dyn_cols:
        df_dyn[col] = pd.to_numeric(df_dyn[col], errors='coerce')

    # Replicate NCR 4th district row for TAGUIG-PATEROS
    ncr_4th = df_dyn[df_dyn['Province'] == 'NCR - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_dyn = pd.concat([df_dyn, extra], ignore_index=True)

    # Maguindanao split (2022 feature file has del Norte / del Sur)
    mag_base = df_dyn[df_dyn['Province'] == 'MAGUINDANAO']
    for sub in ['MAGUINDANAO DEL NORTE', 'MAGUINDANAO DEL SUR']:
        if not mag_base.empty and sub not in df_dyn['Province'].values:
            extra = mag_base.copy()
            extra['Province'] = sub
            df_dyn = pd.concat([df_dyn, extra], ignore_index=True)

    # Remove duplicate province rows before merging
    df_dyn = df_dyn.drop_duplicates(subset=['Province']).copy()

    return df_dyn[['Province'] + dyn_cols], dyn_cols


# ---------------------------------------------------------------------------
# MASTER MERGE
# ---------------------------------------------------------------------------

def merge_socioeconomic_data(df_features: pd.DataFrame,
                              pov_csv_path: Path = None,
                              ethnicity_csv_path: Path = ETHNICITY_DEPENDENCY_FILE,
                              ari_xlsx_path: Path = ARI_DEPENDENCY_FILE,
                              dynasty_csv_path: Path = None) -> pd.DataFrame:
    """Left-join poverty (2021), IRA dependency (2022), ethnicity, and
    dynasty (2019 proxy) onto the province-level topological feature table."""

    merged = df_features.copy()
    merged['Province'] = merged['Province'].str.strip().str.upper()
    feature_provinces = merged['Province'].unique()

    # -- Ethnicity --
    eth_map = load_demographic_ethnicity_map(
        ethnicity_csv_path, feature_provinces=feature_provinces
    )
    merged['Dominant_Dialect'] = merged['Province'].map(eth_map).fillna('N/A')

    impute_cols: list[str] = []

    # -- Poverty (2021 / 2023 fallback) --
    if pov_csv_path:
        print("\n-> Integrating Poverty Data (2021; 2023 fallback for Maguindanao splits)...")
        df_pov, pov_cols = process_poverty_data(pov_csv_path)
        df_pov['Province'] = df_pov['Province'].str.strip().str.upper()

        missing_pov = set(feature_provinces) - set(df_pov['Province'].unique())
        print("--- POVERTY DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_pov:
            print(f"[!] Provinces missing in Poverty dataset: {sorted(missing_pov)}")
        else:
            print("[\u2713] Perfect join found for Poverty Data.")

        merged = pd.merge(merged, df_pov[['Province'] + pov_cols],
                          on='Province', how='left')
        impute_cols.extend(pov_cols)

    # -- IRA Dependency (2022 XLSX) --
    if ari_xlsx_path:
        print("\n-> Integrating LGU IRA Dependency Data (FY2022)...")
        df_ira, ira_cols = process_ira_data(ari_xlsx_path)
        df_ira['Province'] = df_ira['Province'].str.strip().str.upper()

        missing_ira = set(feature_provinces) - set(df_ira['Province'].unique())
        print("--- IRA DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_ira:
            print(f"[!] Provinces missing in IRA dataset: {sorted(missing_ira)}")
        else:
            print("[\u2713] Perfect join found for IRA Dependency Data.")

        merged = pd.merge(merged, df_ira, on='Province', how='left')
        impute_cols.extend(ira_cols)

    # -- Dynasty (2019 proxy) --
    if dynasty_csv_path:
        print(f"\n-> Integrating Dynasty & Inequality Data ({DYNASTY_YEAR} proxy for 2022)...")
        df_dyn, dyn_cols = process_dynasty_data(dynasty_csv_path)
        df_dyn['Province'] = df_dyn['Province'].str.strip().str.upper()

        missing_dyn = set(feature_provinces) - set(df_dyn['Province'].unique())
        print("--- DYNASTY DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_dyn:
            print(f"[!] Provinces missing in Dynasty dataset: {sorted(missing_dyn)}")
        else:
            print("[\u2713] Perfect join found for Dynasty Data.")

        if 'Region' in df_dyn.columns:
            df_dyn = df_dyn.drop(columns=['Region'])

        merged = pd.merge(merged, df_dyn, on='Province', how='left')

        print("-> Imputing missing Dynasty (HHI/GINI) values using local Regional Means...")
        for col in dyn_cols:
            if merged[col].isna().any():
                merged[col] = merged.groupby('Region')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                if merged[col].isna().any():
                    merged[col] = merged[col].fillna(merged[col].mean())

        impute_cols.extend(dyn_cols)

    # Report any remaining NaN after all imputation
    for col in impute_cols:
        if col in merged.columns:
            n_missing = merged[col].isna().sum()
            if n_missing > 0:
                missing_provs = merged.loc[merged[col].isna(), 'Province'].tolist()
                print(f"   [!] {n_missing} missing values remaining for '{col}': "
                      f"{missing_provs}")

    print("\n--- SOCIOECONOMIC DATA INTEGRATION COMPLETE ---")
    return merged
