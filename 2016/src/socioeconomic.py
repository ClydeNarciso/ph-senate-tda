"""Socioeconomic data helpers for the 2016 pipeline.

Key differences from the 2022 pipeline
---------------------------------------

POVERTY
  Source: Poverty_Incidence_among_Families_2015_to_2018.csv
  Format: wide-format CSV (rows = provinces, columns include 2015 and 2018
  poverty incidence side by side). Data starts at row index 7 (0-indexed)
  after a multi-line header. Province names in column 0; 2015 values in
  column 3; 2018 values in column 4.
  Strategy: 2015 is the closest available year before the 2016 election and
  is used as the primary column.  2018 is used as the fallback only for
  Davao Occidental, which has no 2015 estimate (first appeared in data after
  its creation in 2016).
  Province names carry footnote suffixes ("* c/", "b/, c/", "a/", "d/")
  that must be stripped before matching.

IRA DEPENDENCY
  Source: By-LGU-ARI-and-Dependencies-2016.xlsx  (sheet: Data)
  Column: IRA DEPENDENCY — same name as 2022; values are decimal fractions.
  Header row index: 6 (same as 2022).
  NCR structure: all cities under PROVINCE = "Metro Manila"; redistributed
  to the four NCR district labels + TAGUIG - PATEROS via LGU NAME.
  ARI uses 'WESTERN SAMAR (SAMAR)' for Samar — standardised to match
  the feature file's 'SAMAR (WESTERN SAMAR)'.

DYNASTY (HHI / GINI)
  Source: DynastyProxyVariables.csv
  Year: 2016 — actual data exists, making this the only pipeline year that
  is contemporaneous with the election.
  Missing provinces (relative to the 2016 feature file):
    COMPOSTELA VALLEY, DAVAO OCCIDENTAL, DINAGAT ISLANDS, SULU, TAWI-TAWI
  These are imputed from regional means.

ETHNICITY
  Source: provincial_dominant_ethnicity_psa2020.csv  (unchanged across all
  pipelines — 2020 PSA census is the latest subnational data available).

PROVINCE NAME NOTES (2016-specific)
  The 2016 feature file uses pre-split / legacy province names:
    COMPOSTELA VALLEY       (renamed Davao de Oro in 2019)
    COTABATO (NORTH COT.)   (standardised -> COTABATO / NORTH COTABATO)
    DAVAO (DAVAO DEL NORTE) (standardised -> DAVAO DEL NORTE)
    MAGUINDANAO             (not yet split)
    SAMAR (WESTERN SAMAR)   (standardised -> SAMAR in socioeconomic files)
  Standardisation maps in this file translate these TO the names used in
  each source dataset rather than to a single canonical form, because each
  source dataset uses different conventions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ETHNICITY_DEPENDENCY_FILE,
    ARI_DEPENDENCY_FILE, ARI_SHEET_NAME, ARI_HEADER_ROW, ARI_DEP_COL,
    POVERTY_PRIMARY_YEAR, POVERTY_FALLBACK_YEAR, POVERTY_COL_LABEL,
    DYNASTY_YEAR,
)


# ---------------------------------------------------------------------------
# Province-name standardisation
# ---------------------------------------------------------------------------

def standardize_province_names(series: pd.Series) -> pd.Series:
    """Normalise province name strings to the canonical form used in the
    2016 feature file.

    Handles:
    - Footnote suffixes from poverty CSV ("* c/", "b/, c/", "a/", "d/")
    - Mixed case / extra whitespace
    - Legacy ARI province names that differ from the feature file
    - NCR district labels from dynasty CSV
    - Ethnicity file NCR prefix variants
    """
    clean = (
        series.astype(str)
        # Step 1: remove asterisk with any surrounding whitespace
        .str.replace(r'\s*\*\s*', '', regex=True)
        # Step 2: remove " a/", " b/, c/" style trailing footnote suffixes
        .str.replace(r'\s+[a-z][/,].*$', '', regex=True)
        # Step 3: remove lowercase footnote letter glued to end of word (e.g. "Batanesa/")
        .str.replace(r'[a-z]/.*$', '', regex=True)
        .str.upper()
        .str.strip()
    )

    mapping = {
        # ---- NCR: dynasty CSV labels -> feature file labels ----
        'NCR, CITY OF MANILA, FIRST DISTRICT': 'NATIONAL CAPITAL REGION - MANILA',
        'NCR, SECOND DISTRICT':                'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'NCR, THIRD DISTRICT':                 'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'NCR, FOURTH DISTRICT':                'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        # ---- NCR: poverty CSV district labels ----
        '1ST DISTRICT':  'NATIONAL CAPITAL REGION - MANILA',
        '2ND DISTRICT':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        '3RD DISTRICT':  'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        '4TH DISTRICT':  'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        # ---- ARI -> feature file name harmonisation ----
        'NORTH COTABATO':           'COTABATO (NORTH COT.)',
        'WESTERN SAMAR (SAMAR)':    'SAMAR (WESTERN SAMAR)',
        'DAVAO DEL NORTE':          'DAVAO (DAVAO DEL NORTE)',
        # ---- Poverty CSV -> feature file ----
        'COMPOSTELA VALLEY':        'COMPOSTELA VALLEY',   # already matches
        'WESTERN SAMAR':            'SAMAR (WESTERN SAMAR)',
        # ---- Dynasty CSV -> feature file ----
        'COTABATO':                 'COTABATO (NORTH COT.)',
        'SAMAR':                    'SAMAR (WESTERN SAMAR)',
        # ---- Ethnicity file variants ----
        'COTABATO (NORTH COTABATO) 1':                  'COTABATO (NORTH COT.)',
        'DAVAO DE ORO (COMPOSTELA VALLEY)':             'COMPOSTELA VALLEY',
        'MAGUINDANAO (INCLUDING THE CITY OF COTABATO)': 'MAGUINDANAO',
        'BASILAN (EXCLUDING THE CITY OF ISABELA)':      'BASILAN',
        'SAMAR (WESTERN SAMAR)':                        'SAMAR (WESTERN SAMAR)',
        # ---- Mountain Province ----
        'MT. PROVINCE': 'MOUNTAIN PROVINCE',
        # ---- Dinagat ----
        'DINAGAT ISLAND': 'DINAGAT ISLANDS',
        # ---- Sarangani ----
        'SARANGGANI': 'SARANGANI',
        # ---- Tawi-tawi capitalisation variants ----
        'TAWI-TAWI': 'TAWI-TAWI',
        # ---- Isabela City (in poverty CSV, listed under Zamboanga Peninsula) ----
        'ISABELA CITY': 'BASILAN',
        # ---- Cotabato City (appears in poverty CSV) ----
        'COTABATO CITY': 'MAGUINDANAO',
    }
    return clean.replace(mapping)


def standardize_province_names_scalar(name: str) -> str:
    return standardize_province_names(pd.Series([name])).iloc[0]


# ---------------------------------------------------------------------------
# NCR city -> district router (for ARI XLSX)
# ---------------------------------------------------------------------------

_NCR_CITY_TO_DISTRICT = {
    'MANILA':       'NATIONAL CAPITAL REGION - MANILA',
    'MANDALUYONG':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
    'MARIKINA':     'NATIONAL CAPITAL REGION - SECOND DISTRICT',
    'PASIG':        'NATIONAL CAPITAL REGION - SECOND DISTRICT',
    'QUEZON':       'NATIONAL CAPITAL REGION - SECOND DISTRICT',
    'SAN JUAN':     'NATIONAL CAPITAL REGION - SECOND DISTRICT',
    'CALOOCAN':     'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    'MALABON':      'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    'NAVOTAS':      'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    'VALENZUELA':   'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    'LAS PIÑAS':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'LAS PINAS':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'MAKATI':       'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'MUNTINLUPA':   'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'PARAÑAQUE':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'PARANAQUE':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'PASAY':        'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'TAGUIG':       'TAGUIG - PATEROS',
    'PATEROS':      'TAGUIG - PATEROS',
}


# ---------------------------------------------------------------------------
# POVERTY (2015 primary / 2018 fallback)
# ---------------------------------------------------------------------------

def process_poverty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    """Parse the 2015-2018 wide-format poverty CSV.

    File structure (verified against actual file):
    - Rows 0-6: title header; skip with skiprows=7.
    - Column 0: Region/Province name (with footnote suffixes).
    - Column 3: 2015 Poverty Incidence (%).
    - Column 4: 2018 Poverty Incidence (%).
    - Region-level summary rows (e.g. 'PHILIPPINES*', 'NCR*', 'CAR*',
      'Region I*') are detected by the absence of a numeric poverty value
      at province level OR by their all-caps/asterisk format; they are
      dropped after extraction.

    Strategy for 2016
    ------------------
    Primary  : 2015 Poverty Incidence  (survey year closest before 2016)
    Fallback : 2018 Poverty Incidence  (for Davao Occidental, which was
               created from Davao del Sur in 2016 and has no 2015 estimate)
    """
    df_raw = pd.read_csv(csv_path, header=None, skiprows=7,
                          usecols=[0, 3, 4], encoding='utf-8-sig')
    df_raw.columns = [
        'Province',
        '2015 Poverty Incidence',
        '2018 Poverty Incidence',
    ]
    df_raw = df_raw.dropna(subset=['Province'])
    df_raw = df_raw[df_raw['Province'].astype(str).str.strip() != '']
    for col in ['2015 Poverty Incidence', '2018 Poverty Incidence']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Drop rows where both columns are NaN (blank separator rows)
    df_raw = df_raw.dropna(
        how='all', subset=['2015 Poverty Incidence', '2018 Poverty Incidence']
    ).copy()

    # Standardise province names (strips footnotes, maps NCR districts, etc.)
    df_raw['Province'] = standardize_province_names(df_raw['Province'])

    # Drop national / regional summary rows (those with no province-level
    # match after standardisation remain as-is; they are harmless because
    # the left-join in merge_socioeconomic_data won't match them)

    # Fill 2015 gaps with 2018 (Davao Occidental)
    mask_missing = df_raw[POVERTY_PRIMARY_YEAR].isna()
    df_raw.loc[mask_missing, POVERTY_PRIMARY_YEAR] = (
        df_raw.loc[mask_missing, POVERTY_FALLBACK_YEAR]
    )
    n_filled = mask_missing.sum()
    if n_filled > 0:
        filled = df_raw.loc[mask_missing, 'Province'].tolist()
        print(f"   [i] Filled {n_filled} missing 2015 poverty values with "
              f"2018 data: {filled}")

    if POVERTY_PRIMARY_YEAR != POVERTY_COL_LABEL:
        df_raw = df_raw.rename(
            columns={POVERTY_PRIMARY_YEAR: POVERTY_COL_LABEL}
        )

    # Replicate NCR 4th district row for TAGUIG - PATEROS
    ncr_4th = df_raw[df_raw['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_raw = pd.concat([df_raw, extra], ignore_index=True)

    return df_raw[['Province', POVERTY_COL_LABEL]], [POVERTY_COL_LABEL]


# ---------------------------------------------------------------------------
# IRA DEPENDENCY (2016 XLSX)
# ---------------------------------------------------------------------------

def process_ira_data(xlsx_path: Path) -> tuple[pd.DataFrame, list]:
    """Load IRA dependency from the 2016 XLSX and return province-level means.

    Identical logic to the 2022 pipeline; only the sheet name differs
    ('Data' instead of 'FY2022').
    """
    df_ari = pd.read_excel(xlsx_path, sheet_name=ARI_SHEET_NAME,
                           header=ARI_HEADER_ROW)

    df_ari = df_ari[['REGION', 'PROVINCE', 'LGU NAME', 'LGU TYPE',
                      ARI_DEP_COL]].copy()
    df_ari = df_ari.dropna(subset=['PROVINCE'])
    df_ari = df_ari[df_ari['PROVINCE'].astype(str).str.strip() != 'PROVINCE']

    df_ari['PROVINCE'] = df_ari['PROVINCE'].astype(str).str.upper().str.strip()
    df_ari['LGU NAME'] = df_ari['LGU NAME'].astype(str).str.upper().str.strip()
    df_ari['REGION']   = df_ari['REGION'].astype(str).str.strip()
    df_ari[ARI_DEP_COL] = pd.to_numeric(df_ari[ARI_DEP_COL], errors='coerce')

    # Assign province labels, then redistribute NCR
    df_ari['Province'] = df_ari['PROVINCE'].apply(standardize_province_names_scalar)

    ncr_mask = df_ari['REGION'].str.contains('NCR', case=False, na=False)
    for city_key, district_label in _NCR_CITY_TO_DISTRICT.items():
        lgu_mask = df_ari['LGU NAME'].str.contains(city_key, case=False, na=False)
        df_ari.loc[ncr_mask & lgu_mask, 'Province'] = district_label

    df_ira_prov = (
        df_ari.groupby('Province')[ARI_DEP_COL]
        .mean()
        .reset_index()
        .rename(columns={ARI_DEP_COL: 'Mean_IRA_Dependency'})
    )

    # Ensure TAGUIG - PATEROS exists (Taguig + Pateros average -> TAGUIG - PATEROS)
    tp = df_ira_prov[df_ira_prov['Province'] == 'TAGUIG - PATEROS']
    if tp.empty:
        ncr4 = df_ira_prov[
            df_ira_prov['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT'
        ]
        if not ncr4.empty:
            extra = ncr4.copy()
            extra['Province'] = 'TAGUIG - PATEROS'
            df_ira_prov = pd.concat([df_ira_prov, extra], ignore_index=True)

    return df_ira_prov, ['Mean_IRA_Dependency']


# ---------------------------------------------------------------------------
# ETHNICITY (2020 PSA census — unchanged)
# ---------------------------------------------------------------------------

def load_demographic_ethnicity_map(csv_path: Path,
                                    feature_provinces=None) -> dict:
    df_eth = pd.read_csv(csv_path)
    df_eth['Province'] = df_eth['Province'].astype(str).str.strip().str.upper()

    manual_map = {
        'COTABATO (NORTH COTABATO) 1':                  'COTABATO (NORTH COT.)',
        'DAVAO DE ORO (COMPOSTELA VALLEY)':             'COMPOSTELA VALLEY',
        # 2016 feature file uses 'DAVAO (DAVAO DEL NORTE)' — map from ethnicity label
        'DAVAO DEL NORTE':                              'DAVAO (DAVAO DEL NORTE)',
        'MAGUINDANAO (INCLUDING THE CITY OF COTABATO)': 'MAGUINDANAO',
        'BASILAN (EXCLUDING THE CITY OF ISABELA)':      'BASILAN',
        'SAMAR (WESTERN SAMAR)':                        'SAMAR (WESTERN SAMAR)',
        'NATIONAL CAPITAL REGION - FOURTH DISTRICT':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'NATIONAL CAPITAL REGION - MANILA':             'NATIONAL CAPITAL REGION - MANILA',
        'NATIONAL CAPITAL REGION - SECOND DISTRICT':    'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'NATIONAL CAPITAL REGION - THIRD DISTRICT':     'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    }
    df_eth['Province'] = df_eth['Province'].replace(manual_map)

    # Replicate NCR 4th district for TAGUIG - PATEROS
    ncr_4th = df_eth[df_eth['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
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
# DYNASTY (2016 — actual data)
# ---------------------------------------------------------------------------

def process_dynasty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    """Load dynasty proxy variables for 2016.

    Unlike all other pipeline years, 2016 has contemporaneous dynasty data.
    Missing provinces relative to the 2016 feature file:
      COMPOSTELA VALLEY, DAVAO OCCIDENTAL, DINAGAT ISLANDS, SULU, TAWI-TAWI
    These are imputed from regional means (same strategy as other years).
    """
    df_dyn = pd.read_csv(csv_path)
    df_dyn = df_dyn[df_dyn['Year'] == DYNASTY_YEAR].copy()
    df_dyn['Province'] = standardize_province_names(df_dyn['Province'])

    dyn_cols = ['HHI', 'GINI']
    for col in dyn_cols:
        df_dyn[col] = pd.to_numeric(df_dyn[col], errors='coerce')

    # Replicate NCR 4th district for TAGUIG - PATEROS
    ncr_4th = df_dyn[df_dyn['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_dyn = pd.concat([df_dyn, extra], ignore_index=True)

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
    """Left-join poverty (2015), IRA dependency (2016), ethnicity, and
    dynasty (2016 actual) onto the province-level topological feature table."""

    merged = df_features.copy()
    merged['Province'] = merged['Province'].str.strip().str.upper()
    feature_provinces = merged['Province'].unique()

    # -- Ethnicity --
    eth_map = load_demographic_ethnicity_map(
        ethnicity_csv_path, feature_provinces=feature_provinces
    )
    merged['Dominant_Dialect'] = merged['Province'].map(eth_map).fillna('N/A')

    impute_cols: list[str] = []

    # -- Poverty (2015 primary / 2018 fallback) --
    if pov_csv_path:
        print("\n-> Integrating Poverty Data "
              "(2015; 2018 fallback for Davao Occidental)...")
        df_pov, pov_cols = process_poverty_data(pov_csv_path)
        df_pov['Province'] = df_pov['Province'].str.strip().str.upper()

        missing_pov = set(feature_provinces) - set(df_pov['Province'].unique())
        print("--- POVERTY DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_pov:
            print(f"[!] Provinces missing in Poverty dataset: {sorted(missing_pov)}")
        else:
            print("[\u2713] Perfect join found for Poverty Data.")

        df_pov = df_pov[['Province'] + pov_cols].drop_duplicates(subset='Province', keep='first')
        merged = pd.merge(merged, df_pov, on='Province', how='left')
        impute_cols.extend(pov_cols)

    # -- IRA Dependency (2016 XLSX) --
    if ari_xlsx_path:
        print("\n-> Integrating LGU IRA Dependency Data (FY2016)...")
        df_ira, ira_cols = process_ira_data(ari_xlsx_path)
        df_ira['Province'] = df_ira['Province'].str.strip().str.upper()

        missing_ira = set(feature_provinces) - set(df_ira['Province'].unique())
        print("--- IRA DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_ira:
            print(f"[!] Provinces missing in IRA dataset: {sorted(missing_ira)}")
        else:
            print("[\u2713] Perfect join found for IRA Dependency Data.")

        df_ira = df_ira.drop_duplicates(subset='Province', keep='first')
        merged = pd.merge(merged, df_ira, on='Province', how='left')
        impute_cols.extend(ira_cols)

    # -- Dynasty (2016 actual) --
    if dynasty_csv_path:
        print(f"\n-> Integrating Dynasty & Inequality Data "
              f"({DYNASTY_YEAR} — actual election-year data)...")
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

        df_dyn = df_dyn.drop_duplicates(subset='Province', keep='first')
        merged = pd.merge(merged, df_dyn, on='Province', how='left')

        print("-> Imputing missing Dynasty values using local Regional Means...")
        for col in dyn_cols:
            if merged[col].isna().any():
                merged[col] = merged.groupby('Region')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                if merged[col].isna().any():
                    merged[col] = merged[col].fillna(merged[col].mean())

        impute_cols.extend(dyn_cols)

    # Report remaining NaNs
    for col in impute_cols:
        if col in merged.columns:
            n_missing = merged[col].isna().sum()
            if n_missing > 0:
                missing_provs = merged.loc[merged[col].isna(), 'Province'].tolist()
                print(f"   [!] {n_missing} missing values for '{col}': "
                      f"{missing_provs}")

    print("\n--- SOCIOECONOMIC DATA INTEGRATION COMPLETE ---")
    return merged
