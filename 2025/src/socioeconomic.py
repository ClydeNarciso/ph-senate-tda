"""Socioeconomic data helpers for the 2025 pipeline.

Key differences from the 2022 pipeline
---------------------------------------

POVERTY
  Source: PSA_Full_Year_Poverty_Statistics_-_2018_2021_2023.csv
  Format: semicolon-delimited, hierarchical (regions indented with '..' /
  '....'; provinces with '....').  The Geolocation column contains both
  region and province labels — province rows are extracted by stripping the
  leading dots.
  Strategy: 2023 is the closest available year to the 2025 election and is
  used as the primary column.  Maguindanao del Norte and del Sur have no
  separate 2023 entry (listed only as a combined 'Maguindanao' in 2023);
  their 2021 figures are used as the fallback.

NTA DEPENDENCY (was IRA DEPENDENCY in 2022)
  Source: By-LGU-ARI-and-Dependencies-2025.xlsx  (sheet: ARI 2025)
  Column renamed from 'IRA DEPENDENCY' to 'NTA DEPENDENCY' in the 2025
  release (National Tax Allotment replaced IRA as the statutory term).
  Header row, fraction format, and NCR city-to-district logic are
  otherwise identical to the 2022 pipeline.

FLEMMS 2024 (NEW — replaces poverty as an additional socioeconomic layer)
  Source: 2024_FLEMMS_Statistical_Tables_Press_Release_2.xlsx, Table 3.
  Metric: Functional Literacy Rate (both sexes, %) for the population
  10-64 years old.
  Structural note: provinces with Highly Urbanized Cities (HUCs) are split
  into e.g. "Benguet (Excluding City of Baguio)" + "City of Baguio".  We
  do a simple average of the province-proper and its HUC(s) to get a single
  province-level literacy figure (population weights are not available at
  this level of extraction without reading population columns separately;
  simple averaging is a close enough approximation for the BallMapper use).

DYNASTY (HHI / GINI)
  Source: DynastyProxyVariables.csv  (unchanged)
  Last available year is 2019; used as a proxy for 2025.

ETHNICITY
  Source: provincial_dominant_ethnicity_psa2020.csv  (unchanged)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ETHNICITY_DEPENDENCY_FILE,
    ARI_DEPENDENCY_FILE, ARI_SHEET_NAME, ARI_HEADER_ROW, ARI_DEP_COL,
    POVERTY_PRIMARY_YEAR, POVERTY_FALLBACK_YEAR, POVERTY_COL_LABEL,
    DYNASTY_YEAR,
    FLEMMS_FILE, FLEMMS_SHEET, FLEMMS_COL_LABEL,
)


# ---------------------------------------------------------------------------
# Province-name standardisation
# ---------------------------------------------------------------------------

def standardize_province_names(series: pd.Series) -> pd.Series:
    """Normalise province name strings to the canonical form used in the
    2025 feature file (full 'NATIONAL CAPITAL REGION - X' for NCR).

    Handles:
    - PSA footnote suffixes ("a/", "b/", "r1" etc.)
    - Leading dot-indent hierarchy markers ('....') from the poverty CSV
    - Mixed case / extra whitespace
    - HUC exclusion parentheticals from FLEMMS
      ("Benguet (Excluding City of Baguio)" -> "BENGUET")
    """
    clean = (
        series.astype(str)
        # Strip FLEMMS parenthetical "excluding" qualifiers before uppercasing
        .str.replace(r'\s*\(Excluding[^)]*\)', '', regex=True)
        .str.replace(r'\s*\(Including[^)]*\)', '', regex=True)
        # Strip PSA footnote suffixes
        .str.replace(r'\s+[a-z0-9]+[/,].*', '', regex=True)
        # Strip leading dot-indent markers from hierarchical poverty CSV
        .str.replace(r'^\.+', '', regex=True)
        .str.upper()
        .str.strip()
    )

    mapping = {
        # ---- NCR: 2025 feature file uses full prefix ----
        'NATIONAL CAPITAL REGION (NCR)':           'NATIONAL CAPITAL REGION - MANILA',
        '1ST DISTRICT':  'NATIONAL CAPITAL REGION - MANILA',
        '2ND DISTRICT':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        '3RD DISTRICT':  'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        '4TH DISTRICT':  'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        # ---- Dynasty CSV NCR labels ----
        'NCR, CITY OF MANILA, FIRST DISTRICT':  'NATIONAL CAPITAL REGION - MANILA',
        'NCR, SECOND DISTRICT':                 'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'NCR, THIRD DISTRICT':                  'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'NCR, FOURTH DISTRICT':                 'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        # ---- Cotabato ----
        'NORTH COTABATO':           'COTABATO',
        'COTABATO (NORTH COT.)':    'COTABATO',
        # ---- Samar ----
        'SAMAR (WESTERN SAMAR)':    'SAMAR',
        # ---- Mountain Province ----
        'MT. PROVINCE':             'MOUNTAIN PROVINCE',
        # ---- Dinagat ----
        'DINAGAT ISLAND':           'DINAGAT ISLANDS',
        # ---- Sarangani ----
        'SARANGGANI':               'SARANGANI',
        # ---- Davao ----
        'COMPOSTELA VALLEY':        'DAVAO DE ORO',
        'DAVAO (DAVAO DEL NORTE)':  'DAVAO DEL NORTE',
        # ---- FLEMMS HUC city labels that need to collapse to province ----
        'CITY OF BAGUIO':           'BENGUET',
        'CITY OF ANGELES':          'PAMPANGA',
        'CITY OF OLONGAPO':         'ZAMBALES',
        'CITY OF LUCENA':           'QUEZON',
        'CITY OF PUERTO PRINCESA':  'PALAWAN',
        'CITY OF ILOILO':           'ILOILO',
        'CITY OF BACOLOD':          'NEGROS OCCIDENTAL',
        'CITY OF CEBU':             'CEBU',
        'CITY OF LAPU-LAPU':        'CEBU',
        'CITY OF MANDAUE':          'CEBU',
        'CITY OF TACLOBAN':         'LEYTE',
        'CITY OF ISABELA':          'BASILAN',    # Isabela City is in Basilan
        'CITY OF ZAMBOANGA':        'ZAMBOANGA DEL SUR',
        'CITY OF ILIGAN':           'LANAO DEL NORTE',
        'CITY OF CAGAYAN DE ORO':   'MISAMIS ORIENTAL',
        'CITY OF DAVAO':            'DAVAO DEL SUR',
        'CITY OF GENERAL SANTOS':   'SOUTH COTABATO',
        'CITY OF BUTUAN':           'AGUSAN DEL NORTE',
        # ---- NCR HUC cities in FLEMMS collapse to districts ----
        'CITY OF MANDALUYONG':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'CITY OF MANILA':       'NATIONAL CAPITAL REGION - MANILA',
        'CITY OF SAN JUAN':     'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'CITY OF MARIKINA':     'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'QUEZON CITY':          'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'CITY OF MAKATI':       'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF PASIG':        'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'CITY OF TAGUIG':       'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'PATEROS':              'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF CALOOCAN':     'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'CITY OF MALABON':      'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'CITY OF NAVOTAS':      'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'CITY OF VALENZUELA':   'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'CITY OF LAS PIÑAS':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF LAS PINAS':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF MUNTINLUPA':   'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF PARAÑAQUE':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF PARANAQUE':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'PASAY CITY':           'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        # ---- Maguindanao split ----
        'MAGUINDANAO DEL NORTE ':   'MAGUINDANAO DEL NORTE',
        'MAGUINDANAO DEL SUR ':     'MAGUINDANAO DEL SUR',
        # Strip trailing spaces that appear in FLEMMS labels
        'MAGUINDANAO DEL NORTE (INCLUDING COTABATO CITY)': 'MAGUINDANAO DEL NORTE',
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
    'TAGUIG':       'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
    'PATEROS':      'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
}


# ---------------------------------------------------------------------------
# POVERTY (2023 primary / 2021 fallback)
# ---------------------------------------------------------------------------

def process_poverty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    """Parse the hierarchical PSA poverty CSV and return a province-level table.

    The CSV has a single Geolocation column whose values use dot-indentation
    to encode hierarchy (e.g. '....Abra' is a province under a region row
    that starts with '..').  Province rows always have exactly four leading
    dots.  Region rows have two.  National/summary rows have none.

    Closest-year strategy for 2025
    --------------------------------
    Primary  : 2023 Poverty Incidence  (survey closest to 2025 election)
    Fallback : 2021 Poverty Incidence  (for Maguindanao del Norte / del Sur
               which do not have separate 2023 entries — only a combined
               'Maguindanao' 2023 figure exists)
    """
    df_raw = pd.read_csv(csv_path, sep=';', skiprows=1, header=0)
    df_raw.columns = [
        'Geolocation',
        '2018 Poverty Incidence',
        '2021 Poverty Incidence',
        '2023 Poverty Incidence',
    ]
    df_raw = df_raw.replace('..', np.nan)
    for col in ['2018 Poverty Incidence', '2021 Poverty Incidence',
                '2023 Poverty Incidence']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Province rows: exactly four leading dots (sub-province rows like
    # Maguindanao del Norte have six leading dots — keep those too)
    prov_mask = df_raw['Geolocation'].astype(str).str.match(r'^\.{4,}')
    df_prov = df_raw[prov_mask].copy()

    # Strip leading dots and clean province name
    df_prov['Province'] = standardize_province_names(df_prov['Geolocation'])

    # Fill 2023 gaps with 2021 values (Maguindanao del Norte / del Sur)
    mask_missing_primary = df_prov[POVERTY_PRIMARY_YEAR].isna()
    df_prov.loc[mask_missing_primary, POVERTY_PRIMARY_YEAR] = (
        df_prov.loc[mask_missing_primary, POVERTY_FALLBACK_YEAR]
    )
    n_filled = mask_missing_primary.sum()
    if n_filled > 0:
        filled = df_prov.loc[mask_missing_primary, 'Province'].tolist()
        print(f"   [i] Filled {n_filled} missing 2023 poverty values with "
              f"2021 data: {filled}")

    if POVERTY_PRIMARY_YEAR != POVERTY_COL_LABEL:
        df_prov = df_prov.rename(
            columns={POVERTY_PRIMARY_YEAR: POVERTY_COL_LABEL}
        )

    df_prov = df_prov[['Province', POVERTY_COL_LABEL]].copy()

    # Replicate NCR district rows that map from sub-labels
    # The poverty CSV only has '1st District', '2nd District', etc.
    # standardize_province_names already maps those to the full NCR labels.

    return df_prov, [POVERTY_COL_LABEL]


# ---------------------------------------------------------------------------
# NTA DEPENDENCY (2025 XLSX — was IRA DEPENDENCY in 2022)
# ---------------------------------------------------------------------------

def process_nta_data(xlsx_path: Path) -> tuple[pd.DataFrame, list]:
    """Load NTA dependency from the 2025 XLSX and return province-level means.

    Structural notes (verified against actual file)
    -----------------------------------------------
    - Sheet: ARI 2025
    - Column headers at Excel row 7 (pandas header=6).
    - Dependency column: 'NTA DEPENDENCY' (replaces 'IRA DEPENDENCY').
    - NCR: all cities listed under PROVINCE = 'Metro Manila'; redistributed
      to NCR district labels via LGU NAME, same as 2022.
    - Values: decimal fractions (0-1).
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

    # Assign province labels; redistribute NCR via LGU NAME
    df_ari['Province'] = df_ari['PROVINCE'].apply(standardize_province_names_scalar)

    ncr_mask = df_ari['REGION'].str.contains('NCR', case=False, na=False)
    for city_key, district_label in _NCR_CITY_TO_DISTRICT.items():
        lgu_mask = df_ari['LGU NAME'].str.contains(city_key, case=False, na=False)
        df_ari.loc[ncr_mask & lgu_mask, 'Province'] = district_label

    df_nta_prov = (
        df_ari.groupby('Province')[ARI_DEP_COL]
        .mean()
        .reset_index()
        .rename(columns={ARI_DEP_COL: 'Mean_NTA_Dependency'})
    )

    return df_nta_prov, ['Mean_NTA_Dependency']


# ---------------------------------------------------------------------------
# FLEMMS 2024 — Functional Literacy Rate
# ---------------------------------------------------------------------------

def process_flemms_data(xlsx_path: Path,
                         feature_provinces=None) -> tuple[pd.DataFrame, list]:
    """Extract province-level functional literacy rates from FLEMMS Table 3.

    The table lists provinces and, where present, their HUCs as separate
    rows (e.g. 'Benguet (Excluding City of Baguio)' and 'City of Baguio').
    Both rows are mapped to the same province via standardize_province_names
    and then averaged to produce a single province-level rate.

    Column used: col index 8 = 'Functional Literacy Rate (Both Sexes, %)'
    """
    df_raw = pd.read_excel(xlsx_path, sheet_name=FLEMMS_SHEET, header=None)
    data = df_raw[[1, 8]].copy()
    data.columns = ['Geolocation', FLEMMS_COL_LABEL]
    data = data.dropna(subset=['Geolocation', FLEMMS_COL_LABEL])
    data[FLEMMS_COL_LABEL] = pd.to_numeric(data[FLEMMS_COL_LABEL], errors='coerce')
    data = data.dropna(subset=[FLEMMS_COL_LABEL])

    # Drop title/header/notes rows
    exclude_keywords = r'Table|Region\s*/\s*Province|Notes|Source|Philippines'
    data = data[~data['Geolocation'].astype(str).str.contains(
        exclude_keywords, case=False, regex=True
    )]

    # Standardise names (collapses HUC cities into their parent province)
    data['Province'] = standardize_province_names(data['Geolocation'])

    # Average duplicate rows that arose from HUC splitting
    df_flemms = (
        data.groupby('Province')[FLEMMS_COL_LABEL]
        .mean()
        .reset_index()
    )

    if feature_provinces is not None:
        missing = set(feature_provinces) - set(df_flemms['Province'].unique())
        print("\n--- FLEMMS SOURCE PROVINCE ALIGNMENT CHECK ---")
        if missing:
            print(f"[!] {len(missing)} provinces unmatched in FLEMMS dataset: "
                  f"{sorted(missing)}")
        else:
            print("[\u2713] All primary model provinces aligned with FLEMMS records.")

    return df_flemms, [FLEMMS_COL_LABEL]


# ---------------------------------------------------------------------------
# ETHNICITY (2020 PSA census — unchanged)
# ---------------------------------------------------------------------------

def load_demographic_ethnicity_map(csv_path: Path,
                                    feature_provinces=None) -> dict:
    df_eth = pd.read_csv(csv_path)
    df_eth['Province'] = df_eth['Province'].astype(str).str.strip().str.upper()

    manual_map = {
        'COTABATO (NORTH COTABATO) 1':                  'COTABATO',
        'DAVAO DE ORO (COMPOSTELA VALLEY)':             'DAVAO DE ORO',
        'MAGUINDANAO (INCLUDING THE CITY OF COTABATO)': 'MAGUINDANAO DEL NORTE',
        'BASILAN (EXCLUDING THE CITY OF ISABELA)':      'BASILAN',
        'SAMAR (WESTERN SAMAR)':                        'SAMAR',
        # 2025 NCR format
        'NATIONAL CAPITAL REGION - FOURTH DISTRICT':    'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'NATIONAL CAPITAL REGION - MANILA':             'NATIONAL CAPITAL REGION - MANILA',
        'NATIONAL CAPITAL REGION - SECOND DISTRICT':    'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'NATIONAL CAPITAL REGION - THIRD DISTRICT':     'NATIONAL CAPITAL REGION - THIRD DISTRICT',
    }
    df_eth['Province'] = df_eth['Province'].replace(manual_map)

    # Maguindanao split
    mag_base = df_eth[df_eth['Province'] == 'MAGUINDANAO DEL NORTE']
    if not mag_base.empty and 'MAGUINDANAO DEL SUR' not in df_eth['Province'].values:
        extra = mag_base.copy()
        extra['Province'] = 'MAGUINDANAO DEL SUR'
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
# DYNASTY (2019 proxy — unchanged)
# ---------------------------------------------------------------------------

def process_dynasty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    df_dyn = pd.read_csv(csv_path)
    df_dyn = df_dyn[df_dyn['Year'] == DYNASTY_YEAR].copy()
    df_dyn['Province'] = standardize_province_names(df_dyn['Province'])

    dyn_cols = ['HHI', 'GINI']
    for col in dyn_cols:
        df_dyn[col] = pd.to_numeric(df_dyn[col], errors='coerce')

    # Maguindanao split
    mag_base = df_dyn[df_dyn['Province'] == 'MAGUINDANAO']
    for sub in ['MAGUINDANAO DEL NORTE', 'MAGUINDANAO DEL SUR']:
        if not mag_base.empty and sub not in df_dyn['Province'].values:
            extra = mag_base.copy()
            extra['Province'] = sub
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
                              dynasty_csv_path: Path = None,
                              flemms_xlsx_path: Path = None) -> pd.DataFrame:
    """Left-join poverty (2023), NTA dependency (2025), FLEMMS literacy
    (2024), ethnicity, and dynasty (2019 proxy) onto the feature table."""

    merged = df_features.copy()
    merged['Province'] = merged['Province'].str.strip().str.upper()
    feature_provinces = merged['Province'].unique()

    # -- Ethnicity --
    eth_map = load_demographic_ethnicity_map(
        ethnicity_csv_path, feature_provinces=feature_provinces
    )
    merged['Dominant_Dialect'] = merged['Province'].map(eth_map).fillna('N/A')

    impute_cols: list[str] = []

    # -- Poverty (2023 primary / 2021 fallback) --
    if pov_csv_path:
        print("\n-> Integrating Poverty Data "
              "(2023 primary; 2021 fallback for Maguindanao splits)...")
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

    # -- NTA Dependency (2025 XLSX) --
    if ari_xlsx_path:
        print("\n-> Integrating LGU NTA Dependency Data (FY2025)...")
        df_nta, nta_cols = process_nta_data(ari_xlsx_path)
        df_nta['Province'] = df_nta['Province'].str.strip().str.upper()

        missing_nta = set(feature_provinces) - set(df_nta['Province'].unique())
        print("--- NTA DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_nta:
            print(f"[!] Provinces missing in NTA dataset: {sorted(missing_nta)}")
        else:
            print("[\u2713] Perfect join found for NTA Dependency Data.")

        merged = pd.merge(merged, df_nta, on='Province', how='left')
        impute_cols.extend(nta_cols)

    # -- FLEMMS 2024 Functional Literacy --
    if flemms_xlsx_path:
        print("\n-> Integrating FLEMMS 2024 Functional Literacy Data...")
        df_fl, fl_cols = process_flemms_data(
            flemms_xlsx_path, feature_provinces=feature_provinces
        )
        df_fl['Province'] = df_fl['Province'].str.strip().str.upper()

        missing_fl = set(feature_provinces) - set(df_fl['Province'].unique())
        print("--- FLEMMS DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_fl:
            print(f"[!] Provinces missing in FLEMMS dataset: {sorted(missing_fl)}")
        else:
            print("[\u2713] Perfect join found for FLEMMS Data.")

        merged = pd.merge(merged, df_fl, on='Province', how='left')
        impute_cols.extend(fl_cols)

    # -- Dynasty (2019 proxy) --
    if dynasty_csv_path:
        print(f"\n-> Integrating Dynasty & Inequality Data "
              f"({DYNASTY_YEAR} proxy for 2025)...")
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
