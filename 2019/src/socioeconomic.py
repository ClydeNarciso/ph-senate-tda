"""Socioeconomic data helpers: province-name standardization and loading /
merging of poverty, IRA dependency, ethnicity, and dynasty datasets."""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ETHNICITY_DEPENDENCY_FILE,
    ARI_DEPENDENCY_FILE,
)


def standardize_province_names(series: pd.Series) -> pd.Series:
    """Normalise province name strings to a canonical upper-case form."""
    clean = (
        series.astype(str)
        .str.replace(r'\s+[a-z0-9]{1,2}[/,].*', '', regex=True)
        .str.upper()
        .str.strip()
    )
    mapping = {
        '1ST DISTRICT': 'NATIONAL CAPITAL REGION - MANILA',
        '2ND DISTRICT': 'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        '3RD DISTRICT': 'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        '4TH DISTRICT': 'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'NCR, SECOND DISTRICT': 'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'NCR, THIRD DISTRICT': 'NATIONAL CAPITAL REGION - THIRD DISTRICT',
        'NCR, FOURTH DISTRICT': 'NATIONAL CAPITAL REGION - FOURTH DISTRICT',
        'CITY OF MANILA': 'NATIONAL CAPITAL REGION - MANILA',
        'NCR, CITY OF MANILA, FIRST DISTRICT': 'NATIONAL CAPITAL REGION - MANILA',
        'MT. PROVINCE': 'MOUNTAIN PROVINCE',
        'DINAGAT ISLAND': 'DINAGAT ISLANDS',
        'DAVAO DEL NORTE': 'DAVAO (DAVAO DEL NORTE)',
        'NORTH COTABATO': 'COTABATO (NORTH COT.)',
        'SARANGGANI': 'SARANGANI',
        'COTABATO': 'COTABATO (NORTH COT.)',
        'SAMAR': 'SAMAR (WESTERN SAMAR)',
        'DAVAO DE ORO': 'COMPOSTELA VALLEY',
    }
    return clean.replace(mapping)


def process_poverty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    df_pov = (
        pd.read_csv(csv_path, sep=';', skiprows=2, encoding='latin1')
        .replace('..', np.nan)
        .dropna(subset=['Province'])
    )
    df_pov['Province'] = standardize_province_names(df_pov['Province'])

    pov_cols = [c for c in df_pov.columns if 'Poverty Incidence' in c]
    for col in pov_cols:
        df_pov[col] = pd.to_numeric(df_pov[col], errors='coerce')

    # Replicate 4th-district row under the Taguig-Pateros label
    ncr_4th = df_pov[df_pov['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_pov = pd.concat([df_pov, extra], ignore_index=True)

    return df_pov, pov_cols


def process_ira_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    df_ira = pd.read_csv(csv_path, header=6, encoding='latin1')
    df_ira = df_ira.dropna(subset=['PROVINCE', 'LGU NAME'])
    df_ira = df_ira[df_ira['PROVINCE'] != 'PROVINCE']

    df_ira['PROVINCE'] = df_ira['PROVINCE'].astype(str).str.upper().str.strip()
    df_ira['LGU NAME'] = df_ira['LGU NAME'].astype(str).str.upper().str.strip()
    df_ira['Province'] = standardize_province_names(df_ira['PROVINCE'])

    df_ira['IRA DEPENDENCY'] = (
        df_ira['IRA DEPENDENCY']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df_ira['IRA DEPENDENCY'] = pd.to_numeric(df_ira['IRA DEPENDENCY'], errors='coerce')

    ncr_city_router = {
        'MANILA':       'NATIONAL CAPITAL REGION - MANILA',
        'MANDALUYONG':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'MARIKINA':     'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'PASIG':        'NATIONAL CAPITAL REGION - SECOND DISTRICT',
        'QUEZON CITY':  'NATIONAL CAPITAL REGION - SECOND DISTRICT',
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
    for city, district in ncr_city_router.items():
        mask = df_ira['LGU NAME'].str.contains(city, case=False, na=False)
        df_ira.loc[mask, 'Province'] = district

    df_ira_prov = (
        df_ira.groupby('Province')['IRA DEPENDENCY']
        .mean()
        .reset_index()
        .rename(columns={'IRA DEPENDENCY': 'Mean_IRA_Dependency'})
    )

    # Ensure FOURTH DISTRICT entry exists alongside TAGUIG-PATEROS
    ncr_4th = df_ira_prov[df_ira_prov['Province'] == 'TAGUIG - PATEROS']
    if (not ncr_4th.empty
            and 'NATIONAL CAPITAL REGION - FOURTH DISTRICT'
            not in df_ira_prov['Province'].values):
        extra = ncr_4th.copy()
        extra['Province'] = 'NATIONAL CAPITAL REGION - FOURTH DISTRICT'
        df_ira_prov = pd.concat([df_ira_prov, extra], ignore_index=True)

    return df_ira_prov, ['Mean_IRA_Dependency']


def load_demographic_ethnicity_map(csv_path: Path, feature_provinces=None) -> dict:
    df_eth = pd.read_csv(csv_path)
    df_eth['Province'] = df_eth['Province'].astype(str).str.strip().str.upper()

    manual_map = {
        'COTABATO (NORTH COTABATO) 1':              'COTABATO (NORTH COT.)',
        'DAVAO DE ORO (COMPOSTELA VALLEY)':         'COMPOSTELA VALLEY',
        'MAGUINDANAO (INCLUDING THE CITY OF COTABATO)': 'MAGUINDANAO',
        'DAVAO DEL NORTE':                          'DAVAO (DAVAO DEL NORTE)',
    }
    df_eth['Province'] = df_eth['Province'].replace(manual_map)

    ncr_4th = df_eth[df_eth['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_eth = pd.concat([df_eth, extra], ignore_index=True)

    if feature_provinces is not None:
        missing = set(feature_provinces) - set(df_eth['Province'].unique())
        print("\n--- ETHNICITY SOURCE PROVINCE ALIGNMENT CHECK ---")
        if missing:
            print(f"[!] {len(missing)} provinces unmatched in Ethnicity dataset: {sorted(missing)}")
        else:
            print("[\u2713] All primary model provinces aligned with Ethnicity records.")

    return dict(zip(df_eth['Province'], df_eth['Dominant_Ethnicity']))


def process_dynasty_data(csv_path: Path) -> tuple[pd.DataFrame, list]:
    df_dyn = pd.read_csv(csv_path)

    # Filter for 2019 data
    df_dyn = df_dyn[df_dyn['Year'].astype(str) == '2019'].copy()
    df_dyn['Province'] = standardize_province_names(df_dyn['Province'])

    dyn_cols = ['HHI', 'GINI']
    for col in dyn_cols:
        df_dyn[col] = pd.to_numeric(df_dyn[col], errors='coerce')

    ncr_4th = df_dyn[df_dyn['Province'] == 'NATIONAL CAPITAL REGION - FOURTH DISTRICT']
    if not ncr_4th.empty:
        extra = ncr_4th.copy()
        extra['Province'] = 'TAGUIG - PATEROS'
        df_dyn = pd.concat([df_dyn, extra], ignore_index=True)

    # Drop duplicate rows for the same province to prevent entry multiplication during merge
    df_dyn = df_dyn.drop_duplicates(subset=['Province']).copy()

    # Region intentionally excluded from the selection slice
    return df_dyn[['Province'] + dyn_cols], dyn_cols


def merge_socioeconomic_data(df_features: pd.DataFrame,
                              pov_csv_path: Path = None,
                              ethnicity_csv_path: Path = ETHNICITY_DEPENDENCY_FILE,
                              ari_csv_path: Path = ARI_DEPENDENCY_FILE,
                              dynasty_csv_path: Path = None) -> pd.DataFrame:
    """Left-join poverty, IRA dependency, ethnicity, and dynasty onto the feature table."""

    merged = df_features.copy()
    merged['Province'] = merged['Province'].str.strip().str.upper()
    feature_provinces = merged['Province'].unique()

    eth_map = load_demographic_ethnicity_map(
        ethnicity_csv_path, feature_provinces=feature_provinces
    )
    merged['Dominant_Dialect'] = merged['Province'].map(eth_map).fillna('N/A')

    impute_cols: list[str] = []

    if pov_csv_path:
        print("\n-> Integrating Poverty Data...")
        df_pov, pov_cols = process_poverty_data(pov_csv_path)
        df_pov['Province'] = df_pov['Province'].str.strip().str.upper()

        missing_pov = set(feature_provinces) - set(df_pov['Province'].unique())
        print("--- POVERTY DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_pov:
            print(f"[!] Provinces missing in Poverty dataset: {sorted(missing_pov)}")
        else:
            print("[\u2713] Perfect join found for Poverty Data.")

        merged = pd.merge(merged, df_pov, on='Province', how='left')
        impute_cols.extend(pov_cols)

    if ari_csv_path:
        print("\n-> Integrating LGU IRA Dependency Data...")
        df_ira, ira_cols = process_ira_data(ari_csv_path)
        df_ira['Province'] = df_ira['Province'].str.strip().str.upper()

        missing_ira = set(feature_provinces) - set(df_ira['Province'].unique())
        print("--- IRA DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_ira:
            print(f"[!] Provinces missing in IRA dataset: {sorted(missing_ira)}")
        else:
            print("[\u2713] Perfect join found for IRA Dependency Data.")

        merged = pd.merge(merged, df_ira, on='Province', how='left')
        impute_cols.extend(ira_cols)

    if dynasty_csv_path:
        print("\n-> Integrating Dynasty & Inequality Data (2019)...")
        df_dyn, dyn_cols = process_dynasty_data(dynasty_csv_path)
        df_dyn['Province'] = df_dyn['Province'].str.strip().str.upper()

        missing_dyn = set(feature_provinces) - set(df_dyn['Province'].unique())
        print("--- DYNASTY DATA PROVINCE ALIGNMENT CHECK ---")
        if missing_dyn:
            print(f"[!] Provinces missing in Dynasty dataset: {sorted(missing_dyn)}")
        else:
            print("[\u2713] Perfect join found for Dynasty Data.")

        # Drop the Region column from dynasty data if it exists to prevent suffix overlapping
        if 'Region' in df_dyn.columns:
            df_dyn = df_dyn.drop(columns=['Region'])

        merged = pd.merge(merged, df_dyn, on='Province', how='left')

        # -------------------------------------------------------------
        # REGIONAL IMPUTATION FOR MISSING DYNASTY/GINI VALUES
        # -------------------------------------------------------------
        print("-> Imputing missing Dynasty (HHI/GINI) values using local Regional Means...")
        for col in dyn_cols:
            if merged[col].isna().any():
                merged[col] = merged.groupby('Region')[col].transform(lambda x: x.fillna(x.mean()))
                if merged[col].isna().any():
                    # Fallback to national mean if entire region is missing
                    merged[col] = merged[col].fillna(merged[col].mean())

        impute_cols.extend(dyn_cols)

    for col in impute_cols:
        if col in merged.columns:
            n_missing = merged[col].isna().sum()
            if n_missing > 0:
                missing_provs = merged.loc[merged[col].isna(), 'Province'].tolist()
                print(f"   [!] {n_missing} missing values remaining for '{col}': {missing_provs}")

    print("\n--- SOCIOECONOMIC DATA INTEGRATION COMPLETE ---")
    return merged
