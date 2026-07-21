"""Precinct-level cleaning for the 2025 election data.

Changes vs 2022
---------------
- OAV pseudo-province set updated: 'NORTH AND LATIN AMERICAS' replaces
  'AMERICAS'.
- Two additional non-geographic province labels must be dropped:
  'LAV'  (single-precinct artefact row under Region LAV)
  'SPECIAL GEOGRAPHIC AREA' (BARMM zone with only 59 precincts — too
  small for meaningful TDA at province level).
- All other logic is identical to the 2022 pipeline.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    CLEANED_DATA_PATH, TABLES_DIR,
    OAV_PSEUDO_PROVINCES, NON_PROVINCE_LABELS,
)


def preprocess_and_clean_data(df_raw: pd.DataFrame,
                               clr_cols: list,
                               contamination: float = 0.01,
                               exclude_oav: bool = True) -> pd.DataFrame:
    """Clean raw 2025 precinct data:
    - Remove OAV rows (Region == 'OAV') and overseas pseudo-provinces.
    - Remove non-province artefact rows (LAV, SPECIAL GEOGRAPHIC AREA).
    - Compute per-row vote proportions.
    - Apply per-province Isolation Forest outlier removal.
    """
    vote_cols = [
        'Admin_votes', 'Opposition_votes',
        'Independent_votes', 'Non-Aligned_votes',
    ]

    if CLEANED_DATA_PATH.exists():
        df_cached = pd.read_csv(CLEANED_DATA_PATH)
        if 'Admin_prop' in df_cached.columns:
            print("-> Loading valid cached Cleaned Data...")
            return df_cached
        print("-> Cache outdated. Recalculating...")

    print("--- STARTING PREPROCESSING ---")
    df = df_raw.copy()
    initial_len = len(df)

    if exclude_oav:
        mask_region = df['Region'].astype(str).str.contains(
            'OAV|OVERSEAS', case=False, na=False
        )
        mask_pseudo = df['Province'].astype(str).str.upper().str.strip().isin(
            OAV_PSEUDO_PROVINCES
        )
        df = df[~(mask_region | mask_pseudo)].copy()
        print(f"Excluded {initial_len - len(df)} OAV/Overseas precinct rows "
              f"({initial_len} -> {len(df)}).")

    # Drop non-province artefact rows unique to the 2025 data
    mask_non_prov = df['Province'].astype(str).str.upper().str.strip().isin(
        NON_PROVINCE_LABELS
    )
    n_non_prov = mask_non_prov.sum()
    if n_non_prov > 0:
        dropped_labels = df.loc[mask_non_prov, 'Province'].unique().tolist()
        df = df[~mask_non_prov].copy()
        print(f"Excluded {n_non_prov} non-province rows "
              f"({dropped_labels}) -> {len(df)} rows remaining.")

    if all(col in df.columns for col in vote_cols):
        df['Total_Votes'] = df[vote_cols].sum(axis=1).replace(0, 1e-9)
        df['Admin_prop'] = df['Admin_votes'] / df['Total_Votes']

    print(f"--- STARTING LOCAL OUTLIER REMOVAL "
          f"({contamination * 100:.1f}% per province) ---")
    df_clean_features = df.dropna(subset=clr_cols).copy()
    df_clean_features['Is_Outlier'] = False

    for province, group in df_clean_features.groupby('Province', observed=True):
        if len(group) > 50:
            iso = IsolationForest(
                contamination=contamination, random_state=42, n_jobs=-1
            )
            labels = iso.fit_predict(group[clr_cols])
            df_clean_features.loc[group.index, 'Is_Outlier'] = (labels == -1)

    report_data = [
        {
            'Province': province,
            'Original Points': len(group),
            'Points Removed': group['Is_Outlier'].sum(),
            'Points Remaining': len(group) - group['Is_Outlier'].sum(),
            '% Removed': (group['Is_Outlier'].sum() / len(group) * 100
                          if len(group) > 0 else 0.0),
        }
        for province, group in df_clean_features.groupby('Province', observed=True)
    ]

    df_report = (
        pd.DataFrame(report_data)
        .sort_values('% Removed', ascending=False)
        .reset_index(drop=True)
    )
    print("\n--- OUTLIER REMOVAL REPORT (TOP 5) ---")
    try:
        from IPython.display import display
        display(
            df_report.head(5)
            .style.format({'% Removed': '{:.2f}%'})
            .background_gradient(cmap='Reds', subset=['% Removed'])
        )
    except Exception:
        print(df_report.head(5).to_string())

    df_report.to_csv(TABLES_DIR / '2025_outlier_report.csv', index=False)
    with open(TABLES_DIR / '2025_outlier_report.tex', 'w') as f:
        f.write(
            df_report.style
            .format({'% Removed': '{:.2f}\\%'})
            .hide(axis="index")
            .to_latex(convert_css=True, hrules=True)
        )

    df_cleaned = (
        df_clean_features[~df_clean_features['Is_Outlier']]
        .drop(columns=['Is_Outlier'])
        .copy()
    )
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"-> Saved cleaned data to {CLEANED_DATA_PATH.name}")
    return df_cleaned
