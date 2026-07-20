"""Precinct-level cleaning: OAV removal, vote-proportion calculation, and
per-province outlier removal via Isolation Forest."""
import pandas as pd
from IPython.display import display
from sklearn.ensemble import IsolationForest

from config import CLEANED_DATA_PATH, TABLES_DIR


def preprocess_and_clean_data(df_raw: pd.DataFrame,
                               clr_cols: list,
                               contamination: float = 0.01,
                               exclude_oav: bool = True) -> pd.DataFrame:
    """Clean raw precinct data: remove OAV rows, compute proportions,
    and flag / remove per-province outliers via Isolation Forest."""

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

    if exclude_oav:
        initial_len = len(df)
        df = df[
            ~df['Region'].astype(str).str.contains(
                'OAV|OVERSEAS', case=False, na=False
            )
        ].copy()
        print(f"Excluded {initial_len - len(df)} OAV/Overseas precinct rows.")

    if all(col in df.columns for col in vote_cols):
        df['Total_Votes'] = df[vote_cols].sum(axis=1).replace(0, 1e-9)
        df['Admin_prop'] = df['Admin_votes'] / df['Total_Votes']

    print(f"--- STARTING LOCAL OUTLIER REMOVAL ({contamination * 100}% per province) ---")
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
    display(
        df_report.head(5)
        .style.format({'% Removed': '{:.2f}%'})
        .background_gradient(cmap='Reds', subset=['% Removed'])
    )

    df_report.to_csv(TABLES_DIR / '2019_outlier_report.csv', index=False)
    with open(TABLES_DIR / '2019_outlier_report.tex', 'w') as f:
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
