"""TDA feature extraction for the 2022 pipeline.

Identical logic to the 2019 pipeline; only cache filenames differ
(2022_h0_intervals.csv, 2022_provincial_topological_features.csv).
"""
import numpy as np
import pandas as pd
import gudhi as gd
from gudhi.representations import Landscape
from persim.persistent_entropy import persistent_entropy

from config import INTERVALS_CACHE_PATH, TOPOLOGY_RESULTS_PATH


def compute_and_save_intervals(df_clean: pd.DataFrame,
                                cols: list) -> dict:
    """Compute H0 persistence intervals via Vietoris-Rips per province."""

    if INTERVALS_CACHE_PATH.exists():
        print("-> Loading cached H0 Intervals...")
        df_cache = pd.read_csv(INTERVALS_CACHE_PATH)
        intervals_dict: dict[str, np.ndarray] = {
            province: grp[['Birth', 'Death']].values
            for province, grp in df_cache.groupby('Province', sort=False)
        }
        return intervals_dict

    print("--- COMPUTING H0 INTERVALS (VIETORIS-RIPS) ---")
    intervals_dict = {}

    for province, df_prov in df_clean.groupby('Province', observed=True):
        X = df_prov[cols].values
        if len(X) > 0:
            rc = gd.RipsComplex(points=X)
            st = rc.create_simplex_tree(max_dimension=1)
            st.persistence(homology_coeff_field=2, persistence_dim_max=True)
            h0_diag = np.array(st.persistence_intervals_in_dimension(0))
            finite_h0 = h0_diag[np.isfinite(h0_diag[:, 1])]
            if len(finite_h0) > 0:
                intervals_dict[province] = finite_h0

    rows = [
        {'Province': province, 'Birth': float(b), 'Death': float(d)}
        for province, arr in intervals_dict.items()
        for b, d in arr
    ]
    pd.DataFrame(rows).to_csv(INTERVALS_CACHE_PATH, index=False)
    return intervals_dict


def extract_provincial_features_4d(df_clean: pd.DataFrame,
                                    intervals_dict: dict) -> pd.DataFrame:
    """Summarise each province's H0 diagram into scalar TDA features."""

    if TOPOLOGY_RESULTS_PATH.exists():
        df_features = pd.read_csv(TOPOLOGY_RESULTS_PATH)
        if 'H0_Total_Persistence' in df_features.columns:
            print("-> Loading cached 4D Topological Features...")
            return df_features

    print("--- CALCULATING 4D TOPOLOGICAL SUMMARIES ---")
    provinces = list(intervals_dict.keys())
    all_finite_h0_diagrams = list(intervals_dict.values())

    landscape_transformer = Landscape(num_landscapes=5, resolution=200)
    global_landscapes = landscape_transformer.fit_transform(all_finite_h0_diagrams)

    rips_stats = []
    for idx, province in enumerate(provinces):
        finite_h0 = all_finite_h0_diagrams[idx]
        df_prov = df_clean[df_clean['Province'] == province]

        rips_stats.append({
            'Province': province,
            'Region': df_prov['Region'].iloc[0],
            'Mean_Admin_Prop': df_prov['Admin_prop'].mean(),
            'H0_PersistentEntropy': persistent_entropy(finite_h0, normalize=True)[0],
            'H0_L2_Norm': np.linalg.norm(global_landscapes[idx]),
            'H0_Total_Persistence': float(np.sum(finite_h0[:, 1] - finite_h0[:, 0])),
        })

    df_features = pd.DataFrame(rips_stats)
    df_features.to_csv(TOPOLOGY_RESULTS_PATH, index=False)
    return df_features
