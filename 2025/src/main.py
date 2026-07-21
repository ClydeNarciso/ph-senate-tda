"""
Main execution script for the 2025 electoral-topology / socioeconomic
analysis pipeline.

Data notes
----------
- Poverty    : 2023 (closest PSA survey year to 2025 election).
               Maguindanao del Norte/Sur gaps filled from 2021 data.
- NTA dep.   : FY2025 XLSX (actual 2025 data; column renamed from
               IRA DEPENDENCY to NTA DEPENDENCY).
- FLEMMS     : 2024 functional literacy rate by province (new dimension).
- Dynasty    : 2019 proxy (no 2022/2025 data exists in the dataset).
- Ethnicity  : 2020 PSA census.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from preprocessing import preprocess_and_clean_data
from topology import compute_and_save_intervals, extract_provincial_features_4d
from socioeconomic import merge_socioeconomic_data
from ballmapper import run_and_visualize_ballmapper, print_intersection_matrix_once
from visualization import generate_correlation_heatmap, generate_topology_relationship_plots
from reporting import display_clean_summary, export_summary_to_latex
from robustness import (
    run_epsilon_robustness,
    run_stepwise_stability,
    run_epsilon_comparative_robustness,
)

warnings.filterwarnings('ignore')


def main():
    # -------------------------------------------------------------------
    # 1. Pipeline: clean data -> H0 intervals -> 4D features
    # -------------------------------------------------------------------
    df_raw = pd.read_csv(config.DATA_FILE)
    df_clean = preprocess_and_clean_data(df_raw, config.CLR_COLS)
    intervals_dict = compute_and_save_intervals(df_clean, config.CLR_COLS)
    df_features = extract_provincial_features_4d(df_clean, intervals_dict)

    # Standardize topological features
    feature_cols = ['H0_L2_Norm', 'H0_PersistentEntropy', 'H0_Total_Persistence']
    X_scaled = StandardScaler().fit_transform(df_features[feature_cols].values)
    df_features[[f"{col}_scaled" for col in feature_cols]] = X_scaled

    # Merge all socioeconomic data
    df_features = merge_socioeconomic_data(
        df_features,
        pov_csv_path=config.POVERTY_DATA_FILE,
        ethnicity_csv_path=config.ETHNICITY_DEPENDENCY_FILE,
        ari_xlsx_path=config.ARI_DEPENDENCY_FILE,
        dynasty_csv_path=config.DYNASTY_DATA_FILE,
        flemms_xlsx_path=config.FLEMMS_FILE,
    )

    # -------------------------------------------------------------------
    # 2. Initial diagnostics
    # -------------------------------------------------------------------
    generate_correlation_heatmap(df_features)
    run_epsilon_robustness(df_features)
    if config.RUN_STEPWISE_STABILITY:
        run_stepwise_stability(intervals_dict)
    else:
        print("\n-> Skipping step-wise stability "
              "(RUN_STEPWISE_STABILITY=False in config)")

    # -------------------------------------------------------------------
    # 3. Covariate coloring configurations
    # -------------------------------------------------------------------
    pov_label    = config.POVERTY_COL_LABEL          # "2023 Poverty Incidence"
    flemms_label = config.FLEMMS_COL_LABEL            # "Functional_Literacy_Rate"

    color_configs = {
        'Admin_Share': {
            'col': 'Mean_Admin_Prop',
            'cmap': config.ADMIN_CMAP,
            'vmin': 0.0, 'vmax': 1.0,
            'label': 'Average Admin Vote Share',
            'format': lambda val, loc: f'{val*100:.0f}%',
        },
        'Poverty': {
            'col': pov_label,
            'cmap': 'RdYlBu_r',
            'vmin': None, 'vmax': None,
            'label': f'Mean {pov_label}',
            'format': lambda val, loc: f'{val:.1f}%',
        },
        'Literacy': {
            'col': flemms_label,
            'cmap': 'YlGnBu',
            'vmin': None, 'vmax': None,
            'label': 'Mean Functional Literacy Rate (2024, %)',
            'format': lambda val, loc: f'{val:.1f}%',
        },
        'Ethnicity': {
            'col': 'Dominant_Dialect',
            'cmap': None,
            'vmin': None, 'vmax': None,
            'label': 'Node Majority Ethnicity',
            'format': None,
        },
        'Dynasty_HHI': {
            'col': 'HHI',
            'cmap': 'viridis',
            'vmin': None, 'vmax': None,
            'label': f'Mean {config.DYNASTY_YEAR} HHI (Dynasty Concentration)',
            'format': lambda val, loc: f'{val:.2f}',
        },
        'Inequality_GINI': {
            'col': 'GINI',
            'cmap': 'plasma',
            'vmin': None, 'vmax': None,
            'label': f'Mean {config.DYNASTY_YEAR} GINI Coefficient',
            'format': lambda val, loc: f'{val:.3f}',
        },
    }

    # -------------------------------------------------------------------
    # 4. BallMapper loop over each covariate coloring mode
    # -------------------------------------------------------------------
    target_modes = [
        'Admin_Share', 'Poverty', 'Literacy',
        'Ethnicity', 'Dynasty_HHI', 'Inequality_GINI',
    ]
    last_bm_instance = None

    for mode in target_modes:
        print(f"\n{'='*60}")
        print(f" EXECUTING PIPELINE FOR TARGET MODE: {mode}")
        print(f"{'='*60}")

        cfg = color_configs[mode]

        df_bm_summary, bm = run_and_visualize_ballmapper(
            df_features=df_features,
            X_scaled=X_scaled,
            eps=config.BM_EPSILON,
            save_dir=config.FIGURES_DIR,
            color_col=cfg['col'],
            cmap=cfg['cmap'],
            vmin=cfg['vmin'],
            vmax=cfg['vmax'],
            cbar_label=cfg['label'],
            cbar_format=cfg['format'],
            mode_name=mode,
        )
        last_bm_instance = bm

        summary_csv_path = config.TABLES_DIR / f'summary_table_{mode}.csv'
        df_bm_summary.to_csv(summary_csv_path, index=False)
        print(f"-> Saved Summary Table to: {summary_csv_path.name}")

        display_clean_summary(df_bm_summary, cfg['col'])
        export_summary_to_latex(
            df_bm_summary,
            config.TABLES_DIR / f"summary_table_{mode}.tex",
            cfg['col'],
        )

        generate_topology_relationship_plots(
            df_bm_summary=df_bm_summary,
            color_col=cfg['col'],
            cmap=cfg['cmap'],
            mode_name=mode,
        )

    # -------------------------------------------------------------------
    # 5. Global post-loop exports
    # -------------------------------------------------------------------
    print(f"\n{'='*60}\n COMPILING COMPREHENSIVE OVERALL EXPORTS\n{'='*60}")

    if last_bm_instance is not None:
        print_intersection_matrix_once(
            bm=last_bm_instance,
            df_features=df_features,
            save_dir=config.TABLES_DIR,
            suffix="_master",
        )

        provincial_topo_path = (
            config.TABLES_DIR / 'provincial_topological_summaries_all.csv'
        )
        topo_export_cols = [
            'Province', 'Region',
            'H0_PersistentEntropy', 'H0_L2_Norm', 'H0_Total_Persistence',
            'H0_PersistentEntropy_scaled', 'H0_L2_Norm_scaled',
            'H0_Total_Persistence_scaled',
        ]
        valid_topo_cols = [c for c in topo_export_cols if c in df_features.columns]
        df_features[valid_topo_cols].to_csv(provincial_topo_path, index=False)
        print(f"-> Saved Provincial Topological Summaries to: "
              f"{provincial_topo_path.name}")

        master_nodes_data = []
        nodes = sorted(last_bm_instance.Graph.nodes)

        for node in nodes:
            pts = last_bm_instance.points_covered_by_landmarks[node]
            df_pts = df_features.iloc[pts]

            mode_dialect = (
                df_pts['Dominant_Dialect'].mode().iloc[0]
                if not df_pts.empty else 'N/A'
            )
            total_elements = len(df_pts)
            maj_count = (
                int(df_pts['Dominant_Dialect'].value_counts().iloc[0])
                if total_elements > 0 else 0
            )

            master_nodes_data.append({
                'Node ID': node,
                'No. Provinces': total_elements,
                'Provinces Included': ', '.join(df_pts['Province'].tolist()),
                'Provinces (Region)': ', '.join(
                    f"{r['Province']} ({r['Region']})"
                    for _, r in df_pts.iterrows()
                ),
                'No. of Provinces in each Region': ', '.join(
                    f"{reg} ({cnt})"
                    for reg, cnt in df_pts['Region'].value_counts().items()
                ),
                'Mean Entropy (Raw)': df_pts['H0_PersistentEntropy'].mean(),
                'Mean L2 Norm (Raw)': df_pts['H0_L2_Norm'].mean(),
                'Mean Total Persistence (Raw)': df_pts['H0_Total_Persistence'].mean(),
                'Mean Entropy (Scaled)': df_pts['H0_PersistentEntropy_scaled'].mean(),
                'Mean L2 Norm (Scaled)': df_pts['H0_L2_Norm_scaled'].mean(),
                'Mean Total Persistence (Scaled)': df_pts['H0_Total_Persistence_scaled'].mean(),
                'Mean Admin Vote Share': df_pts['Mean_Admin_Prop'].mean(),
                f'Mean {pov_label}': (
                    df_pts[pov_label].mean()
                    if pov_label in df_pts.columns else np.nan
                ),
                f'Mean {flemms_label}': (
                    df_pts[flemms_label].mean()
                    if flemms_label in df_pts.columns else np.nan
                ),
                'Mean HHI': (
                    df_pts['HHI'].mean() if 'HHI' in df_pts.columns else np.nan
                ),
                'Mean GINI': (
                    df_pts['GINI'].mean() if 'GINI' in df_pts.columns else np.nan
                ),
                'Majority Dominant Dialect': mode_dialect,
                'Major Dialect Ratio': f"{maj_count}/{total_elements}",
                'Provinces (Dialect)': ', '.join(
                    f"{r['Province']} ({r['Dominant_Dialect']})"
                    for _, r in df_pts.iterrows()
                ),
            })

        df_master_summary = pd.DataFrame(master_nodes_data)
        master_csv_path = config.TABLES_DIR / 'summary_table_master.csv'
        df_master_summary.to_csv(master_csv_path, index=False)
        print(f"-> Saved Comprehensive Master Summary Table to: "
              f"{master_csv_path.name}")

    # -------------------------------------------------------------------
    # 6. Comparative epsilon-robustness
    # -------------------------------------------------------------------
    run_epsilon_comparative_robustness(
        df_data=df_features,
        target_col=pov_label,
        vulnerable_label='High Poverty',
        baseline_label='Low Poverty',
        metric_name='Poverty Incidence',
        cmap_name='RdYlBu_r',
        invert_colors=False,
    )

    run_epsilon_comparative_robustness(
        df_data=df_features,
        target_col=flemms_label,
        vulnerable_label='Low Literacy',
        baseline_label='High Literacy',
        metric_name='Functional Literacy Rate',
        cmap_name='YlGnBu',
        invert_colors=True,      # low literacy = vulnerable
    )

    run_epsilon_comparative_robustness(
        df_data=df_features,
        target_col='HHI',
        vulnerable_label='Low HHI',
        baseline_label='High HHI',
        metric_name='HHI',
        cmap_name='viridis',
        invert_colors=True,
    )

    run_epsilon_comparative_robustness(
        df_data=df_features,
        target_col='GINI',
        vulnerable_label='High GINI',
        baseline_label='Low GINI',
        metric_name='Mean GINI',
        cmap_name='plasma',
        invert_colors=False,
    )

    return df_features


if __name__ == "__main__":
    df_features = main()
