"""
main.py — Master execution controller for the ZZPH pipeline.

Execution stages
----------------
1. PREPROCESS   Load the four raw CSVs, harmonise province names, drop OAV
                rows, and apply per-province Isolation Forest outlier removal.
                Results are cached in results/1_preprocessed_data/.

2. RADII        Compute per-province optimal Vietoris-Rips radii using the
                k-NN distance percentile method.
                Results are cached in results/2_knn_radius_optimization/.

3. ZIGZAG       Run the seven-step zigzag filtration province-by-province in
                parallel (joblib) and extract three H0 scalar features.
                Results are cached in results/3_zigzag_results/.

4. SOCIOECONOMIC Merge poverty (2015-2023 avg), literacy (FLEMMS 2024), and
                IRA/NTA dependency (2016-2025 avg) onto the feature table.
                Append population-weighted mean admin vote share.

5. DIAGNOSTICS  Feature correlation heatmap and epsilon-robustness plots.

6. BALLMAPPER   Build the BallMapper graph (ε = BM_EPSILON) and write all
                node tables, per-covariate compact tables, intersection matrix,
                network graphs, and scatter plots.
                Results are saved to results/4_ballmapper_clusters/.

7. COMPARATIVE  Per-covariate comparative epsilon-robustness (Poverty,
                Literacy, IRA Dependency).
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from preprocessing import (
    process_dataframe,
    remove_outliers_provincial,
    log_province_coverage,
)
from topology import build_radius_cache, compute_zigzag_features
from socioeconomic import load_and_merge_socioeconomic, append_admin_vote_share
from ballmapper import (
    build_unified_ballmapper_structure,
    generate_correlation_heatmap,
    generate_topology_scatter_plots,
    visualize_ballmapper_coloring,
)
from robustness import run_epsilon_robustness, run_epsilon_comparative_robustness

warnings.filterwarnings("ignore")


def main() -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # 1. PREPROCESS
    # -----------------------------------------------------------------------
    if all(f.exists() for f in config.EXPECTED_PREP_FILES):
        print("-> Loading cached preprocessed data…")
        df_16, df_19 = (
            pd.read_csv(config.FILE_16_CLEAN),
            pd.read_csv(config.FILE_19_CLEAN),
        )
        df_22, df_25 = (
            pd.read_csv(config.FILE_22_CLEAN),
            pd.read_csv(config.FILE_25_CLEAN),
        )
    else:
        print("Preprocessing raw datasets…")
        raw_dfs = [
            process_dataframe(pd.read_csv(config.FILE_16_RAW)),
            process_dataframe(pd.read_csv(config.FILE_19_RAW)),
            process_dataframe(pd.read_csv(config.FILE_22_RAW)),
            process_dataframe(pd.read_csv(config.FILE_25_RAW)),
        ]
        cleaned_dfs, _ = remove_outliers_provincial(
            raw_dfs,
            feature_columns=config.FEATURE_COLS,
            contamination=config.OUTLIER_CONTAMINATION,
            min_points=config.OUTLIER_MIN_POINTS,
            output_dir=config.PREP_DIR,
        )
        df_16, df_19, df_22, df_25 = cleaned_dfs
        for df, path in zip(cleaned_dfs, config.EXPECTED_PREP_FILES):
            df.to_csv(path, index=False)

    # -----------------------------------------------------------------------
    # 2. RADII + 3. ZIGZAG FEATURES
    # -----------------------------------------------------------------------
    common_provinces = sorted(
        set(df_16["Province"])
        & set(df_19["Province"])
        & set(df_22["Province"])
        & set(df_25["Province"])
    )
    print(f"\nCommon provinces across all 4 elections: {len(common_provinces)}")

    radii   = build_radius_cache(common_provinces, df_16, df_19, df_22, df_25)
    df_topo = compute_zigzag_features(common_provinces, radii, df_16, df_19, df_22, df_25)

    # -----------------------------------------------------------------------
    # 4. SOCIOECONOMIC MERGE + ADMIN VOTE SHARE
    # -----------------------------------------------------------------------
    df_master = load_and_merge_socioeconomic(df_topo)
    df_master = append_admin_vote_share(df_master, df_16, df_19, df_22, df_25)

    if df_master.empty:
        raise ValueError(
            "df_master is empty after the socioeconomic merge. "
            "Check province name standardisation and file paths."
        )

    # Standardise topological features
    topo_cols = ["H0_L2_Norm", "H0_PersistentEntropy", "H0_Total_Persistence"]
    X_scaled  = StandardScaler().fit_transform(df_master[topo_cols].values)
    df_master[[f"{c}_scaled" for c in topo_cols]] = X_scaled

    # Province-drop audit
    log_province_coverage(df_16, df_19, df_22, df_25, df_master)

    # -----------------------------------------------------------------------
    # 5. DIAGNOSTICS
    # -----------------------------------------------------------------------
    generate_correlation_heatmap(df_master)
    run_epsilon_robustness(df_master)

    # -----------------------------------------------------------------------
    # 6. BALLMAPPER
    # -----------------------------------------------------------------------
    bm_obj, df_summary = build_unified_ballmapper_structure(
        df_master, X_scaled, eps=config.BM_EPSILON, save_dir=config.BALLMAPPER_DIR
    )

    for mode, cfg in config.COLOR_CONFIGS.items():
        raw_cmap = cfg["cmap"]
        mpl_cmap = raw_cmap["mpl"] if isinstance(raw_cmap, dict) else raw_cmap

        visualize_ballmapper_coloring(
            bm=bm_obj,
            df_features=df_master,
            color_col=cfg["col"],
            cmap=mpl_cmap,
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            cbar_label=cfg["label"],
            cbar_format=cfg["format"],
            mode_name=mode.lower(),
            eps=config.BM_EPSILON,
            save_dir=config.BALLMAPPER_DIR,
        )

        generate_topology_scatter_plots(
            df_summary=df_summary,
            color_col=cfg["summary_name"],
            cmap=raw_cmap,
            mode_name=mode.lower(),
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            save_dir=config.BALLMAPPER_DIR,
        )

    # -----------------------------------------------------------------------
    # 7. COMPARATIVE ROBUSTNESS
    # -----------------------------------------------------------------------
    for mode, cfg in config.COLOR_CONFIGS.items():
        if cfg["robustness"] is None:
            continue
        rb = cfg["robustness"]
        run_epsilon_comparative_robustness(
            df_master=df_master,
            target_col=cfg["col"],
            vulnerable_label=rb["vulnerable_label"],
            baseline_label=rb["baseline_label"],
            metric_name=rb["metric_name"],
            cmap_name=cfg["cmap"] if isinstance(cfg["cmap"], str) else "viridis",
            invert_colors=rb["invert_colors"],
            n_reps=config.NUM_REPETITIONS,
            chosen_eps=config.BM_EPSILON,
            cache_dir=config.ROBUSTNESS_DIR,
        )

    print("\n" + "=" * 60)
    print("ZZPH PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Figures & tables: {config.BALLMAPPER_DIR}")
    print(f"  Zigzag features:  {config.FINAL_FEATURES_FILE}")

    return df_master


if __name__ == "__main__":
    main()
