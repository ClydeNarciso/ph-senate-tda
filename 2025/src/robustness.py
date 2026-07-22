"""Stability / robustness diagnostics:
- run_stepwise_stability: landscape-level and resolution convergence checks
- run_epsilon_robustness: BallMapper stability across epsilon (single population)
- run_epsilon_comparative_robustness: BallMapper stability across epsilon,
  split into "vulnerable" vs "baseline" subgroups by a covariate's median
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gudhi.representations import Landscape
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pyballmapper import BallMapper

from config import (
    CONVERGENCE_CACHE_PATH,
    ROBUSTNESS_CACHE_PATH,
    COMPARATIVE_CACHE_DIR,
    FIGURES_DIR,
    NUM_REPETITIONS,
    BM_EPSILON,
)


def run_stepwise_stability(intervals_dict: dict) -> None:
    if CONVERGENCE_CACHE_PATH.exists():
        print("\n-> Loading cached Step-wise Convergence Results...")
        df_conv = pd.read_csv(CONVERGENCE_CACHE_PATH)
        step_corr_k = df_conv['k_corr'].dropna().tolist()
        step_corr_res = df_conv['res_corr'].dropna().tolist()
    else:
        print("\n--- RUNNING STEP-WISE STABILITY ANALYSIS ---")
        diagrams = list(intervals_dict.values())
        k_values = list(range(1, 26))
        res_values = np.linspace(50, 1250, 25, dtype=int)

        print("  Computing norms across landscape levels (k)...")
        norms_k = []
        for k in tqdm(k_values, desc="Landscape levels (k)"):
            landscape = Landscape(num_landscapes=k, resolution=200)
            L_transformed = landscape.fit_transform(diagrams)
            norms_k.append([np.linalg.norm(L) for L in L_transformed])
        norms_k = np.array(norms_k)

        print("  Computing norms across resolutions...")
        norms_res = []
        for res in tqdm(res_values, desc="Resolution levels (res)"):
            landscape = Landscape(num_landscapes=5, resolution=res)
            L_transformed = landscape.fit_transform(diagrams)
            norms_res.append([np.linalg.norm(L) for L in L_transformed])
        norms_res = np.array(norms_res)

        step_corr_k = [pearsonr(norms_k[i], norms_k[i-1])[0] for i in range(1, len(k_values))]
        step_corr_res = [pearsonr(norms_res[i], norms_res[i-1])[0] for i in range(1, len(res_values))]

        max_len = max(len(step_corr_k), len(step_corr_res))
        df_conv = pd.DataFrame({
            'k_corr': step_corr_k + [np.nan]*(max_len - len(step_corr_k)),
            'res_corr': step_corr_res + [np.nan]*(max_len - len(step_corr_res))
        })
        df_conv.to_csv(CONVERGENCE_CACHE_PATH, index=False)

    k_steps = list(range(2, 26))
    res_steps = np.linspace(50, 1250, 25, dtype=int)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].plot(k_steps, step_corr_k, color="#f13f3f", marker="o", markersize=5, linewidth=1.5, label=r"Corr($k_i, k_{i-1}$)")
    axes[0].axvline(x=5, color="green", linestyle="--", alpha=0.6, label="Chosen ($k=5$)")
    axes[0].set_title("Stability across Landscape Levels", fontsize=12, pad=10)
    axes[0].set_xlabel("Landscape Levels ($k$)", fontsize=11)
    axes[0].set_ylabel("Pearson $r$", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")

    axes[1].plot(res_steps, step_corr_res, color="#46a0fb", marker="s", markersize=5, linewidth=1.5, label=r"Corr($res_i, res_{i-1}$)")
    axes[1].axvline(x=200, color="green", linestyle="--", alpha=0.6, label="Chosen ($res=200$)")
    axes[1].set_title("Stability across Resolution", fontsize=12, pad=10)
    axes[1].set_xlabel("Resolution (Bins)", fontsize=11)
    axes[1].set_ylabel("Pearson $r$", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].legend(frameon=False, fontsize=10, loc="lower right")
    import seaborn as sns
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stepwise_stability.png', dpi=150, bbox_inches='tight')
    # plt.show()


def _run_single_ballmapper(X, eps, seed):
    """Worker function to run a single random permutation of BallMapper."""
    np.random.seed(seed)
    # Shuffle data to randomize greedy landmark selection
    X_shuffled = np.random.permutation(X)

    bm = BallMapper(X=X_shuffled, eps=eps)
    G = bm.Graph
    num_balls = len(G.nodes)

    if num_balls > 0:
        sizes = [len(bm.points_covered_by_landmarks[node]) for node in G.nodes]
        avg_ball_size = np.mean(sizes)
    else:
        avg_ball_size = 0.0

    return num_balls, avg_ball_size


def run_epsilon_robustness(df_features):
    if ROBUSTNESS_CACHE_PATH.exists():
        print("\n-> Loading cached Epsilon Robustness data...")
        df_robustness = pd.read_csv(ROBUSTNESS_CACHE_PATH)
    else:
        print(f"\n--- RUNNING BALLMAPPER EPSILON ROBUSTNESS TEST ({NUM_REPETITIONS} REPS) ---")
        X_scaled = StandardScaler().fit_transform(
            df_features[['H0_L2_Norm', 'H0_PersistentEntropy', 'H0_Total_Persistence']].values
        )

        # Epsilon from 0.1 to 1.0 inclusive, in 0.05 increments
        eps_values = np.arange(0.1, 1.05, 0.05)
        robustness_metrics = []

        for eps in tqdm(eps_values, desc="Evaluating Epsilons"):
            # backend='threading' avoids the BrokenProcessPool serialization error
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(_run_single_ballmapper)(X_scaled, eps, seed)
                for seed in range(NUM_REPETITIONS)
            )

            num_balls_list, avg_sizes_list = zip(*results)

            robustness_metrics.append({
                'Epsilon': eps,
                'Num_Balls_Mean': np.mean(num_balls_list),
                'Num_Balls_CI_Lower': np.percentile(num_balls_list, 2.5),
                'Num_Balls_CI_Upper': np.percentile(num_balls_list, 97.5),
                'Avg_Ball_Size_Mean': np.mean(avg_sizes_list),
                'Avg_Ball_Size_CI_Lower': np.percentile(avg_sizes_list, 2.5),
                'Avg_Ball_Size_CI_Upper': np.percentile(avg_sizes_list, 97.5)
            })

        df_robustness = pd.DataFrame(robustness_metrics)
        df_robustness.to_csv(ROBUSTNESS_CACHE_PATH, index=False)

    chosen_eps = BM_EPSILON
    explicit_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # PLOT 1: Number of Balls
    fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    ax1.plot(df_robustness['Epsilon'], df_robustness['Num_Balls_Mean'], color='tab:blue', linewidth=2)
    ax1.plot(df_robustness['Epsilon'], df_robustness['Num_Balls_CI_Lower'], color='tab:blue', linewidth=1.0, alpha=0.7)
    ax1.plot(df_robustness['Epsilon'], df_robustness['Num_Balls_CI_Upper'], color='tab:blue', linewidth=1.0, alpha=0.7)

    ax1.axvline(x=0.1, color='black', linestyle=':', linewidth=1.2)
    ax1.axvline(x=1.0, color='black', linestyle=':', linewidth=1.2)
    ax1.set_xticks(explicit_ticks)
    ax1.tick_params(axis='x', labelfontfamily='sans-serif')

    ax1.set_xlabel('Epsilon', family='sans-serif', fontsize=12)
    ax1.set_ylabel('Number of Balls', family='sans-serif')

    if chosen_eps:
        ax1.axvline(x=chosen_eps, color='black', linestyle=':', linewidth=1.2)

    ax1.set_title(f'Ball Mapper Stability: Number of Balls ({NUM_REPETITIONS} Seeds)',
                  family='sans-serif', fontsize=13, pad=15)

    plt.savefig(FIGURES_DIR / 'ballmapper_num_balls_robustness.jpg', bbox_inches='tight')
    # plt.show()

    # PLOT 2: Average Ball Size
    fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=300)
    ax2.plot(df_robustness['Epsilon'], df_robustness['Avg_Ball_Size_Mean'], color='tab:blue', linewidth=2)
    ax2.plot(df_robustness['Epsilon'], df_robustness['Avg_Ball_Size_CI_Lower'], color='tab:blue', linewidth=1.0, alpha=0.7)
    ax2.plot(df_robustness['Epsilon'], df_robustness['Avg_Ball_Size_CI_Upper'], color='tab:blue', linewidth=1.0, alpha=0.7)

    ax2.axvline(x=0.1, color='black', linestyle=':', linewidth=1.2)
    ax2.axvline(x=1.0, color='black', linestyle=':', linewidth=1.2)
    ax2.set_xticks(explicit_ticks)
    ax2.tick_params(axis='x', labelfontfamily='sans-serif')

    ax2.set_xlabel('Epsilon', family='sans-serif', fontsize=12)
    ax2.set_ylabel('Average Ball Size', family='sans-serif')

    if chosen_eps:
        ax2.axvline(x=chosen_eps, color='black', linestyle=':', linewidth=1.2)

    ax2.set_title(f'Ball Mapper Stability: Average Ball Size ({NUM_REPETITIONS} Seeds)',
                  family='sans-serif', fontsize=13, pad=15)

    plt.savefig(FIGURES_DIR / 'ballmapper_ball_size_robustness.jpg', bbox_inches='tight')
    # plt.show()


def _run_single_comparative_ballmapper(X_scaled, target_values, median_val, eps, seed):
    """Worker function for parallel processing."""
    np.random.seed(seed)

    shuffled_idx = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[shuffled_idx]
    target_shuffled = target_values[shuffled_idx]

    bm = BallMapper(X=X_shuffled, eps=eps)
    G = bm.Graph

    high_target_sizes = []
    low_target_sizes = []

    for node in G.nodes:
        points_in_ball = bm.points_covered_by_landmarks[node]
        ball_avg_val = np.mean(target_shuffled[points_in_ball])
        ball_size = len(points_in_ball)

        if ball_avg_val >= median_val:
            high_target_sizes.append(ball_size)
        else:
            low_target_sizes.append(ball_size)

    return (
        len(high_target_sizes),
        len(low_target_sizes),
        np.mean(high_target_sizes) if high_target_sizes else 0,
        np.mean(low_target_sizes) if low_target_sizes else 0
    )


def run_epsilon_comparative_robustness(df_data, target_col, vulnerable_label, baseline_label,
                                        metric_name, cmap_name, invert_colors=False):
    """BallMapper epsilon-robustness split into high/low subgroups by a covariate's median."""
    target_values = df_data[target_col].values
    global_median = np.median(target_values[~np.isnan(target_values)])

    print(f"\n========================================================")
    print(f"THRESHOLD ALIGNMENT FOR: {metric_name}")
    print(f"Global Median (All Provinces): {global_median:.4f}")
    print(f"Nodes >= {global_median:.4f} are classed as: {'Low Literacy' if invert_colors else 'High ' + metric_name}")
    print(f"========================================================\n")

    cache_file = COMPARATIVE_CACHE_DIR / f"robustness_cache_{target_col}.csv"

    if cache_file.exists():
        print(f"-> Loading CACHED robustness data for {metric_name} from: {cache_file.name}")
        df_robustness = pd.read_csv(cache_file)
    else:
        print(f"--- RUNNING COMPARATIVE ROBUSTNESS TEST: {metric_name} ({NUM_REPETITIONS} REPS) ---")

        features = ['H0_L2_Norm', 'H0_PersistentEntropy', 'H0_Total_Persistence']
        X_scaled = StandardScaler().fit_transform(df_data[features].values)

        eps_values = np.arange(0.1, 1.05, 0.05)
        robustness_metrics = []

        for eps in tqdm(eps_values, desc=f"Evaluating Radii for {metric_name}"):
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(_run_single_comparative_ballmapper)(X_scaled, target_values, global_median, eps, seed)
                for seed in range(NUM_REPETITIONS)
            )

            c_high, c_low, s_high, s_low = zip(*results)

            robustness_metrics.append({
                'Epsilon': eps,
                'Num_High_Mean': np.mean(c_high),
                'Num_High_CI_Low': np.percentile(c_high, 2.5),
                'Num_High_CI_High': np.percentile(c_high, 97.5),
                'Num_Low_Mean': np.mean(c_low),
                'Num_Low_CI_Low': np.percentile(c_low, 2.5),
                'Num_Low_CI_High': np.percentile(c_low, 97.5),
                'Size_High_Mean': np.mean(s_high),
                'Size_High_CI_Low': np.percentile(s_high, 2.5),
                'Size_High_CI_High': np.percentile(s_high, 97.5),
                'Size_Low_Mean': np.mean(s_low),
                'Size_Low_CI_Low': np.percentile(s_low, 2.5),
                'Size_Low_CI_High': np.percentile(s_low, 97.5)
            })

        df_robustness = pd.DataFrame(robustness_metrics)
        df_robustness.to_csv(cache_file, index=False)
        print(f"-> Saved computation results to: {cache_file.name}")

    chosen_eps = BM_EPSILON
    explicit_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    cmap = plt.get_cmap(cmap_name)

    if not invert_colors:
        vul_num_mean, vul_num_low, vul_num_high = df_robustness['Num_High_Mean'], df_robustness['Num_High_CI_Low'], df_robustness['Num_High_CI_High']
        base_num_mean, base_num_low, base_num_high = df_robustness['Num_Low_Mean'], df_robustness['Num_Low_CI_Low'], df_robustness['Num_Low_CI_High']
        vul_size_mean, vul_size_low, vul_size_high = df_robustness['Size_High_Mean'], df_robustness['Size_High_CI_Low'], df_robustness['Size_High_CI_High']
        base_size_mean, base_size_low, base_size_high = df_robustness['Size_Low_Mean'], df_robustness['Size_Low_CI_Low'], df_robustness['Size_Low_CI_High']

        vulnerable_color = cmap(0.9)
        baseline_color = cmap(0.1)
    else:
        vul_num_mean, vul_num_low, vul_num_high = df_robustness['Num_Low_Mean'], df_robustness['Num_Low_CI_Low'], df_robustness['Num_Low_CI_High']
        base_num_mean, base_num_low, base_num_high = df_robustness['Num_High_Mean'], df_robustness['Num_High_CI_Low'], df_robustness['Num_High_CI_High']
        vul_size_mean, vul_size_low, vul_size_high = df_robustness['Size_Low_Mean'], df_robustness['Size_Low_CI_Low'], df_robustness['Size_Low_CI_High']
        base_size_mean, base_size_low, base_size_high = df_robustness['Size_High_Mean'], df_robustness['Size_High_CI_Low'], df_robustness['Size_High_CI_High']

        vulnerable_color = cmap(0.1)
        baseline_color = cmap(0.9)

    # STANDALONE PLOT 1: Number of Balls
    fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    ax1.plot(df_robustness['Epsilon'], vul_num_mean, color=vulnerable_color, linewidth=2, label=vulnerable_label)
    ax1.plot(df_robustness['Epsilon'], vul_num_low, color=vulnerable_color, linewidth=0.8, alpha=0.5)
    ax1.plot(df_robustness['Epsilon'], vul_num_high, color=vulnerable_color, linewidth=0.8, alpha=0.5)

    ax1.plot(df_robustness['Epsilon'], base_num_mean, color=baseline_color, linewidth=2, label=baseline_label)
    ax1.plot(df_robustness['Epsilon'], base_num_low, color=baseline_color, linewidth=0.8, alpha=0.4)
    ax1.plot(df_robustness['Epsilon'], base_num_high, color=baseline_color, linewidth=0.8, alpha=0.4)

    ax1.axvline(x=0.1, color='black', linestyle=':', linewidth=1.2)
    ax1.axvline(x=1.0, color='black', linestyle=':', linewidth=1.2)
    if chosen_eps:
        ax1.axvline(x=chosen_eps, color='black', linestyle=':', linewidth=1.2)

    ax1.set_xticks(explicit_ticks)
    ax1.set_xlabel('Epsilon', family='sans-serif', fontsize=12)
    ax1.set_ylabel('Number of Nodes', family='sans-serif')
    ax1.legend(loc='upper right', framealpha=1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{target_col}_numnodes.jpg')
    # plt.show()

    # STANDALONE PLOT 2: Average Ball Size
    fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=300)

    ax2.plot(df_robustness['Epsilon'], vul_size_mean, color=vulnerable_color, linewidth=2, label=vulnerable_label)
    ax2.plot(df_robustness['Epsilon'], vul_size_low, color=vulnerable_color, linewidth=0.8, alpha=0.5)
    ax2.plot(df_robustness['Epsilon'], vul_size_high, color=vulnerable_color, linewidth=0.8, alpha=0.5)

    ax2.plot(df_robustness['Epsilon'], base_size_mean, color=baseline_color, linewidth=2, label=baseline_label)
    ax2.plot(df_robustness['Epsilon'], base_size_low, color=baseline_color, linewidth=0.8, alpha=0.4)
    ax2.plot(df_robustness['Epsilon'], base_size_high, color=baseline_color, linewidth=0.8, alpha=0.4)

    ax2.axvline(x=0.1, color='black', linestyle=':', linewidth=1.2)
    ax2.axvline(x=1.0, color='black', linestyle=':', linewidth=1.2)
    if chosen_eps:
        ax2.axvline(x=chosen_eps, color='black', linestyle=':', linewidth=1.2)

    ax2.set_xticks(explicit_ticks)
    ax2.set_xlabel('Epsilon', family='sans-serif', fontsize=12)
    ax2.set_ylabel('Average Node Size', family='sans-serif')
    ax2.legend(loc='upper left', framealpha=1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{target_col}_avgnodesize.jpg')
    # plt.show()
