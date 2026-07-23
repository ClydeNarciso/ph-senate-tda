"""
robustness.py — BallMapper epsilon-robustness diagnostics for the ZZPH pipeline.

Two analyses are implemented:

run_epsilon_robustness
    Sweeps epsilon from 0.1 to 1.0 in 0.05 increments and records the
    mean ± 95% CI for number of balls and average ball size across
    ``NUM_REPETITIONS`` random permutations.  Produces two standalone plots.

run_epsilon_comparative_robustness
    For a given socioeconomic covariate, splits nodes into "vulnerable"
    (above/below the global median depending on ``invert_colors``) and
    "baseline" groups at each epsilon, and records separate node-count and
    average-size statistics for each group.  Produces two standalone plots.

All heavy computation is cached in ``ROBUSTNESS_DIR`` so the pipeline can
be resumed without re-running the full bootstrap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pyballmapper import BallMapper
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    BALLMAPPER_DIR,
    BM_EPSILON,
    NUM_REPETITIONS,
    ROBUSTNESS_CACHE_FILE,
    ROBUSTNESS_DIR,
)

_EPS_VALUES    = np.arange(0.1, 1.05, 0.05)
_EXPLICIT_TICKS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickling with joblib)
# ---------------------------------------------------------------------------

def _run_single_ballmapper(
    X: np.ndarray, eps: float, seed: int
) -> tuple[int, float]:
    np.random.seed(seed)
    bm = BallMapper(X=np.random.permutation(X), eps=eps)
    sizes = [len(bm.points_covered_by_landmarks[n]) for n in bm.Graph.nodes]
    return len(sizes), (np.mean(sizes) if sizes else 0.0)


def _run_single_comparative_ballmapper(
    X_scaled: np.ndarray,
    target_values: np.ndarray,
    median_val: float,
    eps: float,
    seed: int,
) -> tuple[int, int, float, float]:
    np.random.seed(seed)
    idx = np.random.permutation(len(X_scaled))
    bm  = BallMapper(X=X_scaled[idx], eps=eps)
    tv  = target_values[idx]

    high_sizes, low_sizes = [], []
    for node in bm.Graph.nodes:
        pts  = bm.points_covered_by_landmarks[node]
        mean = np.mean(tv[pts])
        sz   = len(pts)
        (high_sizes if mean >= median_val else low_sizes).append(sz)

    return (
        len(high_sizes),
        len(low_sizes),
        float(np.mean(high_sizes)) if high_sizes else 0.0,
        float(np.mean(low_sizes))  if low_sizes  else 0.0,
    )


# ---------------------------------------------------------------------------
# Global epsilon robustness
# ---------------------------------------------------------------------------

def run_epsilon_robustness(
    df_features: pd.DataFrame,
    cache_path: Path = ROBUSTNESS_CACHE_FILE,
    n_reps: int = NUM_REPETITIONS,
    chosen_eps: float = BM_EPSILON,
    save_dir: Path = BALLMAPPER_DIR,
) -> None:
    """Sweep epsilon and plot mean ± 95% CI for ball count and ball size."""
    if cache_path.exists():
        print("\n-> Loading cached Epsilon Robustness data…")
        df_rob = pd.read_csv(cache_path)
    else:
        print(f"\n--- RUNNING EPSILON ROBUSTNESS ({n_reps} reps) ---")
        X = StandardScaler().fit_transform(
            df_features[["H0_L2_Norm", "H0_PersistentEntropy", "H0_Total_Persistence"]].values
        )
        rows = []
        for eps in tqdm(_EPS_VALUES, desc="Evaluating Epsilons"):
            res = Parallel(n_jobs=-1)(
                delayed(_run_single_ballmapper)(X, eps, s) for s in range(n_reps)
            )
            nb, sz = zip(*res)
            rows.append({
                "Epsilon":             eps,
                "Num_Balls_Mean":      np.mean(nb),
                "Num_Balls_CI_Lower":  np.percentile(nb, 2.5),
                "Num_Balls_CI_Upper":  np.percentile(nb, 97.5),
                "Avg_Ball_Size_Mean":  np.mean(sz),
                "Avg_Ball_Size_CI_Lower": np.percentile(sz, 2.5),
                "Avg_Ball_Size_CI_Upper": np.percentile(sz, 97.5),
            })
        df_rob = pd.DataFrame(rows)
        df_rob.to_csv(cache_path, index=False)

    _plot_robustness_single(
        df_rob, chosen_eps, save_dir,
        y_col="Num_Balls_Mean",
        lo_col="Num_Balls_CI_Lower",
        hi_col="Num_Balls_CI_Upper",
        ylabel="Number of Balls",
        fname="ballmapper_num_balls_robustness.jpg",
        n_reps=n_reps,
    )
    _plot_robustness_single(
        df_rob, chosen_eps, save_dir,
        y_col="Avg_Ball_Size_Mean",
        lo_col="Avg_Ball_Size_CI_Lower",
        hi_col="Avg_Ball_Size_CI_Upper",
        ylabel="Average Ball Size",
        fname="ballmapper_ball_size_robustness.jpg",
        n_reps=n_reps,
    )


def _plot_robustness_single(
    df_rob: pd.DataFrame,
    chosen_eps: float,
    save_dir: Path,
    y_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    fname: str,
    n_reps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(df_rob["Epsilon"], df_rob[y_col],  color="tab:blue", linewidth=2)
    ax.plot(df_rob["Epsilon"], df_rob[lo_col], color="tab:blue", linewidth=1.0, alpha=0.7)
    ax.plot(df_rob["Epsilon"], df_rob[hi_col], color="tab:blue", linewidth=1.0, alpha=0.7)
    ax.axvline(x=0.1,        color="black", linestyle=":", linewidth=1.2)
    ax.axvline(x=1.0,        color="black", linestyle=":", linewidth=1.2)
    ax.axvline(x=chosen_eps, color="black", linestyle=":", linewidth=1.2)
    ax.set_xticks(_EXPLICIT_TICKS)
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"BallMapper Stability: {ylabel} ({n_reps} Seeds)",
        fontsize=13, pad=15,
    )
    plt.savefig(save_dir / fname, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {fname}")


# ---------------------------------------------------------------------------
# Comparative epsilon robustness
# ---------------------------------------------------------------------------

def run_epsilon_comparative_robustness(
    df_master: pd.DataFrame,
    target_col: str,
    vulnerable_label: str,
    baseline_label: str,
    metric_name: str,
    cmap_name: str,
    invert_colors: bool = False,
    n_reps: int = NUM_REPETITIONS,
    chosen_eps: float = BM_EPSILON,
    cache_dir: Path = ROBUSTNESS_DIR,
) -> None:
    """Run and plot comparative epsilon robustness for one covariate."""
    target_values = df_master[target_col].values
    global_median = float(np.median(target_values))

    print(f"\n{'='*58}")
    print(f"COMPARATIVE ROBUSTNESS: {metric_name}")
    print(f"Global median: {global_median:.4f}  |  "
          f"Nodes ≥ median → "
          f"{'Low ' + metric_name if invert_colors else 'High ' + metric_name}")
    print(f"{'='*58}")

    cache_path = cache_dir / f"robustness_cache_{target_col}.csv"

    if cache_path.exists():
        print(f"-> Loading cached data from {cache_path.name}")
        df_rob = pd.read_csv(cache_path)
    else:
        print(f"--- RUNNING COMPARATIVE ROBUSTNESS ({n_reps} reps) ---")
        X = StandardScaler().fit_transform(
            df_master[["H0_L2_Norm", "H0_PersistentEntropy", "H0_Total_Persistence"]].values
        )
        rows = []
        for eps in tqdm(_EPS_VALUES, desc=f"Evaluating radii for {metric_name}"):
            res = Parallel(n_jobs=-1, backend="threading")(
                delayed(_run_single_comparative_ballmapper)(
                    X, target_values, global_median, eps, s
                )
                for s in range(n_reps)
            )
            c_hi, c_lo, s_hi, s_lo = zip(*res)
            rows.append({
                "Epsilon":          eps,
                "Num_High_Mean":    np.mean(c_hi),
                "Num_High_CI_Low":  np.percentile(c_hi, 2.5),
                "Num_High_CI_High": np.percentile(c_hi, 97.5),
                "Num_Low_Mean":     np.mean(c_lo),
                "Num_Low_CI_Low":   np.percentile(c_lo, 2.5),
                "Num_Low_CI_High":  np.percentile(c_lo, 97.5),
                "Size_High_Mean":   np.mean(s_hi),
                "Size_High_CI_Low": np.percentile(s_hi, 2.5),
                "Size_High_CI_High":np.percentile(s_hi, 97.5),
                "Size_Low_Mean":    np.mean(s_lo),
                "Size_Low_CI_Low":  np.percentile(s_lo, 2.5),
                "Size_Low_CI_High": np.percentile(s_lo, 97.5),
            })
        df_rob = pd.DataFrame(rows)
        df_rob.to_csv(cache_path, index=False)

    cmap = plt.get_cmap(cmap_name)
    if not invert_colors:
        vul_color  = cmap(0.9)
        base_color = cmap(0.1)
        vn_m, vn_l, vn_h = df_rob["Num_High_Mean"],  df_rob["Num_High_CI_Low"],  df_rob["Num_High_CI_High"]
        bn_m, bn_l, bn_h = df_rob["Num_Low_Mean"],   df_rob["Num_Low_CI_Low"],   df_rob["Num_Low_CI_High"]
        vs_m, vs_l, vs_h = df_rob["Size_High_Mean"],  df_rob["Size_High_CI_Low"],  df_rob["Size_High_CI_High"]
        bs_m, bs_l, bs_h = df_rob["Size_Low_Mean"],   df_rob["Size_Low_CI_Low"],   df_rob["Size_Low_CI_High"]
    else:
        vul_color  = cmap(0.1)
        base_color = cmap(0.9)
        vn_m, vn_l, vn_h = df_rob["Num_Low_Mean"],   df_rob["Num_Low_CI_Low"],   df_rob["Num_Low_CI_High"]
        bn_m, bn_l, bn_h = df_rob["Num_High_Mean"],  df_rob["Num_High_CI_Low"],  df_rob["Num_High_CI_High"]
        vs_m, vs_l, vs_h = df_rob["Size_Low_Mean"],   df_rob["Size_Low_CI_Low"],   df_rob["Size_Low_CI_High"]
        bs_m, bs_l, bs_h = df_rob["Size_High_Mean"],  df_rob["Size_High_CI_Low"],  df_rob["Size_High_CI_High"]

    _plot_comparative(
        df_rob, vn_m, vn_l, vn_h, bn_m, bn_l, bn_h,
        vul_color, base_color, vulnerable_label, baseline_label,
        chosen_eps, ylabel="Number of Nodes",
        out_path=cache_dir / f"{target_col}_numnodes.jpg",
    )
    _plot_comparative(
        df_rob, vs_m, vs_l, vs_h, bs_m, bs_l, bs_h,
        vul_color, base_color, vulnerable_label, baseline_label,
        chosen_eps, ylabel="Average Node Size",
        out_path=cache_dir / f"{target_col}_avgnodesize.jpg",
    )


def _plot_comparative(
    df_rob: pd.DataFrame,
    vul_mean, vul_lo, vul_hi,
    base_mean, base_lo, base_hi,
    vul_color, base_color,
    vulnerable_label: str,
    baseline_label: str,
    chosen_eps: float,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    eps = df_rob["Epsilon"]

    ax.plot(eps, vul_mean,  color=vul_color,  linewidth=2,   label=vulnerable_label)
    ax.plot(eps, vul_lo,    color=vul_color,  linewidth=0.8, alpha=0.5)
    ax.plot(eps, vul_hi,    color=vul_color,  linewidth=0.8, alpha=0.5)
    ax.plot(eps, base_mean, color=base_color, linewidth=2,   label=baseline_label)
    ax.plot(eps, base_lo,   color=base_color, linewidth=0.8, alpha=0.4)
    ax.plot(eps, base_hi,   color=base_color, linewidth=0.8, alpha=0.4)

    ax.axvline(x=0.1,        color="black", linestyle=":", linewidth=1.2)
    ax.axvline(x=1.0,        color="black", linestyle=":", linewidth=1.2)
    ax.axvline(x=chosen_eps, color="black", linestyle=":", linewidth=1.2)
    ax.set_xticks(_EXPLICIT_TICKS)
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right" if "Number" in ylabel else "upper left", framealpha=1)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {out_path.name}")
