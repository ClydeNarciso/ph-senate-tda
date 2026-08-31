"""BallMapper graph construction, node visualization, and node-overlap
(intersection matrix) helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
from IPython.display import display
from pyballmapper import BallMapper

from config import ADMIN_CMAP, FIGURES_DIR, TABLES_DIR

# ---------------------------------------------------------------------------
# Quartile-based profile catalogue
# ---------------------------------------------------------------------------
# Profile code = "{L2_tier}-{TotalPersistence_tier}"
#   L2 Norm tier        : H = high (rigid/stable), L = low (fragmented), A = average
#   Total Pers. tier    : H = high (many long-lived components), L = low, A = average
# Cut points: Q1 (25th pct) and Q3 (75th pct) across all BallMapper nodes.
# Methodology verified against collaborator's edited summary_table_master.csv
# (23/23 perfect match).

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "A-A": "A-A: Standard Plurality",
    "A-H": "A-H: Fragile Coalition",
    "A-L": "A-L: Cohesive Swing-Vote",
    "H-A": "H-A: Polarized Mainstream",
    "H-H": "H-H: Hyper-Fragmentation",
    "H-L": "H-L: Unified Stronghold",
    "L-A": "L-A: Contested Center",
    "L-H": "L-H: Hyper-Local Fragmentation",
    "L-L": "L-L: Competitive/Balanced",
}

PROFILE_COLOURS: dict[str, str] = {
    "A-A": "#78909C",
    "A-H": "#AB47BC",
    "A-L": "#26A69A",
    "H-A": "#EF5350",
    "H-H": "#B71C1C",
    "H-L": "#FF7043",
    "L-A": "#42A5F5",
    "L-H": "#7E57C2",
    "L-L": "#66BB6A",
}


def _compute_node_dialect_label(df_features: pd.DataFrame, points_in_node: list) -> str:
    """Return 'count/total' provinces sharing the plurality dialect in a node."""
    # Using iloc to strictly maintain position index
    dialects = df_features.iloc[points_in_node]['Dominant_Dialect']
    total = len(dialects)
    if total == 0:
        return '0/0'
    majority_count = int(dialects.value_counts().iloc[0])
    return f'{majority_count}/{total}'


def compute_intersection_matrix(bm: BallMapper, df_features: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing which provinces overlap between node pairs."""
    nodes = sorted(bm.Graph.nodes)
    node_to_provs = {
        n: set(df_features.iloc[bm.points_covered_by_landmarks[n]]['Province'])
        for n in nodes
    }
    matrix = pd.DataFrame('', index=nodes, columns=nodes, dtype=str)
    for n in nodes:
        matrix.loc[n, n] = '-'

    has_any = False
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            inter = node_to_provs[i] & node_to_provs[j]
            if inter:
                label = ', '.join(sorted(inter))
                matrix.loc[i, j] = label
                matrix.loc[j, i] = label
                has_any = True

    return matrix, has_any


def print_intersection_matrix_once(bm: BallMapper,
                                    df_features: pd.DataFrame,
                                    save_dir: Path = TABLES_DIR,
                                    suffix: str = '') -> pd.DataFrame:
    """Print and optionally save the node intersection matrix (call once)."""
    print("\n--- TOPOLOGICAL INTERSECTIONS MATRIX (BRIDGES) ---")
    matrix, has_any = compute_intersection_matrix(bm, df_features)

    if not has_any:
        print("No topological intersections found at this epsilon level.")
    else:
        display(
            matrix.style
            .set_properties(**{'text-align': 'center', 'border': '1px solid black'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center'), ('font-weight', 'bold')],
            }])
        )

    print("-------------------------------------------\n")

    if save_dir and has_any:
        out = Path(save_dir) / f"2019_intersection_matrix{suffix}.csv"
        matrix.to_csv(out)
        print(f"-> Saved Intersection Matrix to {out.name}")

    return matrix


def compute_quartile_profiles(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Assign a Quartile-based Profile to every BallMapper node."""
    df = df_summary.copy()
    l2_col = "Mean L2 Norm (Scaled)"
    tp_col = "Mean Total Persistence (Scaled)"

    if l2_col not in df.columns or tp_col not in df.columns:
        print(f"  [!] Cannot compute profiles: missing '{l2_col}' or '{tp_col}'.")
        df["Quartile-based Profile"] = "N/A"
        df["Profile Description"]    = "N/A"
        return df

    q1_l2 = df[l2_col].quantile(0.25)
    q3_l2 = df[l2_col].quantile(0.75)
    q1_tp = df[tp_col].quantile(0.25)
    q3_tp = df[tp_col].quantile(0.75)

    def _tier(val: float, q1: float, q3: float) -> str:
        if val <= q1:
            return "L"
        elif val >= q3:
            return "H"
        return "A"

    df["Quartile-based Profile"] = (
        df[l2_col].apply(lambda x: _tier(x, q1_l2, q3_l2)) + "-" +
        df[tp_col].apply(lambda x: _tier(x, q1_tp, q3_tp))
    )
    df["Profile Description"] = (
        df["Quartile-based Profile"].map(PROFILE_DESCRIPTIONS).fillna("Unknown")
    )

    print("\n--- QUARTILE-BASED PROFILE DISTRIBUTION ---")
    for code, cnt in df["Quartile-based Profile"].value_counts().items():
        desc = PROFILE_DESCRIPTIONS.get(code, "")
        print(f"  {code}  ({cnt} node{'s' if cnt != 1 else ''})  —  {desc}")
    print(f"  Cut points:  L2 Q1={q1_l2:.3f}, Q3={q3_l2:.3f} | "
          f"TotPers Q1={q1_tp:.3f}, Q3={q3_tp:.3f}")

    return df


def run_and_visualize_ballmapper_profile(
    df_features: pd.DataFrame,
    X_scaled: np.ndarray,
    eps: float,
    save_dir: Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, BallMapper]:
    """BallMapper graph coloured by Quartile-based Profile."""
    print("\n--- RUNNING PROFILE BALLMAPPER ---")
    bm = BallMapper(X=X_scaled, eps=eps)
    nodes_list = list(bm.Graph.nodes)

    # Node summary (topological features only)
    rows = []
    for node in nodes_list:
        pts    = bm.points_covered_by_landmarks[node]
        df_pts = df_features.iloc[pts]
        rows.append({
            "Node ID":                          node,
            "No. Provinces":                    len(pts),
            "Mean L2 Norm (Scaled)":            df_pts["H0_L2_Norm_scaled"].mean(),
            "Mean Total Persistence (Scaled)":  df_pts["H0_Total_Persistence_scaled"].mean(),
            "Mean Entropy (Scaled)":            df_pts["H0_PersistentEntropy_scaled"].mean(),
            "Provinces Included":               ", ".join(df_pts["Province"].tolist()),
            "Provinces (Region)":               ", ".join(
                f"{r['Province']} ({r['Region']})"
                for _, r in df_pts.iterrows()
            ),
        })
    df_summary = pd.DataFrame(rows)
    df_summary = compute_quartile_profiles(df_summary)

    # Standard layout matching the other graphs
    pos = nx.spring_layout(
        bm.Graph, seed=50, k=1.8, iterations=150, scale=2.5
    )

    profile_map = dict(zip(df_summary["Node ID"], df_summary["Quartile-based Profile"]))
    present = sorted(df_summary["Quartile-based Profile"].unique())

    node_colours = [
        PROFILE_COLOURS.get(profile_map.get(n, ""), "#E0E0E0")
        for n in nodes_list
    ]
    node_sizes = [
        400 + len(bm.points_covered_by_landmarks[n]) * 150
        for n in nodes_list
    ]
    node_labels = {n: f"{n}\n{profile_map.get(n, '?')}" for n in nodes_list}

    # Calculate edge intersection labels
    edge_labels = {}
    for i, j in bm.Graph.edges():
        pts_i = set(df_features.iloc[bm.points_covered_by_landmarks[i]]["Province"])
        pts_j = set(df_features.iloc[bm.points_covered_by_landmarks[j]]["Province"])
        shared = pts_i & pts_j
        if shared:
            edge_labels[(i, j)] = str(len(shared))

    # Draw
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    nx.draw_networkx_edges(
        bm.Graph, pos, ax=ax, alpha=0.5, width=2.5, edge_color="#444444"
    )
    nx.draw_networkx_nodes(
        bm.Graph, pos, ax=ax,
        node_size=node_sizes, node_color=node_colours,
        edgecolors='black', linewidths=1.5, alpha=0.95,
    )
    nx.draw_networkx_labels(
        bm.Graph, pos, labels=node_labels, ax=ax,
        font_size=12, font_family='serif', font_weight='bold',
    )
    
    if edge_labels:
        nx.draw_networkx_edge_labels(
            bm.Graph, pos, edge_labels=edge_labels, ax=ax,
            font_size=9, font_family='serif', font_color='black',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9)
        )
    
    ax.axis("off")

    # Match categorical legend style
    legend_handles = [
        mpatches.Patch(
            facecolor=PROFILE_COLOURS.get(p, "#E0E0E0"),
            edgecolor="black", linewidth=1.0,
            label=PROFILE_DESCRIPTIONS.get(p, p),
        )
        for p in present
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(1.2, 1),
        title="Quartile-based Profile",
        title_fontproperties={'family': 'serif', 'size': 12, 'weight': 'bold'},
        prop={'family': 'serif', 'size': 11},
    )

    if save_dir:
        out = Path(save_dir) / f"bm_eps_{eps}_profile.jpg"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"-> Saved: {out.name}")

    return df_summary, bm


def run_and_visualize_ballmapper(
    df_features: pd.DataFrame,
    X_scaled: np.ndarray,
    eps: float,
    save_dir: Path = FIGURES_DIR,
    color_col: str = 'Mean_Admin_Prop',
    cmap=None,
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = 'Average Value',
    cbar_format=None,
    mode_name: str = 'Model',
) -> tuple[pd.DataFrame, BallMapper]:

    print(f"\n--- RUNNING BALLMAPPER (epsilon={eps}) | Color By: {color_col} ---")
    bm = BallMapper(X=X_scaled, eps=eps)

    is_categorical = (
        df_features[color_col].dtype == object
        or color_col == 'Dominant_Dialect'
    )

    nodes = list(bm.Graph.nodes)
    node_sizes: list[float] = []
    node_colors: list = []
    node_mode_vals: list = []
    node_averages: list[float] = []
    summary_data: list[dict] = []

    if is_categorical:
        for node in nodes:
            pts = bm.points_covered_by_landmarks[node]
            # Use iloc for safe positional array extraction
            mode_val = (df_features.iloc[pts][color_col].mode().iloc[0]
                        if len(pts) > 0 else 'N/A')
            node_mode_vals.append(mode_val)

        unique_cats = sorted(set(node_mode_vals))

        if cmap is None or not isinstance(cmap, mcolors.ListedColormap):
            discrete_cmap = plt.get_cmap('tab20', len(unique_cats))
            cat_to_color = {cat: mcolors.to_hex(discrete_cmap(i % 20)) for i, cat in enumerate(unique_cats)}
        else:
            cat_to_color = {cat: mcolors.to_hex(cmap.colors[i % len(cmap.colors)]) for i, cat in enumerate(unique_cats)}

        node_colors = [cat_to_color[m] for m in node_mode_vals]
    else:
        for node in nodes:
            pts = bm.points_covered_by_landmarks[node]
            avg = df_features.iloc[pts][color_col].mean()
            node_averages.append(avg)

        if vmin is None:
            vmin = min(node_averages) if node_averages else 0
        if vmax is None:
            vmax = max(node_averages) if node_averages else 1
        node_colors = node_averages
        if cmap is None:
            cmap = ADMIN_CMAP

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    for idx, node in enumerate(nodes):
        pts = bm.points_covered_by_landmarks[node]
        node_sizes.append(400 + len(pts) * 150)

        df_pts = df_features.iloc[pts]
        avg_admin = df_pts['Mean_Admin_Prop'].mean()
        target_val = (node_mode_vals[idx] if is_categorical else node_averages[idx])
        dialect_label = _compute_node_dialect_label(df_features, pts)

        summary_data.append({
            'Node ID': node,
            'No. Provinces': len(pts),
            (f'Majority {color_col}' if is_categorical
             else f'Mean {color_col}'): target_val,
            'Major Dialect Ratio': dialect_label,
            'Admin Share (%)': round(avg_admin * 100, 2),
            'Mean Entropy (Raw)': df_pts['H0_PersistentEntropy'].mean(),
            'Mean Entropy (Scaled)': df_pts['H0_PersistentEntropy_scaled'].mean(),
            'Mean L2 Norm (Raw)': df_pts['H0_L2_Norm'].mean(),
            'Mean L2 Norm (Scaled)': df_pts['H0_L2_Norm_scaled'].mean(),
            'Mean Total Persistence (Raw)': df_pts['H0_Total_Persistence'].mean(),
            'Mean Total Persistence (Scaled)': df_pts['H0_Total_Persistence_scaled'].mean(),
            'No. of Provinces in each Region': ', '.join(
                f"{reg} ({cnt})" for reg, cnt in df_pts['Region'].value_counts().items()
            ),
            'Provinces Included': ', '.join(df_pts['Province'].tolist()),
        })

    pos = nx.spring_layout(
        bm.Graph, seed=50, k=1.8, iterations=150, scale=2.5
    )
    nx.draw_networkx_edges(
        bm.Graph, pos, ax=ax, alpha=0.5, width=2.5, edge_color="#444444"
    )

    if is_categorical:
        nx.draw_networkx_nodes(
            bm.Graph, pos, ax=ax,
            node_size=node_sizes, node_color=node_colors,
            edgecolors='black', linewidths=1.5, alpha=0.95,
        )
    else:
        nodes_plot = nx.draw_networkx_nodes(
            bm.Graph, pos, ax=ax,
            node_size=node_sizes, node_color=node_colors,
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors='black', linewidths=1.5, alpha=0.95,
        )

    nx.draw_networkx_labels(
        bm.Graph, pos, ax=ax,
        font_size=12, font_family='serif', font_weight='bold',
    )

    # Calculate and draw edge intersection labels for the standard graphs too
    edge_labels = {}
    for i, j in bm.Graph.edges():
        pts_i = set(df_features.iloc[bm.points_covered_by_landmarks[i]]["Province"])
        pts_j = set(df_features.iloc[bm.points_covered_by_landmarks[j]]["Province"])
        shared = pts_i & pts_j
        if shared:
            edge_labels[(i, j)] = str(len(shared))
            
    if edge_labels:
        nx.draw_networkx_edge_labels(
            bm.Graph, pos, edge_labels=edge_labels, ax=ax,
            font_size=9, font_family='serif', font_color='black',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9)
        )

    ax.axis('off')

    if is_categorical:
        legend_elements = [
            mpatches.Patch(facecolor=col, edgecolor='black', label=str(cat))
            for cat, col in cat_to_color.items()
        ]
        ax.legend(
            handles=legend_elements, loc='upper right',
            title=cbar_label, bbox_to_anchor=(1.2, 1),
            prop={'family': 'serif', 'size': 11},
        )
    else:
        cbar = plt.colorbar(nodes_plot, ax=ax, fraction=0.03, pad=0.02, aspect=30)
        if cbar_format:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_format))
        cbar.set_label(cbar_label, fontweight='bold', fontsize=12, labelpad=15)

    if save_dir:
        out_path = Path(save_dir) / f'bm_eps_{eps}_{mode_name}.jpg'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')

    target_sort_col = (f'Majority {color_col}' if is_categorical else f'Mean {color_col}')
    df_summary = (
        pd.DataFrame(summary_data)
        .sort_values(target_sort_col, ascending=False)
        .reset_index(drop=True)
    )
    df_summary = compute_quartile_profiles(df_summary)
    return df_summary, bm