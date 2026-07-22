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
        styled_matrix = (
            matrix.style
            .set_properties(**{'text-align': 'center', 'border': '1px solid black'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center'), ('font-weight', 'bold')],
            }])
        )
        try:
            display(styled_matrix)
        except NameError:
            # display() not available in non-Jupyter environments
            print(matrix.to_string())

    print("-------------------------------------------\n")

    if save_dir and has_any:
        out = Path(save_dir) / f"2022_intersection_matrix{suffix}.csv"
        matrix.to_csv(out)
        print(f"-> Saved Intersection Matrix to {out.name}")

    return matrix


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
        bm.Graph, seed=2022, k=0.8
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
    # plt.show()

    target_sort_col = (f'Majority {color_col}' if is_categorical else f'Mean {color_col}')
    df_summary = (
        pd.DataFrame(summary_data)
        .sort_values(target_sort_col, ascending=False)
        .reset_index(drop=True)
    )
    return df_summary, bm
