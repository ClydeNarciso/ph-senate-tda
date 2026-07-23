"""
ballmapper.py — BallMapper graph construction and all output generation
for the ZZPH pipeline.

Outputs produced
----------------
Per run:
  4_ballmapper_clusters/
    unified_bm_summary_eps_{eps}.csv      — master node table (all covariates)
    unified_bm_summary_eps_{eps}.tex      — LaTeX version
    unified_intersection_matrix_eps_{eps}.csv — pairwise province overlap matrix
    bm_summary_{suffix}_eps_{eps}.csv     — per-covariate compact table
    bm_summary_{suffix}_eps_{eps}.tex     — two-column parallel LaTeX layout
    bm_graph_eps_{eps}_{mode}.png         — coloured network graph
    scatter_{mode}_{metric}.jpg           — topology vs covariate scatterplots
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import networkx as nx
import plotly.express as px
from scipy.stats import pearsonr
from pyballmapper import BallMapper

from config import BALLMAPPER_DIR, BM_EPSILON, COLOR_CONFIGS


# ---------------------------------------------------------------------------
# Node summary construction
# ---------------------------------------------------------------------------

def build_unified_ballmapper_structure(
    df_features: pd.DataFrame,
    X_scaled: np.ndarray,
    eps: float = BM_EPSILON,
    save_dir=BALLMAPPER_DIR,
) -> tuple[BallMapper, pd.DataFrame]:
    """Construct the BallMapper graph and produce all node-level tables.

    The unified summary table has one row per BallMapper node and includes:
    - node size (number of provinces)
    - mean values for all four socioeconomic covariates
    - mean scaled topological features
    - sorted province list and region breakdown

    Four per-covariate compact tables are also written in both CSV and a
    two-column parallel LaTeX layout to minimise vertical space in papers.
    The intersection matrix records which provinces are shared between nodes.

    Parameters
    ----------
    df_features:
        Province-level feature DataFrame (output of the socioeconomic merge).
    X_scaled:
        Standardised (n_provinces × 3) array of topological features.
    eps:
        BallMapper radius.
    save_dir:
        Directory for all output files.

    Returns
    -------
    bm : BallMapper
        Fitted BallMapper object.
    df_summary : pd.DataFrame
        Unified node-level summary table.
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    print(f"\n--- BUILDING MASTER BALLMAPPER STRUCTURE | ε = {eps} ---")
    bm = BallMapper(X=X_scaled, eps=eps)

    rows = []
    for node in bm.Graph.nodes:
        pts = bm.points_covered_by_landmarks[node]
        sub = df_features.iloc[pts]

        reg_counts = sub["Region"].value_counts()
        regions_str = ", ".join(f"{r} ({c})" for r, c in reg_counts.items())

        rows.append({
            "Node ID":                     node,
            "No. Provinces":               len(pts),
            "Poverty Incidence (%)":       sub["Mean_Poverty_Incidence"].mean(),
            "Literacy Rate (%)":           sub["Basic_Literacy_Rate"].mean(),
            "IRA Dependency (%)":          sub["Mean_IRA_Dependency"].mean(),
            "Admin Vote Share (%)":        sub["Mean_Admin_Vote"].mean(),
            "Mean Entropy (Scaled)":       sub["H0_PersistentEntropy_scaled"].mean(),
            "Mean L2 Norm (Scaled)":       sub["H0_L2_Norm_scaled"].mean(),
            "Mean Total Persistence (Scaled)": sub["H0_Total_Persistence_scaled"].mean(),
            "Provinces Included":          ", ".join(sub["Province"].tolist()),
            "Regions Included":            regions_str,
        })

    df_summary = (
        pd.DataFrame(rows)
        .sort_values("Node ID")
        .reset_index(drop=True)
    )

    # --- Master CSV + LaTeX ---
    df_summary.to_csv(save_dir / f"unified_bm_summary_eps_{eps}.csv", index=False)
    fmt = {
        "Poverty Incidence (%)":           "{:.2f}",
        "Literacy Rate (%)":               "{:.2f}",
        "IRA Dependency (%)":              "{:.2f}",
        "Admin Vote Share (%)":            "{:.2f}",
        "Mean Entropy (Scaled)":           "{:.2f}",
        "Mean L2 Norm (Scaled)":           "{:.2f}",
        "Mean Total Persistence (Scaled)": "{:.2f}",
    }
    with open(save_dir / f"unified_bm_summary_eps_{eps}.tex", "w") as fh:
        fh.write(
            df_summary.style.format(fmt).hide(axis="index")
            .to_latex(hrules=True)
        )

    # --- Per-covariate compact tables ---
    compact_configs = [
        {"col": "Poverty Incidence (%)",  "cmap": "RdYlBu_r", "suffix": "poverty"},
        {"col": "Literacy Rate (%)",       "cmap": "viridis",   "suffix": "literacy"},
        {"col": "IRA Dependency (%)",      "cmap": "plasma",    "suffix": "ira_dependency"},
        {"col": "Admin Vote Share (%)",    "cmap": "coolwarm",  "suffix": "admin_vote"},
    ]
    for cc in compact_configs:
        df_sub = df_summary[
            ["Node ID", "No. Provinces", cc["col"], "Provinces Included"]
        ].copy()
        df_sub.columns = ["Node ID", "Size", cc["col"], "Provinces Included"]
        df_sub.to_csv(save_dir / f"bm_summary_{cc['suffix']}_eps_{eps}.csv", index=False)

        with open(save_dir / f"bm_summary_{cc['suffix']}_eps_{eps}.tex", "w") as fh:
            fh.write(_parallel_latex_table(df_sub, cc["col"]))

    # --- Intersection matrix ---
    _matrix = _build_intersection_matrix(bm, df_features)
    if not _matrix.empty:
        _matrix.to_csv(save_dir / f"unified_intersection_matrix_eps_{eps}.csv")

    return bm, df_summary


def _parallel_latex_table(df: pd.DataFrame, value_col: str) -> str:
    """Split ``df`` into two side-by-side blocks to save vertical LaTeX space.

    Each block has four columns (Node ID | Size | Value | Provinces).
    A vertical rule separates the two blocks.
    """
    n       = len(df)
    half    = int(np.ceil(n / 2))
    left    = df.iloc[:half].copy().reset_index(drop=True)
    right   = df.iloc[half:].copy().reset_index(drop=True)

    # Pad shorter side
    if len(right) < half:
        pad = pd.DataFrame(
            [[""] * df.shape[1]] * (half - len(right)), columns=df.columns
        )
        right = pd.concat([right, pad], ignore_index=True)

    combined = pd.concat([left, right], axis=1)

    def _fmt(val):
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    return (
        combined.style
        .format(_fmt)
        .hide(axis="index")
        .to_latex(hrules=True, column_format="cccc|cccc")
    )


def _build_intersection_matrix(
    bm: BallMapper,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """Return a node × node DataFrame of shared province lists."""
    nodes = sorted(bm.Graph.nodes)
    node_provs = {
        n: set(df_features.iloc[bm.points_covered_by_landmarks[n]]["Province"].tolist())
        for n in nodes
    }
    matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=str).fillna("")
    has_any = False

    for i in nodes:
        for j in nodes:
            if i == j:
                matrix.loc[i, j] = "-"
            else:
                shared = node_provs[i] & node_provs[j]
                if shared:
                    matrix.loc[i, j] = ", ".join(sorted(shared))
                    has_any = True

    return matrix if has_any else pd.DataFrame()


# ---------------------------------------------------------------------------
# Network visualisation
# ---------------------------------------------------------------------------

def visualize_ballmapper_coloring(
    bm: BallMapper,
    df_features: pd.DataFrame,
    color_col: str,
    cmap,
    vmin: float | None,
    vmax: float | None,
    cbar_label: str,
    cbar_format,
    mode_name: str,
    eps: float = BM_EPSILON,
    save_dir=BALLMAPPER_DIR,
) -> None:
    """Draw and save a coloured BallMapper network graph (matplotlib).

    Node size scales with the number of provinces it contains.
    Node colour encodes the mean value of ``color_col`` across member
    provinces.  Labels are drawn with a white stroke halo for legibility
    on coloured backgrounds.
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    print(f"\n-> Generating BallMapper graph for: {mode_name.upper()}")

    node_avgs = [
        df_features.iloc[bm.points_covered_by_landmarks[n]][color_col].mean()
        for n in bm.Graph.nodes
    ]
    _vmin = vmin if vmin is not None else np.nanmin(node_avgs)
    _vmax = vmax if vmax is not None else np.nanmax(node_avgs)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    pos = nx.spring_layout(bm.Graph, seed=31415, k=1.8, iterations=150, scale=2.5)
    sizes = [400 + len(bm.points_covered_by_landmarks[n]) * 150 for n in bm.Graph.nodes]

    nx.draw_networkx_edges(
        bm.Graph, pos, ax=ax, alpha=0.5, width=2.5, edge_color="#444444"
    )
    nodes_plt = nx.draw_networkx_nodes(
        bm.Graph, pos, ax=ax,
        node_size=sizes, node_color=node_avgs,
        cmap=cmap, vmin=_vmin, vmax=_vmax,
        edgecolors="black", linewidths=1.5, alpha=0.95,
    )
    labels = nx.draw_networkx_labels(
        bm.Graph, pos, ax=ax, font_size=12, font_family="sans-serif"
    )
    for _, txt in labels.items():
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ])

    ax.axis("off")
    cbar = plt.colorbar(nodes_plt, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    if cbar_format:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_format))

    out_path = save_dir / f"bm_graph_eps_{eps}_{mode_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Scatter plots (topology vs covariate)
# ---------------------------------------------------------------------------

def generate_topology_scatter_plots(
    df_summary: pd.DataFrame,
    color_col: str,
    cmap,
    mode_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    save_dir=BALLMAPPER_DIR,
) -> None:
    """Produce three Pearson-annotated scatter plots (one per TDA metric).

    Each plot shows mean node value of ``color_col`` on the x-axis against
    one of the three scaled topological metrics on the y-axis.  A linear
    OLS trend line, Pearson r, and p-value annotation are included.
    """
    from pathlib import Path
    save_dir = Path(save_dir)

    plotly_cmap = cmap["plotly"] if isinstance(cmap, dict) else cmap
    color_range = [vmin, vmax] if vmin is not None and vmax is not None else None

    metrics_map = {
        "Mean Entropy (Scaled)":           "Persistent Entropy",
        "Mean L2 Norm (Scaled)":           "L2 Norm",
        "Mean Total Persistence (Scaled)": "Total Persistence",
    }

    for raw_metric, short_name in metrics_map.items():
        valid = df_summary.dropna(subset=[color_col, raw_metric])
        corr_text = ""
        if len(valid) > 1:
            r_val, p_val = pearsonr(valid[color_col], valid[raw_metric])
            corr_text = (
                f"<sup style='color: black;'>"
                f"Pearson's r = {r_val:.3f}, p = {p_val:.3f}"
                f"</sup>"
            )
            print(f"  {short_name}: r = {r_val:.3f}, p = {p_val:.3f}")

        fig = px.scatter(
            df_summary, x=color_col, y=raw_metric,
            size="No. Provinces", color=color_col,
            color_continuous_scale=plotly_cmap,
            text="Node ID",
            range_color=color_range,
            trendline="ols",
            hover_name="Provinces Included",
            hover_data={"Node ID": False, color_col: ":.2f", raw_metric: ":.3f"},
            height=600, width=1200,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, opacity=1)
        fig.update_traces(
            textposition="top center",
            textfont=dict(family="serif", size=12, color="black"),
            marker=dict(line=dict(color="black", width=1)),
            selector=dict(mode="markers+text"),
        )
        fig.update_traces(
            line=dict(color="rgba(0, 0, 0, 0.5)", dash="dash", width=2),
            selector=dict(mode="lines"),
        )
        fig.update_layout(
            font=dict(family="sans-serif", color="black"),
            title=dict(
                text=corr_text, x=0.5, y=0.9,
                font=dict(size=16, color="black"),
            ),
            coloraxis_colorbar=dict(
                title="", ticksuffix="%",
                orientation="h", yanchor="top", y=-0.05,
                xanchor="center", x=0.5,
                thickness=15, len=0.93,
                tickfont=dict(color="black"),
            ),
            showlegend=False,
            yaxis=dict(
                range=[-9, 9],
                title=dict(
                    text=f"{short_name}<br><span style='font-size:11px'>(Standardized)</span>",
                    font=dict(size=12, color="black"),
                ),
                tickfont=dict(color="black"), zeroline=False,
                showgrid=True, gridcolor="#eaeaea",
                showline=True, linewidth=1, linecolor="black",
            ),
            xaxis=dict(
                title=dict(
                    text=f"<br><br>{color_col}",
                    font=dict(size=12, color="black"),
                ),
                tickfont=dict(color="black"), ticksuffix="%",
                showgrid=True, gridcolor="#eaeaea",
                showline=True, linewidth=1, linecolor="black",
            ),
            margin=dict(l=80, r=80, t=80, b=120),
            paper_bgcolor="white", plot_bgcolor="#ffffff",
        )

        safe_metric = short_name.lower().replace(" ", "_")
        out_path = save_dir / f"scatter_{mode_name}_{safe_metric}.jpg"
        fig.write_image(str(out_path), scale=2, engine="kaleido")
        print(f"   -> Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def generate_correlation_heatmap(
    df_features: pd.DataFrame,
    save_dir=BALLMAPPER_DIR,
) -> None:
    """Save a Seaborn heatmap of the three TDA feature Pearson correlations."""
    import seaborn as sns
    from pathlib import Path
    save_dir = Path(save_dir)

    cols = ["H0_L2_Norm", "H0_PersistentEntropy", "H0_Total_Persistence"]
    corr = df_features[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    labels = [
        "L2 Norm\n(Evolutionary Rigidity)",
        "Entropy\n(Evolutionary Complexity)",
        "Total Persistence\n(Fragmentation)",
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(labels, rotation=0, fontweight="bold")

    out_path = save_dir / "feature_correlation_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"-> Saved correlation heatmap: {out_path.name}")
