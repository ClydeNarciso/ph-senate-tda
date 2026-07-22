"""Plotting: BallMapper-node scatterplots against socioeconomic covariates,
and the raw topological-feature correlation heatmap."""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import display, Image
from scipy.stats import pearsonr

from config import FIGURES_DIR


def generate_topology_relationship_plots(df_bm_summary: pd.DataFrame, color_col: str, cmap, mode_name: str):
    """Generates scatterplots mapping ballmapper nodes against topological metrics, including r & p values."""
    if color_col in ['Dominant_Dialect']:
        return

    clean_target_col = f'Mean {color_col}'
    if clean_target_col not in df_bm_summary.columns:
        return

    # Safely format colormaps for plotly
    if isinstance(cmap, mcolors.Colormap):
        if isinstance(cmap, mcolors.ListedColormap):
            plotly_cmap = [mcolors.to_hex(c) for c in cmap.colors]
        else:
            plotly_cmap = [[i/255.0, mcolors.to_hex(cmap(i/255.0))] for i in range(256)]
    else:
        plotly_cmap = cmap

    metrics_map = {
        'Mean Entropy (Scaled)': "Persistent Entropy",
        'Mean L2 Norm (Scaled)': "L2 Norm",
        'Mean Total Persistence (Scaled)': "Total Persistence"
    }

    for raw_metric, plot_title in metrics_map.items():
        # Calculate Pearson Correlation for the nodes
        valid_df = df_bm_summary.dropna(subset=[clean_target_col, raw_metric])
        if len(valid_df) > 1:
            r_val, p_val = pearsonr(valid_df[clean_target_col], valid_df[raw_metric])
            corr_text = f"<br><sup>Pearson r = {r_val:.3f} (p = {p_val:.3f})</sup>"
        else:
            corr_text = ""

        # Determine if values are raw fractions (0 to 1) or pre-multiplied percentages (0 to 100)
        is_fraction = df_bm_summary[clean_target_col].max() <= 1.0

        if is_fraction:
            hover_fmt = ':.1f%'
            xaxis_config = dict(title=clean_target_col, title_font_size=16, tickformat='.0%')
            cbar_config = dict(title='', tickformat='.0%', orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5, thickness=15, len=0.93)
        else:
            hover_fmt = ':.1f'
            xaxis_config = dict(title=clean_target_col, title_font_size=16, ticksuffix='%')
            cbar_config = dict(title='', ticksuffix='%', orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5, thickness=15, len=0.93)

        fig = px.scatter(
            df_bm_summary, x=clean_target_col, y=raw_metric,
            size='No. Provinces',
            color=clean_target_col, color_continuous_scale=plotly_cmap,
            text='Node ID', trendline='ols',
            hover_name='Provinces Included',
            hover_data={'Node ID': False, clean_target_col: hover_fmt, raw_metric: ':.3f'},
            height=600, width=1200,
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1, opacity=1)
        fig.update_traces(textposition='top center', textfont=dict(family="serif", size=12, color='black'), selector=dict(mode="markers+text"))
        fig.update_traces(line=dict(color="rgba(0, 0, 0, 0.5)", dash="dash", width=2), selector=dict(mode="lines"))

        fig.update_layout(
            font_family='serif',
            title=dict(
                text=f'<b>{plot_title} vs. {color_col}</b>{corr_text}',
                x=0.5, y=0.95, font_size=20,
            ),
            coloraxis_colorbar=cbar_config,
            showlegend=False,
            yaxis=dict(range=[-6, 9], title='Z-score value',
                       title_font_size=14, zeroline=False),
            xaxis=xaxis_config,
            margin=dict(l=80, r=80, t=80, b=120),
            paper_bgcolor='white', plot_bgcolor='#f9f9f9',
        )

        safe_metric_name = plot_title.lower().replace(' ', '_')
        base_filename = FIGURES_DIR / f"scatter_{mode_name}_{safe_metric_name}"
        png_path = f"{base_filename}.png"
        fig.write_image(png_path, scale=2, engine="kaleido")
        # try:
        #     display(Image(filename=png_path))
        # except NameError:
        #     # display() not available in non-Jupyter environments
        #     print(f"  Saved: {png_path}")


def generate_correlation_heatmap(df_features: pd.DataFrame) -> None:
    print("\n--- GENERATING FEATURE CORRELATION HEATMAP ---")
    cols = ['H0_L2_Norm', 'H0_PersistentEntropy', 'H0_Total_Persistence']
    corr = df_features[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        ax=ax,
    )
    clean_labels = [
        'L2 Norm\n(Entrenchment)',
        'Entropy\n(Complexity)',
        'Total Persistence\n(Fragmentation)',
    ]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(clean_labels, rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
