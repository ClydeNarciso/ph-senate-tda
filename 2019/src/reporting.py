"""Styled display and LaTeX export of BallMapper node-summary tables."""
from pathlib import Path

import pandas as pd
from IPython.display import display

from config import SHARP_CMAP, SCALED_COLS


def display_clean_summary(df: pd.DataFrame, color_col: str) -> None:
    majority_col = f'Majority {color_col}'
    mean_col     = f'Mean {color_col}'
    is_categorical = majority_col in df.columns
    target_col = majority_col if is_categorical else mean_col

    max_abs = df[SCALED_COLS].abs().max().max() if not df[SCALED_COLS].empty else 1.0

    styled = df.style
    if not is_categorical and target_col in df.columns:
        styled = styled.background_gradient(cmap='RdYlBu_r', subset=[target_col])
    if 'Admin Share (%)' in df.columns:
        styled = styled.background_gradient(cmap='coolwarm', subset=['Admin Share (%)'])
    for col in SCALED_COLS:
        if col in df.columns:
            styled = styled.background_gradient(
                cmap=SHARP_CMAP, subset=[col],
                vmin=-max_abs, vmax=max_abs,
            )

    fmt = {
        'Mean Entropy (Raw)': '{:.4f}',
        'Mean Entropy (Scaled)': '{:.4f}',
        'Mean L2 Norm (Raw)': '{:.4f}',
        'Mean L2 Norm (Scaled)': '{:.4f}',
        'Mean Total Persistence (Raw)': '{:.4f}',
        'Mean Total Persistence (Scaled)': '{:.4f}',
    }
    if not is_categorical and target_col in df.columns:
        fmt[target_col] = '{:.4f}'

    styled = (
        styled.format(fmt)
        .set_properties(**{'text-align': 'left', 'vertical-align': 'top'})
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
    )
    if hasattr(styled, 'hide'):
        styled = styled.hide(axis="index")
    display(styled)


def export_summary_to_latex(df: pd.DataFrame,
                             filepath: Path,
                             color_col: str) -> None:
    majority_col = f'Majority {color_col}'
    mean_col     = f'Mean {color_col}'
    is_categorical = majority_col in df.columns
    target_col = majority_col if is_categorical else mean_col

    styled = df.style
    if not is_categorical and target_col in df.columns:
        styled = styled.background_gradient(cmap='coolwarm', subset=[target_col])
    if target_col != 'Admin Share (%)' and 'Admin Share (%)' in df.columns:
        styled = styled.background_gradient(cmap='coolwarm', subset=['Admin Share (%)'])

    fmt = {
        'Mean Entropy (Raw)': '{:.4f}',
        'Mean Entropy (Scaled)': '{:.4f}',
        'Mean L2 Norm (Raw)': '{:.4f}',
        'Mean L2 Norm (Scaled)': '{:.4f}',
        'Mean Total Persistence (Raw)': '{:.4f}',
        'Mean Total Persistence (Scaled)': '{:.4f}',
    }
    if not is_categorical and target_col not in ('Admin Share (%)',):
        fmt[target_col] = '{:.4f}'

    with open(filepath, 'w') as f:
        f.write(
            styled.format(fmt)
            .hide(axis="index")
            .to_latex(convert_css=True, hrules=True)
        )
