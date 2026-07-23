"""
visualization.py — Synthetic zigzag persistence demo figure for the ZZPH
pipeline.

Produces a two-row publication figure:
  Row 1 — five Vietoris-Rips complexes arranged left-to-right with
           inclusion arrows (→ ← → ← ) between them.
  Row 2 — H0 barcode spanning all seven zigzag time steps.

This module is self-contained (does not depend on any real election data)
and is intended to illustrate the zigzag filtration concept in papers and
presentations.  Run it directly::

    python visualization.py

or call ``plot_zigzag_demo()`` from a notebook.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import gudhi as gd
import dionysus as d

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

def _spread_cluster(
    center: np.ndarray,
    spread: np.ndarray,
    num_pts: int,
    min_dist: float = 0.4,
    seed_offset: int = 0,
) -> np.ndarray:
    """Generate a spatially spread cluster with a minimum inter-point distance."""
    rng = np.random.default_rng(1 + seed_offset)
    pts: list[np.ndarray] = []
    attempts = 0
    while len(pts) < num_pts and attempts < 2000:
        candidate = rng.uniform(low=center - spread, high=center + spread)
        if not pts or all(np.linalg.norm(candidate - p) >= min_dist for p in pts):
            pts.append(candidate)
        attempts += 1
    return np.array(pts)


def generate_spread_random_data() -> list[np.ndarray]:
    """Return three well-separated 2-D point clusters."""
    np.random.seed(1)
    X1 = _spread_cluster(np.array([-1.0,  0.0]), np.array([0.6, 1.4]), 4, seed_offset=0)
    X2 = _spread_cluster(np.array([ 0.0,  0.0]), np.array([0.5, 1.1]), 4, seed_offset=1)
    X3 = _spread_cluster(np.array([ 1.0,  0.0]), np.array([0.6, 1.3]), 4, seed_offset=2)
    return [X1, X2, X3]


# ---------------------------------------------------------------------------
# 2. Zigzag persistence (H0 only)
# ---------------------------------------------------------------------------

def _get_simplices(
    X_global: np.ndarray,
    idx_list: list[int],
    max_radius: float,
) -> set[tuple[int, ...]]:
    if not idx_list:
        return set()
    rc = gd.RipsComplex(points=X_global[idx_list], max_edge_length=max_radius)
    st = rc.create_simplex_tree(max_dimension=1)
    return {
        tuple(sorted(idx_list[v] for v in s))
        for s, _ in st.get_skeleton(1)
    }


def compute_zigzag_persistence_h0(
    X_list: list[np.ndarray],
    max_radius: float,
) -> tuple[object, list[set], np.ndarray]:
    """Compute zigzag H0 persistence for three clusters.

    Filtration: X1 → X1∪X2 ← X2 → X2∪X3 ← X3

    Returns
    -------
    dgms : dionysus diagram collection
    steps : list of five simplex sets (one per complex)
    X_global : stacked global point cloud
    """
    X_global = np.vstack(X_list)
    indices: list[list[int]] = []
    cur = 0
    for X in X_list:
        indices.append(list(range(cur, cur + len(X))))
        cur += len(X)

    steps = [
        _get_simplices(X_global, indices[0],                      max_radius),
        _get_simplices(X_global, indices[0] + indices[1],         max_radius),
        _get_simplices(X_global, indices[1],                      max_radius),
        _get_simplices(X_global, indices[1] + indices[2],         max_radius),
        _get_simplices(X_global, indices[2],                      max_radius),
    ]

    simplex_to_steps: dict[tuple, list[int]] = defaultdict(list)
    for t, step_set in enumerate(steps):
        for s in step_set:
            simplex_to_steps[s].append(t)

    sorted_simplices = sorted(simplex_to_steps.keys(), key=len)
    times: list[list[int]] = []
    for s in sorted_simplices:
        active = simplex_to_steps[s]
        intervals: list[int] = []
        if not active:
            times.append(intervals)
            continue
        start = active[0]
        for i in range(1, len(active)):
            if active[i] != active[i - 1] + 1:
                intervals.extend([start, active[i - 1] + 1])
                start = active[i]
        intervals.extend([start, active[-1] + 1])
        times.append(intervals)

    f = d.Filtration(sorted_simplices)
    _, dgms, _ = d.zigzag_homology_persistence(f, times)
    return dgms, steps, X_global


# ---------------------------------------------------------------------------
# 3. Publication figure
# ---------------------------------------------------------------------------

def plot_zigzag_demo(
    X_list: list[np.ndarray] | None = None,
    max_radius: float = 0.8,
    save_path: str | None = None,
) -> None:
    """Draw the two-row zigzag demo figure.

    Parameters
    ----------
    X_list:
        Three point-cloud arrays.  Uses ``generate_spread_random_data()``
        when *None*.
    max_radius:
        Vietoris-Rips edge-length threshold.
    save_path:
        If given, the figure is saved here (PNG/JPG/PDF) instead of shown.
    """
    if X_list is None:
        X_list = generate_spread_random_data()

    dgms, steps, X_global = compute_zigzag_persistence_h0(X_list, max_radius)

    plt.rcParams["font.family"] = "serif"
    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 0.7], hspace=0.35)

    # --- Row 1: five complexes interspersed with four arrow subplots ---
    gs_top = gs[0, 0].subgridspec(1, 9, wspace=0.1)

    titles         = [r"$X_1$", r"$X_1 \cup X_2$", r"$X_2$", r"$X_2 \cup X_3$", r"$X_3$"]
    diagram_cols   = [0, 2, 4, 6, 8]
    arrow_cols     = [1, 3, 5, 7]

    for plot_i, step_idx in enumerate(diagram_cols):
        ax = fig.add_subplot(gs_top[0, step_idx])
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.8, 1.8)
        ax.axis("off")
        ax.set_title(titles[plot_i], fontsize=16, fontweight="bold", pad=12)

        for s in steps[plot_i]:
            if len(s) == 2:
                p1, p2 = X_global[s[0]], X_global[s[1]]
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color="#1f77b4", lw=1.4, alpha=0.75, zorder=1,
                )

        nodes = [s[0] for s in steps[plot_i] if len(s) == 1]
        if nodes:
            pts = X_global[nodes]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                color="#0f3c5f", s=40, zorder=2,
                edgecolors="white", lw=0.6,
            )

    for arr_i, step_idx in enumerate(arrow_cols):
        ax = fig.add_subplot(gs_top[0, step_idx])
        ax.axis("off")
        # Even indices = forward inclusion (→), odd = backward (←)
        arrow = r"$\hookrightarrow$" if arr_i % 2 == 0 else r"$\hookleftarrow$"
        txt = ax.text(
            0.5, 0.5, arrow,
            fontsize=36, fontweight="bold", color="black",
            ha="center", va="center", clip_on=False, zorder=10,
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=4, foreground="white"),
            path_effects.Normal(),
        ])

    # --- Row 2: H0 barcode ---
    ax_bar = fig.add_subplot(gs[1, :])
    h0_intervals = sorted(
        [p for p in dgms[0] if p.birth < p.death],
        key=lambda x: (x.birth, -(x.death - x.birth)),
    )
    for y_off, p in enumerate(h0_intervals):
        ax_bar.hlines(
            y=-y_off, xmin=p.birth, xmax=p.death - 1,
            color="#1f77b4", lw=4, capstyle="round",
        )

    ax_bar.set_xlim(-0.2, 4.2)
    ax_bar.set_xticks([0, 1, 2, 3, 4])
    ax_bar.set_xticklabels(titles, fontsize=13)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel(r"$H_0$ Components", fontsize=12)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4, color="#cccccc")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"-> Saved zigzag demo figure to {save_path}")
    else:
        # plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_zigzag_demo(save_path="zigzag_h0_demo.png")
