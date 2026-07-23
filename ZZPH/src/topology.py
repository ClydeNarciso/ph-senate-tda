"""
topology.py — Zigzag persistent homology engine for the ZZPH pipeline.

Implements the seven-step filtration:

  X16  →  X16∪X19  ←  X19  →  X19∪X22  ←  X22  →  X22∪X25  ←  X25
   t=0       t=1       t=2       t=3       t=4       t=5       t=6

Each ``Xᵢ`` is a Vietoris-Rips complex built from the CLR simplex vectors
of one province across all precincts recorded in election year i.  The
union complexes are built on the joint point cloud without duplication.

Per-province radii are optimised via k-NN distance percentiles so that the
filtration captures genuine topological change rather than noise.

All intermediate results are cached on disk so the pipeline can be resumed
after interruption.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import gudhi as gd
import dionysus as d
from gudhi.representations import Landscape
from persim.persistent_entropy import persistent_entropy

from config import (
    FEATURE_COLS,
    KNN_K,
    KNN_MIN_RADIUS,
    KNN_PERCENTILE,
    LANDSCAPE_NUM,
    LANDSCAPE_RANGE,
    LANDSCAPE_RES,
    RADIUS_CACHE_FILE,
    FINAL_FEATURES_FILE,
    ZIGZAG_MAX_TIME,
)


# ---------------------------------------------------------------------------
# Radius optimisation
# ---------------------------------------------------------------------------

def get_optimal_radius(
    province: str,
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
    cols: list[str] = FEATURE_COLS,
    k: int = KNN_K,
    percentile: float = KNN_PERCENTILE,
    min_radius: float = KNN_MIN_RADIUS,
) -> float:
    """Compute the k-NN distance percentile radius for a single province.

    The global point cloud is the union of all four election years so that
    the radius is consistent across the full zigzag filtration.
    """
    frames = [
        df[df["Province"] == province][cols].values
        for df in (df_16, df_19, df_22, df_25)
    ]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return min_radius

    X_global = np.vstack(frames)
    effective_k = min(k, len(X_global) - 1)
    if effective_k < 1:
        return min_radius

    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto")
    distances, _ = nbrs.fit(X_global).kneighbors(X_global)
    return max(float(np.percentile(distances[:, effective_k], percentile)), min_radius)


def build_radius_cache(
    common_provinces: list[str],
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
    cache_path: Path = RADIUS_CACHE_FILE,
) -> dict[str, float]:
    """Load or compute the per-province optimal radius dictionary."""
    radii: dict[str, float] = {}
    if cache_path.exists():
        with open(cache_path) as fh:
            radii = json.load(fh)

    missing = [p for p in common_provinces if p not in radii]
    if missing:
        print(f"Computing radii for {len(missing)} province(s)…")
        for prov in tqdm(missing, desc="Optimising Epsilon Radii"):
            radii[prov] = get_optimal_radius(prov, df_16, df_19, df_22, df_25)
        with open(cache_path, "w") as fh:
            json.dump(radii, fh, indent=4)

    return radii


# ---------------------------------------------------------------------------
# Zigzag filtration
# ---------------------------------------------------------------------------

def _get_simplices(
    X_global: np.ndarray,
    idx_list: list[int],
    max_radius: float,
) -> set[tuple[int, ...]]:
    """Build a Vietoris-Rips 1-skeleton on a subset of the global point cloud."""
    if not idx_list:
        return set()
    rc = gd.RipsComplex(
        points=X_global[idx_list], max_edge_length=max_radius
    )
    st = rc.create_simplex_tree(max_dimension=1)
    return {
        tuple(sorted(idx_list[v] for v in s))
        for s, _ in st.get_skeleton(1)
    }


def run_provincial_zigzag(
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
    province: str,
    cols: list[str],
    max_radius: float,
) -> object:
    """Run the seven-step zigzag filtration for one province.

    Returns
    -------
    dgms : dionysus diagram collection
        Zigzag persistence diagrams (H0 in ``dgms[0]``).
    """
    X_list = [
        df[df["Province"] == province][cols].values
        for df in (df_16, df_19, df_22, df_25)
    ]
    X_global = np.vstack(X_list)

    # Build consecutive global index ranges for each year
    indices: list[list[int]] = []
    cur = 0
    for X in X_list:
        indices.append(list(range(cur, cur + len(X))))
        cur += len(X)

    # Seven-step zigzag: individual → union → individual → …
    steps = [
        _get_simplices(X_global, indices[0],                      max_radius),
        _get_simplices(X_global, indices[0] + indices[1],         max_radius),
        _get_simplices(X_global, indices[1],                      max_radius),
        _get_simplices(X_global, indices[1] + indices[2],         max_radius),
        _get_simplices(X_global, indices[2],                      max_radius),
        _get_simplices(X_global, indices[2] + indices[3],         max_radius),
        _get_simplices(X_global, indices[3],                      max_radius),
    ]

    # Map each simplex to the time steps where it is active
    simplex_to_steps: dict[tuple, list[int]] = defaultdict(list)
    for t, step_set in enumerate(steps):
        for s in step_set:
            simplex_to_steps[s].append(t)

    # Build dionysus time intervals (consecutive run-length encoding)
    times: list[list[int]] = []
    sorted_simplices = sorted(simplex_to_steps.keys(), key=len)

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
    return dgms


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_topological_metrics(dgms: object) -> dict[str, float]:
    """Summarise H0 zigzag persistence diagrams into three scalar features.

    Features
    --------
    H0_PersistentEntropy
        Normalised Shannon entropy of H0 bar lifetimes — measures
        distributional complexity of the connected-component structure.
    H0_L2_Norm
        L2 norm of the persistence landscape vector — measures
        evolutionary rigidity (large = topology barely changes).
    H0_Total_Persistence
        Sum of all H0 bar lifetimes — measures overall fragmentation
        across the four election cycles.
    """
    raw = [
        [p.birth, ZIGZAG_MAX_TIME if (math.isinf(p.death) or p.death >= ZIGZAG_MAX_TIME) else p.death]
        for p in dgms[0]
        if (ZIGZAG_MAX_TIME if (math.isinf(p.death) or p.death >= ZIGZAG_MAX_TIME) else p.death) > p.birth
    ]
    if not raw:
        return {
            "H0_PersistentEntropy": 0.0,
            "H0_L2_Norm":          0.0,
            "H0_Total_Persistence": 0.0,
        }

    dgms_np   = np.array(raw)
    lifespans = dgms_np[:, 1] - dgms_np[:, 0]
    entropy   = float(persistent_entropy(dgms_np, normalize=True)[0])

    ld = Landscape(
        num_landscapes=LANDSCAPE_NUM,
        resolution=LANDSCAPE_RES,
        sample_range=LANDSCAPE_RANGE,
    )
    vec   = ld.fit_transform([dgms_np])[0]
    width = (LANDSCAPE_RANGE[1] - LANDSCAPE_RANGE[0]) / LANDSCAPE_RES
    l2    = float(np.sqrt(np.sum(vec ** 2) * width))

    return {
        "H0_PersistentEntropy":  entropy,
        "H0_L2_Norm":            l2,
        "H0_Total_Persistence":  float(np.sum(lifespans)),
    }


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------

def _process_single_province(
    province: str,
    radius: float,
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
) -> dict:
    """Worker function — safe wrapper around the zigzag engine."""
    try:
        dgms    = run_provincial_zigzag(df_16, df_19, df_22, df_25, province, FEATURE_COLS, radius)
        metrics = extract_topological_metrics(dgms)
        metrics["Province"] = province
        return metrics
    except Exception as exc:  # noqa: BLE001
        return {"Province": province, "error": str(exc)}


def compute_zigzag_features(
    common_provinces: list[str],
    radii: dict[str, float],
    df_16: pd.DataFrame,
    df_19: pd.DataFrame,
    df_22: pd.DataFrame,
    df_25: pd.DataFrame,
    cache_path: Path = FINAL_FEATURES_FILE,
) -> pd.DataFrame:
    """Load or compute the province-level zigzag feature DataFrame.

    All computation is parallelised over provinces using joblib.
    Results are cached to ``cache_path`` after the first run.
    """
    if cache_path.exists():
        print("\n-> Loading cached Longitudinal Zigzag Features…")
        return pd.read_csv(cache_path)

    print(
        f"\nExecuting Zigzag Pipeline for {len(common_provinces)} provinces "
        f"(parallelised)…"
    )
    results = Parallel(n_jobs=-1)(
        delayed(_process_single_province)(
            prov, radii[prov], df_16, df_19, df_22, df_25
        )
        for prov in tqdm(common_provinces, desc="Topological Feature Extraction")
    )

    records = []
    for res in results:
        if "error" in res:
            print(f"  [!] Failure on {res['Province']}: {res['error']}")
        else:
            records.append(res)

    df_topo = pd.DataFrame(records)
    df_topo.to_csv(cache_path, index=False)
    print(f"-> Saved zigzag features to {cache_path.name}")
    return df_topo
