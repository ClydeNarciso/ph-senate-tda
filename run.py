#!/usr/bin/env python3
"""
run.py — Top-level runner for the ph-senate-tda project.

Placement
---------
This file belongs at the root of the project::

    ph-senate-tda/
        run.py          <- this file
        2016/src/
        2019/src/
        2022/src/
        2025/src/
        ZZPH/src/

Usage
-----
    # Run every pipeline in order
    python run.py

    # Run only specific years (space-separated)
    python run.py --years 2016 2019

    # Run only the ZZPH longitudinal pipeline
    python run.py --zzph-only

    # Run individual-year pipelines only (no ZZPH)
    python run.py --skip-zzph

    # Override reps and epsilon across all pipelines
    python run.py --reps 100 --fast

    # Quick smoke-test (100 reps, skip stepwise stability, skip ZZPH)
    python run.py --reps 100 --fast --skip-zzph
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

YEAR_ORDER  = [2016, 2019, 2022, 2025]
PROJECT_DIR = Path(__file__).resolve().parent


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _banner(text: str, width: int = 66) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _run_year_pipeline(
    year: int,
    fast: bool,
    reps: int | None,
    epsilon: float | None,
) -> bool:
    """Import and execute one year's main.py from its src/ subdirectory.

    Returns True on success, False on failure.
    """
    src_dir = PROJECT_DIR / str(year) / "src"
    if not src_dir.exists():
        print(f"  [!] src/ directory not found for {year}: {src_dir}")
        return False

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Import (or reimport) config and main for this year
    # Use importlib so repeated calls in the same process don't collide.
    try:
        cfg  = importlib.import_module("config")
        main = importlib.import_module("main")
    except Exception as exc:
        print(f"  [!] Import error for {year}: {exc}")
        traceback.print_exc()
        return False

    # Apply CLI overrides to the loaded config module
    if fast:
        if hasattr(cfg, "RUN_STEPWISE_STABILITY"):
            cfg.RUN_STEPWISE_STABILITY = False
            print(f"  [config] RUN_STEPWISE_STABILITY = False (--fast)")
    if reps is not None and hasattr(cfg, "NUM_REPETITIONS"):
        cfg.NUM_REPETITIONS = reps
        print(f"  [config] NUM_REPETITIONS = {reps}")
    if epsilon is not None and hasattr(cfg, "BM_EPSILON"):
        cfg.BM_EPSILON = epsilon
        print(f"  [config] BM_EPSILON = {epsilon}")

    try:
        main.main()
    except Exception as exc:
        print(f"  [!] Pipeline error for {year}: {exc}")
        traceback.print_exc()
        return False
    finally:
        # Remove this year's src/ from sys.path so the next year's modules
        # don't shadow it, then evict all modules loaded from it.
        if str(src_dir) in sys.path:
            sys.path.remove(str(src_dir))
        to_remove = [
            mod for mod, obj in sys.modules.items()
            if obj is not None
            and hasattr(obj, "__file__")
            and obj.__file__ is not None
            and str(src_dir) in obj.__file__
        ]
        for mod in to_remove:
            del sys.modules[mod]

    return True


def _run_zzph_pipeline(
    reps: int | None,
    epsilon: float | None,
) -> bool:
    """Import and execute ZZPH/src/main.py."""
    src_dir = PROJECT_DIR / "ZZPH" / "src"
    if not src_dir.exists():
        print(f"  [!] ZZPH src/ directory not found: {src_dir}")
        return False

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        cfg  = importlib.import_module("config")
        main = importlib.import_module("main")
    except Exception as exc:
        print(f"  [!] Import error for ZZPH: {exc}")
        traceback.print_exc()
        return False

    if reps is not None and hasattr(cfg, "NUM_REPETITIONS"):
        cfg.NUM_REPETITIONS = reps
        print(f"  [config] NUM_REPETITIONS = {reps}")
    if epsilon is not None and hasattr(cfg, "BM_EPSILON"):
        cfg.BM_EPSILON = epsilon
        print(f"  [config] BM_EPSILON = {epsilon}")

    try:
        main.main()
    except Exception as exc:
        print(f"  [!] ZZPH pipeline error: {exc}")
        traceback.print_exc()
        return False
    finally:
        if str(src_dir) in sys.path:
            sys.path.remove(str(src_dir))
        to_remove = [
            mod for mod, obj in sys.modules.items()
            if obj is not None
            and hasattr(obj, "__file__")
            and obj.__file__ is not None
            and str(src_dir) in obj.__file__
        ]
        for mod in to_remove:
            del sys.modules[mod]

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ph-senate-tda pipeline suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--years", nargs="+", type=int, choices=YEAR_ORDER, default=YEAR_ORDER,
        help="Which year pipelines to run (default: all four).",
    )
    p.add_argument(
        "--skip-zzph", action="store_true",
        help="Skip the ZZPH longitudinal pipeline.",
    )
    p.add_argument(
        "--zzph-only", action="store_true",
        help="Run only the ZZPH pipeline (skip all year pipelines).",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Set RUN_STEPWISE_STABILITY=False in each year config.",
    )
    p.add_argument(
        "--reps", type=int, default=None,
        help="Override NUM_REPETITIONS in every pipeline.",
    )
    p.add_argument(
        "--epsilon", type=float, default=None,
        help="Override BM_EPSILON in every pipeline.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    _banner("ph-senate-tda  |  Full Pipeline Runner")
    print(f"Project root : {PROJECT_DIR}")
    print(f"Years        : {args.years}")
    print(f"ZZPH         : {'only' if args.zzph_only else ('skip' if args.skip_zzph else 'yes')}")
    print(f"Fast mode    : {args.fast}")
    print(f"Reps override: {args.reps}")
    print(f"Eps override : {args.epsilon}")

    results: dict[str, bool] = {}
    wall_start = time.time()

    # ---- Individual year pipelines ----------------------------------------
    if not args.zzph_only:
        for year in args.years:
            _banner(f"YEAR PIPELINE: {year}")
            t0 = time.time()
            ok = _run_year_pipeline(year, args.fast, args.reps, args.epsilon)
            elapsed = time.time() - t0
            label = str(year)
            results[label] = ok
            status = "DONE" if ok else "FAILED"
            print(f"\n  [{status}] {year} pipeline — {_hms(elapsed)}")

    # ---- ZZPH longitudinal pipeline ----------------------------------------
    if not args.skip_zzph:
        _banner("ZZPH LONGITUDINAL PIPELINE")
        t0 = time.time()
        ok = _run_zzph_pipeline(args.reps, args.epsilon)
        elapsed = time.time() - t0
        results["ZZPH"] = ok
        status = "DONE" if ok else "FAILED"
        print(f"\n  [{status}] ZZPH pipeline — {_hms(elapsed)}")

    # ---- Summary -----------------------------------------------------------
    total_elapsed = time.time() - wall_start
    _banner("RUN SUMMARY")
    all_ok = True
    for label, ok in results.items():
        tick = "\u2713" if ok else "\u2717"
        print(f"  [{tick}]  {label}")
        if not ok:
            all_ok = False
    print(f"\n  Total wall time: {_hms(total_elapsed)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
