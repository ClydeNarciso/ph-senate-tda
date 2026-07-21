#!/usr/bin/env python3
"""
Command-line interface for the 2022 analysis pipeline.

Usage:
    python run.py                   # Full run (all diagnostics)
    python run.py --fast            # Skip step-wise stability
    python run.py --reps 100        # Reduce bootstrap repetitions for testing
    python run.py --fast --reps 100 # Combine options
    python run.py --help
"""
import argparse
import sys

import config
import main as pipeline_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 2022 electoral topology / socioeconomic analysis pipeline."
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Skip step-wise stability analysis. Useful for testing.',
    )
    parser.add_argument(
        '--reps', type=int, default=None,
        help='Override NUM_REPETITIONS (default: 1000).',
    )
    parser.add_argument(
        '--epsilon', type=float, default=None,
        help='Override BM_EPSILON (default: 0.6).',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.fast:
        config.RUN_STEPWISE_STABILITY = False
        print("[config] Step-wise stability DISABLED (--fast)")
    if args.reps is not None:
        config.NUM_REPETITIONS = args.reps
        print(f"[config] NUM_REPETITIONS = {args.reps}")
    if args.epsilon is not None:
        config.BM_EPSILON = args.epsilon
        print(f"[config] BM_EPSILON = {args.epsilon}")

    print(f"\nStarting 2022 pipeline with:")
    print(f"  Poverty year      : {config.POVERTY_COL_LABEL}")
    print(f"  Dynasty proxy year: {config.DYNASTY_YEAR}")
    print(f"  RUN_STEPWISE_STABILITY: {config.RUN_STEPWISE_STABILITY}")
    print(f"  NUM_REPETITIONS: {config.NUM_REPETITIONS}")
    print(f"  BM_EPSILON: {config.BM_EPSILON}")
    print()

    try:
        pipeline_main.main()
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nOutputs saved to:")
        print(f"  Figures : {config.FIGURES_DIR}")
        print(f"  Tables  : {config.TABLES_DIR}")
        print(f"  Cache   : {config.CACHE_DIR}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
