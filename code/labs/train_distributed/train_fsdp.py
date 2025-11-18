"""Entry point for FSDP training demos (baseline vs optimized)."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_fsdp as baseline_run
import labs.train_distributed.optimized_fsdp as optimized_run


def main():
    parser = argparse.ArgumentParser(description="FSDP training examples.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Which variant to execute.",
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
