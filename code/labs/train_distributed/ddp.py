"""Dispatcher for baseline vs optimized DDP runs."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_ddp as baseline_run
import labs.train_distributed.optimized_ddp as optimized_run


def main():
    parser = argparse.ArgumentParser(description="DDP training examples.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Which variant to execute.",
    )
    args, remaining = parser.parse_known_args()

    # Let the chosen script parse its own CLI flags.
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
