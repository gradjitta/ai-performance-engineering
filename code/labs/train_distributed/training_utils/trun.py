#!/usr/bin/env python
"""Tiny helper around torch.distributed.run with friendlier flags."""

import os
import sys
from pathlib import Path

import click
from torch.distributed.run import get_args_parser, run


@click.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("-ng", "--num-gpus", required=True, help="Number of GPUs to use.")
@click.option("-ids", "--device-ids", help="Comma separated list of GPU ids to use.")
def trun(script, num_gpus, device_ids):
    # Ensure the project root is visible to the launched workers.
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)

    cli_args = ["--nproc_per_node", str(num_gpus), script]
    if device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    args = get_args_parser().parse_args(cli_args)
    run(args)


if __name__ == "__main__":
    trun()
