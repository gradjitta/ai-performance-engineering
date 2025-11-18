"""Baseline FSDP example with minimal tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from labs.train_distributed.utils import (
    ThroughputTracker,
    create_collate_fn,
    get_model_flops_per_token,
    load_tinystories,
    setup_tokenizer,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark

# Modern, lightweight causal LM; open weights.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        reshard_after_forward=True,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=True,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision="bf16")

    tokenizer = setup_tokenizer(MODEL_ID)
    dataset = load_tinystories(tokenizer, args.sequence_length, accelerator)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(),
    )

    config = AutoConfig.from_pretrained(MODEL_ID, use_cache=False)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    total_steps = min(args.steps, len(dataloader))
    flop_per_token = get_model_flops_per_token(model.config, args.sequence_length)
    tracker = ThroughputTracker(warmup_steps=5)

    for step, batch in enumerate(dataloader):
        if step >= total_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        metrics = tracker.step(batch["input_ids"].numel(), flop_per_token)
        if accelerator.is_main_process:
            msg = f"[baseline-fsdp] step {step}/{total_steps} loss={loss.item():.4f}"
            if metrics:
                msg += ThroughputTracker.format(metrics)
            if step % 10 == 0 or step == total_steps - 1:
                accelerator.print(msg)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("[baseline-fsdp] training completed")


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "train_fsdp.py",
        base_args=["--mode", "baseline"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:fsdp",
        default_nproc_per_node=None,
        name="baseline_fsdp",
    )
