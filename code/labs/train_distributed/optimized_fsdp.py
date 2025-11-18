"""Optimized FSDP example layering float8, activation checkpointing, and torch.compile."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig
from transformers import AutoConfig, AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, FullyShardedDataParallelPlugin, TorchDynamoPlugin, set_seed
from labs.train_distributed.utils import (
    ThroughputTracker,
    create_collate_fn,
    get_model_flops_per_token,
    gpu_memory_usage,
    load_tinystories,
    setup_tokenizer,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark

# Modern, lightweight causal LM that ships standard kernels (open weights).
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--float8", action="store_true", help="Enable float8 linear layers via torchao.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the sharded model.")
    return parser.parse_args()


def _fused_adamw(params, lr):
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def main():
    args = parse_args()
    set_seed(777)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        activation_checkpointing_policy=None,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        reshard_after_forward=True,
        use_orig_params=True,
        sync_module_states=True,
    )

    handlers = [AORecipeKwargs(float8_recipe=Float8LinearConfig())] if args.float8 else []
    accelerator = Accelerator(
        dynamo_plugin=TorchDynamoPlugin(mode="reduce-overhead"),
        fsdp_plugin=fsdp_plugin,
        mixed_precision="bf16",
        kwargs_handlers=handlers,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if accelerator.num_processes < 2:
        accelerator.print("Warning: world_size < 2; running without true sharding (throughput/memory deltas will be minimal).")

    tokenizer = setup_tokenizer(MODEL_ID)
    dataset = load_tinystories(tokenizer, args.sequence_length, accelerator)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=create_collate_fn(),
    )

    config = AutoConfig.from_pretrained(MODEL_ID, use_cache=False, attn_implementation="eager")
    config.gradient_checkpointing = True

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="eager")
    optimizer = _fused_adamw(model.parameters(), args.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    total_steps = min(args.steps, len(dataloader))
    flop_per_token = get_model_flops_per_token(model.config, args.sequence_length)
    tracker = ThroughputTracker(warmup_steps=10)

    for step, batch in enumerate(dataloader):
        if step >= total_steps:
            break

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

        accelerator.backward(loss)

        if (step + 1) % args.grad_accum == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = tracker.step(batch["input_ids"].numel(), flop_per_token)
        if "warmup_done" in metrics and accelerator.is_main_process:
            accelerator.print("[optimized-fsdp] warmup finished; collecting steady-state metrics")
        elif metrics and accelerator.is_main_process and step % 5 == 0:
            memory = gpu_memory_usage(accelerator.local_process_index)
            metrics.update(memory)
            msg = (
                f"[optimized-fsdp] step {step}/{total_steps} loss={loss.item():.4f}"
                + ThroughputTracker.format(metrics, include_memory=True)
            )
            accelerator.print(msg)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("[optimized-fsdp] training completed")


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "train_fsdp.py",
        base_args=["--mode", "optimized"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:fsdp",
        default_nproc_per_node=None,
        name="optimized_fsdp",
    )
