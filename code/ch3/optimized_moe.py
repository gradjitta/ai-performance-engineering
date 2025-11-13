"""Chapter 3 MoE optimization: sparse routing + CUDA graphs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 MoE example")
    return torch.device("cuda")


class SparseRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim * 2, hidden_dim, dtype=torch.float16))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, hidden_dim * 2, dtype=torch.float16))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, hidden = x.shape
        tokens = x.view(-1, hidden).half()
        logits = self.router(tokens)
        topk_vals, topk_idx = torch.topk(logits.float(), self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1).to(tokens.dtype)

        expanded_tokens = tokens.repeat_interleave(self.top_k, dim=0)
        token_ids = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        expert_indices = topk_idx.reshape(-1)
        weights_flat = weights.reshape(-1, 1)

        counts = torch.bincount(expert_indices, minlength=self.num_experts)
        max_count = counts.max().item()

        padded_tokens = torch.zeros(self.num_experts, max_count, hidden, dtype=tokens.dtype, device=tokens.device)
        padded_weights = torch.zeros(self.num_experts, max_count, 1, dtype=tokens.dtype, device=tokens.device)
        padded_indices = torch.full((self.num_experts, max_count), -1, dtype=torch.int64, device=tokens.device)

        for expert_id in range(self.num_experts):
            count = counts[expert_id].item()
            if count == 0:
                continue
            positions = torch.nonzero(expert_indices == expert_id, as_tuple=False).squeeze(-1)
            padded_tokens[expert_id, :count] = expanded_tokens[positions]
            padded_weights[expert_id, :count, 0] = weights_flat[positions, 0]
            padded_indices[expert_id, :count] = token_ids[positions]

        hidden_1 = torch.matmul(padded_tokens, self.w1.transpose(1, 2))
        hidden_1 = F.gelu(hidden_1)
        expert_outputs = torch.matmul(hidden_1, self.w2.transpose(1, 2))
        expert_outputs = expert_outputs * padded_weights

        output = torch.zeros_like(tokens, dtype=torch.float16)
        flat_outputs = expert_outputs.reshape(-1, hidden)
        flat_indices = padded_indices.reshape(-1)
        valid_mask = flat_indices >= 0
        output.index_add_(0, flat_indices[valid_mask], flat_outputs[valid_mask])

        return output.view(bsz, seq, hidden)


class OptimizedMoEBenchmark(Benchmark):
    """Sparse routed MoE captured inside a CUDA graph."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_dim = 768
        self.num_experts = 12
        self.top_k = 2
        self.batch = 4
        self.seq = 512
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(7)
        self.model = SparseRouter(self.hidden_dim, self.num_experts, self.top_k).to(self.device).half().eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_dim, device=self.device, dtype=torch.float16)
        with torch.no_grad():
            _ = self.model(self.inputs)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.autocast("cuda", dtype=torch.float16):
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=2)

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedMoEBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=4, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nSparse MoE latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
