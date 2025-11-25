"""optimized_attention.py - Tensor-core optimized attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


class OptimizedAttentionBenchmark(BaseBenchmark):
    """FP16 attention to leverage tensor cores."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.seq_len = 128
        self.hidden_dim = 256
        self.num_heads = 8

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)

        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        ).to(self.device).to(torch.float16).eval()

        self.input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._synchronize()
        tokens = float(self.batch_size * self.seq_len)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.batch_size),
        )

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self.input is not None
        with self._nvtx_range("optimized_attention"):
            with torch.no_grad():
                _output, _ = self.model(self.input, self.input, self.input)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.input = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
    "attention.estimated_flops": flops,
    "attention.estimated_bytes": bytes_moved,
    "attention.arithmetic_intensity": arithmetic_intensity,
}

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> OptimizedAttentionBenchmark:
    return OptimizedAttentionBenchmark()
