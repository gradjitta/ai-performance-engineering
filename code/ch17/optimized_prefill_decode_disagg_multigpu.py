"""Optimized prefill/decode wrapper that skips when <2 GPUs are available."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class _SkipBenchmark(BaseBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=0)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "prefill_decode_disag.batch_size": float(getattr(self, 'batch_size', 0)),
            "prefill_decode_disag.seq_len": float(getattr(self, 'seq_len', 0)),
            "prefill_decode_disag.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: prefill/decode optimized multigpu requires >=2 GPUs")


def get_benchmark() -> BaseBenchmark:
    if torch.cuda.device_count() < 2:
        return _SkipBenchmark()
    from ch17.optimized_prefill_decode_disagg import OptimizedDisaggregatedBenchmark
    return OptimizedDisaggregatedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(benchmark)
    print(f"Prefill/decode optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
