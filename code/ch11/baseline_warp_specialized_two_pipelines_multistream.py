"""Baseline dual-pipeline warp specialization benchmark (Chapter 11)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from common.python.hardware_capabilities import ensure_dsmem_supported  # noqa: E402

try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:
    def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
        pass

_EXT_NAME = "baseline_warp_specialized_two_pipelines_ext"


def _get_build_dir() -> Path:
    """Get the torch extension build directory."""
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    repo_root = Path(__file__).resolve().parents[1]
    if base:
        return Path(base) / _EXT_NAME
    return repo_root / ".torch_extensions" / _EXT_NAME


@lru_cache(maxsize=1)
def _load_baseline_extension():
    # Clean stale builds before attempting to load
    build_dir = _get_build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_build_directory(build_dir)
    
    sources = [
        Path(__file__).with_name("baseline_warp_specialized_two_pipelines_extension.cu"),
    ]
    return load(
        name=_EXT_NAME,
        sources=[str(src) for src in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class BaselineDualPipelineBenchmark(BaseBenchmark):
    """Calls the book-era dual-pipeline kernel launched across CUDA streams."""

    def __init__(self) -> None:
        super().__init__()
        self.num_streams = 1
        self.tiles = 128
        self.ext = None  # Loaded lazily in setup()
        self.input_a: torch.Tensor | None = None
        self.input_b: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

        # Match constants from baseline_warp_specialized_two_pipelines_common.cuh
        self.tile_elems = 1024

    def setup(self) -> None:
        # Gracefully skip on GPUs without DSMEM/cluster support.
        ensure_dsmem_supported(description="warp-specialized cluster pipelines")
        # Load extension in setup to avoid timeout during discovery
        self.ext = _load_baseline_extension()
        
        torch.manual_seed(42)
        total_elems = self.tiles * self.tile_elems
        self.input_a = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.input_b = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input_a)
        self._synchronize()
        tokens = float(total_elems * 2)  # two inputs processed per iteration
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=1.0,
        )

    def benchmark_fn(self) -> None:
        assert self.input_a is not None and self.input_b is not None
        with self._nvtx_range("baseline_dual_pipeline_multistream"):
            result = self.ext.baseline_warp_specialized_multistream_forward(
                self.input_a,
                self.input_b,
                self.num_streams,
            )
        self._synchronize()
        if self.output is not None:
            self.output.copy_(result)

    def teardown(self) -> None:
        self.input_a = None
        self.input_b = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            measurement_timeout_seconds=120,
            setup_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_sequential_ms', 10.0),
            overlapped_time_ms=getattr(self, '_overlapped_ms', 5.0),
            num_streams=getattr(self, 'num_streams', 4),
            num_operations=getattr(self, 'num_operations', 4),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaselineDualPipelineBenchmark:
    return BaselineDualPipelineBenchmark()
