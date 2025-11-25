"""baseline_stream_ordered_kv_cache.py - Single-stream KV cache updates."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineStreamOrderedKvCacheBenchmark(StridedStreamBaseline):
    """Baseline: sequential KV cache updates with no overlap."""

    def __init__(self):
        super().__init__(
            "stream_ordered_kv_cache",
            num_elements=18_000_000,
            num_segments=8,
        )


def get_benchmark() -> BaselineStreamOrderedKvCacheBenchmark:
    return BaselineStreamOrderedKvCacheBenchmark()
