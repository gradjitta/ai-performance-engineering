"""Torch extension loader for TMEM-backed KV cache epilogues."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:
    def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
        pass

_BUILD_ERROR: Optional[Exception] = None
_EXT_NAME = "kv_cache_tmem_ext"


def _get_build_dir() -> Path:
    """Get the torch extension build directory."""
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    repo_root = Path(__file__).resolve().parents[2]
    if base:
        return Path(base) / _EXT_NAME
    return repo_root / ".torch_extensions" / _EXT_NAME


@functools.lru_cache(None)
def load_tmem_cache_module():
    """Compile and load the TMEM cache extension once per process."""
    global _BUILD_ERROR
    src = Path(__file__).with_name("tmem_cache_ext.cu")
    
    # Clean stale builds before attempting to load
    build_dir = _get_build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_build_directory(build_dir)
    
    try:
        repo_root = src.resolve().parents[2]
        include_dirs = [
            repo_root / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
            repo_root / "common" / "headers",
            repo_root / "third_party" / "cutlass" / "include",
        ]
        return load(
            name=_EXT_NAME,
            sources=[str(src)],
            extra_cuda_cflags=[
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ]
            + [f"-I{p}" for p in include_dirs],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build failures are surfaced to callers
        _BUILD_ERROR = exc
        raise


def build_error() -> Optional[Exception]:
    """Return the cached build failure, if any."""
    return _BUILD_ERROR


__all__ = ["load_tmem_cache_module", "build_error"]
