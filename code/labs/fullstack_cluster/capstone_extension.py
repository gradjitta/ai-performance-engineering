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

_EXT_NAME = "blackwell_capstone_ext"


def _get_build_dir() -> Path:
    """Get the torch extension build directory."""
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    repo_root = Path(__file__).resolve().parents[2]
    if base:
        return Path(base) / _EXT_NAME
    return repo_root / ".torch_extensions" / _EXT_NAME


@functools.lru_cache(None)
def load_capstone_module():
    """Compile (if needed) and return the CUDA extension."""
    root = Path(__file__).resolve().parent
    src = root / "capstone_kernels.cu"

    # Clean stale builds before attempting to load
    build_dir = _get_build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_build_directory(build_dir)

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_103,code=sm_103",
        "-gencode=arch=compute_121,code=sm_121",
        "-lineinfo",
    ]
    extra_cflags = ["-std=c++20"]

    module = load(
        name=_EXT_NAME,
        sources=[str(src)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module
