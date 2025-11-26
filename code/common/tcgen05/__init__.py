"""Shared tcgen05 kernel loaders and Python wrappers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

from common.python.tcgen05_requirements import ensure_tcgen05_supported

try:  # Ensure TORCH_CUDA_ARCH_LIST stays clamped for GB-series hosts.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional bootstrap
    arch_config = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUTLASS_INCLUDES: list[Path] = []
# Prefer the TransformerEngine-bundled CUTLASS (includes SM100 tcgen05/TMEM), then fall back to local checkouts.
for _cand in (
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "cutlass_latest" / "cutlass-main" / "include",
):
    if _cand.exists():
        _CUTLASS_INCLUDES = [_cand]
        break
_CLANG_HOST = _REPO_ROOT / "third_party" / "llvm" / "bin" / "clang++"


def _tcgen05_cuda_flags() -> list[str]:
    flags = [
        "-std=c++20",
    ]
    for inc in _CUTLASS_INCLUDES:
        flags.append(f"-I{inc}")
    caps: list[tuple[int, int]] = [(10, 0)]
    major, minor = torch.cuda.get_device_capability()
    if major >= 12:
        caps.insert(0, (12, 0))
    elif major >= 10 and (major, minor) not in caps:
        caps.insert(0, (major, minor))
    seen = set()
    for maj, minr in caps:
        if (maj, minr) in seen:
            continue
        seen.add((maj, minr))
        flags.append(f"-gencode=arch=compute_{maj}{minr},code=sm_{maj}{minr}")
    if _CLANG_HOST.exists():
        flags.append(f"-ccbin={_CLANG_HOST}")
    return flags


def _get_extension_build_dir(name: str) -> Path:
    """Get the torch extension build directory for a given extension name."""
    # torch extensions default to ~/.cache/torch_extensions or TORCH_EXTENSIONS_DIR
    import os
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    if base:
        return Path(base) / name
    # Fall back to workspace .torch_extensions
    return _REPO_ROOT / ".torch_extensions" / name


def _clean_stale_build(name: str) -> None:
    """Remove stale build artifacts if .so is missing but build.ninja exists."""
    import shutil
    build_dir = _get_extension_build_dir(name)
    ninja_file = build_dir / "build.ninja"
    so_file = build_dir / f"{name}.so"
    
    if ninja_file.exists() and not so_file.exists():
        # Stale build directory - ninja exists but .so missing means build failed
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass  # Best effort cleanup


def _load_extension(name: str, sources: Sequence[Path]):
    # Clean up stale build artifacts to force rebuild if needed
    _clean_stale_build(name)
    
    try:
        return load(
            name=name,
            sources=[str(src) for src in sources],
            extra_cuda_cflags=_tcgen05_cuda_flags(),
            extra_cflags=["-std=c++20"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    except Exception as e:
        # On failure, retry with verbose=True to capture build errors
        error_msg = str(e)
        if "cannot open shared object file" in error_msg or "No such file" in error_msg:
            # Clean up and retry with verbose output
            _clean_stale_build(name)
            try:
                return load(
                    name=name,
                    sources=[str(src) for src in sources],
                    extra_cuda_cflags=_tcgen05_cuda_flags(),
                    extra_cflags=["-std=c++20"],
                    extra_ldflags=["-lcuda"],
                    verbose=True,  # Show build errors on retry
                )
            except Exception as retry_e:
                raise RuntimeError(
                    f"Failed to build tcgen05 extension '{name}'. "
                    f"Build errors (see above). Original error: {retry_e}"
                ) from retry_e
        raise


@lru_cache(None)
def load_matmul_tcgen05_module():
    """Compile (if needed) and return the Chapter 10 tcgen05 matmul extension."""
    import os
    from common.python.smoke import is_smoke_mode
    if is_smoke_mode():
        raise RuntimeError("SKIPPED: tcgen05 extension disabled in low-memory mode")
    return _load_extension("ch10_matmul_tcgen05_ext", [_REPO_ROOT / "ch10" / "matmul_tcgen05.cu"])


@lru_cache(None)
def load_tiling_tcgen05_module():
    """Compile (if needed) and return the Chapter 8 tcgen05 tiling extension."""
    import os
    from common.python.smoke import is_smoke_mode
    if is_smoke_mode():
        raise RuntimeError("SKIPPED: tcgen05 extension disabled in low-memory mode")
    return _load_extension("ch8_tiling_tcgen05_ext", [_REPO_ROOT / "ch8" / "tiling_kernels_tcgen05.cu"])


def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor, *, module_name: str = "tcgen05 matmul") -> torch.Tensor:
    """Execute the CUTLASS tcgen05 GEMM after ensuring hardware/toolchain support."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tcgen05_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    *,
    module_name: str = "tcgen05 matmul bias+SiLU",
) -> torch.Tensor:
    """Execute the tcgen05 GEMM with TMEM-resident bias+SiLU epilogue."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05_bias_silu(a, b, bias)


def matmul_tiling_tcgen05(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    module_name: str = "tcgen05 tiling matmul",
) -> torch.Tensor:
    """Execute the CUTLASS tcgen05 tiling GEMM."""
    ensure_tcgen05_supported(loader=load_tiling_tcgen05_module, module_name=module_name)
    module = load_tiling_tcgen05_module()
    return module.matmul_tiling_tcgen05(a, b)


__all__ = [
    "load_matmul_tcgen05_module",
    "load_tiling_tcgen05_module",
    "matmul_tcgen05",
    "matmul_tcgen05_bias_silu",
    "matmul_tiling_tcgen05",
]
