"""Shared helpers to fetch required Hugging Face models on demand."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - huggingface_hub should be installed
    snapshot_download = None
    HF_IMPORT_ERROR = exc
else:
    HF_IMPORT_ERROR = None


def ensure_gpt_oss_20b(target_dir: Optional[Path] = None) -> Path:
    """Ensure the openai/gpt-oss-20b model is present locally.

    Downloads the full repository (including safetensors) into
    ``gpt-oss-20b/original`` at repo root by default.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parents[2] / "gpt-oss-20b" / "original"
    target_dir = Path(target_dir)
    if (target_dir / "config.json").exists():
        return target_dir

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required to fetch gpt-oss-20b "
            f"(import error: {HF_IMPORT_ERROR})"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="openai/gpt-oss-20b",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        # Fetch full repo including safetensors to avoid missing-weight errors.
        allow_patterns=None,
        resume_download=True,
    )
    return target_dir
