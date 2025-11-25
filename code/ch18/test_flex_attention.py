"""Sanity checks for FlexAttention availability on Blackwell-class GPUs."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_flex_attention_available() -> None:
    """FlexAttention should import and expose a callable on Blackwell GPUs."""
    major, minor = torch.cuda.get_device_capability()
    # Blackwell is sm_100 (major=10), Grace-Blackwell is sm_12x
    if major < 10:
        pytest.skip(f"FlexAttention requires Blackwell-class GPUs (got sm_{major}{minor})")

    try:
        from torch.nn.attention.flex_attention import flex_attention  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - surface import issues
        pytest.fail(f"torch.nn.attention.flex_attention import failed: {exc}")

    assert callable(flex_attention)
