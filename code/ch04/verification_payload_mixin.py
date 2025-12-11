"""Shared verification helper for ch04 multi-GPU/disaggregated benchmarks.

This module re-exports the canonical mixin from core to keep chapter code
aligned with the central implementation.
"""

from core.benchmark.verification_mixin import VerificationPayload, VerificationPayloadMixin

__all__ = ["VerificationPayload", "VerificationPayloadMixin"]
