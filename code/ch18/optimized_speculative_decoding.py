"""optimized_speculative_decoding.py - Optimized speculative decoding with proper verification.

Demonstrates speculative decoding for parallel token generation with proper
draft-verify loop implementing acceptance/rejection based on target model.

Key Optimization (Ch18):
- Draft model generates K speculative tokens in parallel
- Target model verifies all K+1 positions in a single forward pass
- Accept matching tokens, reject on first mismatch
- Achieves 2-3x speedup when draft model has high acceptance rate

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

class OptimizedSpeculativeDecodingBenchmark(BaseBenchmark):
    """Optimized: Speculative decoding with proper draft-verify loop.
    
    Speculative Decoding Algorithm:
    1. Draft model generates K speculative tokens quickly
    2. Target model verifies all K+1 positions in ONE forward pass
    3. Accept tokens that match target's prediction
    4. On first mismatch: reject remaining draft tokens, use target's token
    
    Why this is faster:
    - Draft model is small/fast (2 layers vs 4 layers)
    - Target model processes K+1 tokens in parallel (not sequentially)
    - If acceptance rate is high, we get K tokens for ~1 target forward pass
    """
    
    def __init__(self):
        super().__init__()
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        self.speculative_length = 4  # K: Number of tokens to predict speculatively
        self.tokens_to_generate = 40  # Match baseline: 10 sequences * 4 tokens
        
        # Metrics tracking
        self.total_accepted = 0
        self.total_generated = 0
        self.verification_rounds = 0
        self.target_forward_passes = 0  # Track actual target model calls
        
        batch_size = 4
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(batch_size * self.tokens_to_generate),
        )
    
    def setup(self) -> None:
        """Setup: Initialize target and draft models with shared LM head."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        # Reset metrics
        self.total_accepted = 0
        self.total_generated = 0
        self.verification_rounds = 0
        self.target_forward_passes = 0
        
        # Match baseline configuration for fair comparison
        self.hidden_dim = 4096
        self.vocab_size = 32000
        
        # Simple model class matching baseline for fair comparison
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size, is_draft):
                super().__init__()
                # Draft model is smaller (1/4 hidden size)
                h = hidden_size // 4 if is_draft else hidden_size
                self.embedding = nn.Embedding(vocab_size, h)
                self.linear1 = nn.Linear(h, h)
                self.linear2 = nn.Linear(h, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = torch.relu(self.linear1(x))
                logits = self.linear2(x)
                return logits
        
        # Target model (large, accurate) - full hidden size
        self.target_model = SimpleLM(self.vocab_size, self.hidden_dim, is_draft=False).to(self.device).eval()
        
        # Draft model (small, fast) - 1/4 hidden size  
        self.draft_model = SimpleLM(self.vocab_size, self.hidden_dim, is_draft=True).to(self.device).eval()
        
        # Input - single token per sequence (like baseline)
        batch_size = 4
        self.batch_size = batch_size
        self.input_ids = torch.randint(0, self.vocab_size, (batch_size, 1), device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def _draft_generate(self, current_ids: torch.Tensor) -> torch.Tensor:
        """Generate K speculative tokens using draft model.
        
        Returns:
            draft_tokens: (batch, K) predicted token IDs
        """
        draft_tokens = []
        ids = current_ids
        
        for _ in range(self.speculative_length):
            # Draft model forward - outputs logits directly
            with torch.no_grad():
                logits = self.draft_model(ids[:, -1:])
                next_token = torch.argmax(logits, dim=-1)
                draft_tokens.append(next_token)
                ids = torch.cat([ids, next_token], dim=1)
        
        return torch.cat(draft_tokens, dim=1)  # [batch, K]
    
    def _verify_tokens(
        self, 
        current_ids: torch.Tensor, 
        draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens against target model.
        
        KEY OPTIMIZATION: Target model verifies ALL positions in ONE forward pass!
        This is the core of speculative decoding's speedup.
        
        Returns:
            final_tokens: Accepted tokens + correction token
            accepted: Number of accepted draft tokens
        """
        # Concatenate input + draft tokens for batched verification
        combined = torch.cat([current_ids, draft_tokens], dim=1)
        
        # Target model forward - ONE pass to verify all K positions
        with torch.no_grad():
            logits = self.target_model(combined)
        
        # Check each draft token against target's prediction
        accepted = 0
        for i in range(self.speculative_length):
            # Target's prediction for position i (relative to end)
            target_token = torch.argmax(logits[:, -(self.speculative_length - i + 1), :], dim=-1)
            if torch.all(target_token == draft_tokens[:, i]):
                accepted += 1
            else:
                break
        
        # Return accepted tokens + next token from target
        if accepted < self.speculative_length:
            final_tokens = draft_tokens[:, :accepted]
            next_token = torch.argmax(logits[:, -(self.speculative_length - accepted), :], dim=-1)
            final_tokens = torch.cat([final_tokens, next_token.unsqueeze(1)], dim=1)
        else:
            final_tokens = draft_tokens
        
        return final_tokens, accepted
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding with batched verification.
        
        KEY OPTIMIZATION: Target model verifies K draft tokens in ONE forward pass,
        instead of K sequential forward passes. This is the core speedup.
        
        Comparison with baseline:
        - Baseline: 40 target forward passes (one per token)
        - Speculative: ~10 target forward passes (one per K=4 tokens on average)
        
        Algorithm:
        1. Draft model generates K tokens (K sequential small-model passes)
        2. Target model verifies ALL K positions in ONE forward pass
        3. Accept matching tokens, reject on first mismatch
        4. Repeat until we've generated tokens_to_generate tokens
        """
        with self._nvtx_range("optimized_speculative_decoding"):
            current_ids = self.input_ids.clone()
            generated = 0
            
            # Generate tokens_to_generate tokens using speculative decoding
            while generated < self.tokens_to_generate:
                # Step 1: Draft phase - generate K speculative tokens (small model)
                with self._nvtx_range("draft_generate"):
                    draft_tokens = self._draft_generate(current_ids)
                
                # Step 2: Verify phase - ONE target forward pass for all K positions
                with self._nvtx_range("verify_tokens"):
                    accepted_tokens, num_accepted = self._verify_tokens(current_ids, draft_tokens)
                
                self.verification_rounds += 1
                self.target_forward_passes += 1  # Only ONE target call per round!
                self.total_accepted += num_accepted
                
                # Update sequence
                current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
                generated += accepted_tokens.shape[1]
            
            self.total_generated = generated
                        
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics explaining WHY it's faster.
        
        Key metrics:
        - target_forward_passes: How many times we called the expensive target model
        - acceptance_rate_pct: Higher = more speedup (draft model is accurate)
        - forward_pass_reduction: Baseline needs 40 calls, we need fewer
        """
        # Calculate acceptance rate
        max_possible = self.verification_rounds * self.speculative_length
        acceptance_rate = (self.total_accepted / max_possible * 100) if max_possible > 0 else 0.0
        
        # Average tokens generated per verification round  
        avg_tokens_per_round = (
            self.total_generated / self.verification_rounds 
            if self.verification_rounds > 0 else 0.0
        )
        
        # Forward pass reduction: baseline uses 40, we use fewer
        baseline_forward_passes = self.tokens_to_generate  # 40 in baseline
        forward_pass_reduction = (
            (baseline_forward_passes - self.target_forward_passes) / baseline_forward_passes * 100
            if baseline_forward_passes > 0 else 0.0
        )
        
        return {
            "decode.tokens_generated": float(self.total_generated),
            "decode.target_forward_passes": float(self.target_forward_passes),
            "decode.method": 1.0,  # 0 = sequential, 1 = speculative
            "spec_decode.speculative_length_k": float(self.speculative_length),
            "spec_decode.verification_rounds": float(self.verification_rounds),
            "spec_decode.acceptance_rate_pct": acceptance_rate,
            "spec_decode.avg_tokens_per_round": avg_tokens_per_round,
            "spec_decode.forward_pass_reduction_pct": forward_pass_reduction,
        }

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_model is None or self.draft_model is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedSpeculativeDecodingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Speculative Decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
