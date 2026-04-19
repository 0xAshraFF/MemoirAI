from __future__ import annotations

import math
import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"\w+")


@dataclass
class CompressionStats:
    original_chars: int
    compressed_chars: int
    original_chunks: int
    kept_chunks: int
    skipped_chunks: int
    estimated_filter_ratio: float
    trust_scores: list[float]


class MemoirPrefillFilter:
    """
    Conservative prompt-side approximation of MemoirAI filtering.

    This does not mutate the model's KV cache. It compresses repeated or near-
    duplicate prompt regions before generation so we can benchmark a real model
    path while the true KV-cache integration is still under construction.
    """

    def __init__(
        self,
        chunk_chars: int = 240,
        similarity_threshold: float = 0.92,
        min_chunk_tokens: int = 24,
    ) -> None:
        self.chunk_chars = chunk_chars
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens

    def compress(self, prompt: str, max_context_chars: int) -> tuple[str, CompressionStats]:
        trimmed = prompt[:max_context_chars]
        chunks = self._chunk_text(trimmed)
        kept: list[str] = []
        signatures: list[set[str]] = []
        trust_scores: list[float] = []

        for chunk in chunks:
            chunk_tokens = set(_TOKEN_RE.findall(chunk.lower()))
            if len(chunk_tokens) < self.min_chunk_tokens:
                kept.append(chunk)
                signatures.append(chunk_tokens)
                trust_scores.append(0.0)
                continue

            best_similarity = 0.0
            for prior in signatures:
                similarity = self._jaccard(chunk_tokens, prior)
                best_similarity = max(best_similarity, similarity)

            trust_scores.append(best_similarity)
            if best_similarity >= self.similarity_threshold:
                continue
            kept.append(chunk)
            signatures.append(chunk_tokens)

        compressed = "\n".join(kept)
        skipped = max(0, len(chunks) - len(kept))
        filter_ratio = skipped / len(chunks) if chunks else 0.0
        return compressed, CompressionStats(
            original_chars=len(trimmed),
            compressed_chars=len(compressed),
            original_chunks=len(chunks),
            kept_chunks=len(kept),
            skipped_chunks=skipped,
            estimated_filter_ratio=filter_ratio,
            trust_scores=trust_scores,
        )

    def _chunk_text(self, prompt: str) -> list[str]:
        lines = [line.rstrip() for line in prompt.splitlines()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > self.chunk_chars:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        if not chunks:
            chunks = [prompt]
        return chunks

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        return inter / union if union else 0.0


def estimate_kv_cache_bytes(char_count: int, hidden_factor: int = 128) -> int:
    """
    Proxy metric for prompt-side comparisons when a backend does not expose
    actual KV-cache bytes.
    """
    approx_tokens = max(1, math.ceil(char_count / 4))
    return approx_tokens * hidden_factor
