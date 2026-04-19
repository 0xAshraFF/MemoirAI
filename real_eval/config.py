from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CompressionMode(str, Enum):
    MEMOIR = "memoir"
    MEMOIR_TURBOQUANT = "memoir_turboquant"


@dataclass
class EvalConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "THUDM/LongBench"
    dataset_source: str = "hf"
    output_path: str = "real_eval/reports/latest_report.json"
    max_examples_per_task: int = 5
    max_context_chars: int = 12000
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0
    memoir_chunk_chars: int = 240
    memoir_similarity_threshold: float = 0.92
    memoir_min_chunk_tokens: int = 24
    memoir_filter_mode: str = "conservative"
    turboquant_bits_key: int = 3
    turboquant_bits_value: int = 4
    turboquant_enabled: bool = False
    device: str = "cpu"
    backend: str = "dummy"
    local_dataset_path: Optional[str] = None
    task_names: list[str] = field(
        default_factory=lambda: ["hotpotqa", "gov_report", "lcc"]
    )
