from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import CompressionMode, EvalConfig
from .longbench import load_longbench_subset
from .memoir_runtime import MemoirPrefillFilter, estimate_kv_cache_bytes
from .metrics import score_prediction
from .model_adapter import build_model_adapter
from .turboquant_adapter import TurboQuantBackend, TurboQuantConfig


def run_benchmark(config: EvalConfig) -> dict:
    memoir_filter = MemoirPrefillFilter(
        chunk_chars=config.memoir_chunk_chars,
        similarity_threshold=config.memoir_similarity_threshold,
        min_chunk_tokens=config.memoir_min_chunk_tokens,
    )
    turboquant = TurboQuantBackend(
        TurboQuantConfig(
            bits_key=config.turboquant_bits_key,
            bits_value=config.turboquant_bits_value,
        )
    )
    dataset = load_longbench_subset(
        dataset_source=config.dataset_source,
        dataset_name=config.dataset_name,
        task_names=config.task_names,
        limit=config.max_examples_per_task,
        local_dataset_path=config.local_dataset_path,
    )

    variants = [
        (CompressionMode.MEMOIR.value, False),
        (CompressionMode.MEMOIR_TURBOQUANT.value, True),
    ]
    runs: list[dict] = []
    for variant_name, turbo_enabled in variants:
        adapter = build_model_adapter(
            backend=config.backend,
            model_name=config.model_name,
            device=config.device,
            extra_model_kwargs=(
                turboquant.generation_kwargs()
                if turbo_enabled and turboquant.is_available
                else None
            ),
        )
        variant_rows: list[dict] = []
        for example in dataset:
            compressed_prompt, compression = memoir_filter.compress(
                example.prompt,
                max_context_chars=config.max_context_chars,
            )
            generation = adapter.generate(
                prompt=compressed_prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            scores = score_prediction(example.task_type, generation.text, example.answers)
            kv_proxy = estimate_kv_cache_bytes(compression.compressed_chars)
            if turbo_enabled:
                kv_proxy = int(kv_proxy * turboquant.memory_multiplier())
            variant_rows.append(
                {
                    "task": example.task,
                    "task_type": example.task_type,
                    "prediction": generation.text,
                    "answers": example.answers,
                    "scores": scores,
                    "latency_sec": generation.latency_sec,
                    "compression": asdict(compression),
                    "kv_cache_proxy_bytes": kv_proxy,
                    "turboquant_enabled": turbo_enabled,
                    "turboquant_backend_available": turboquant.is_available,
                    "metadata": example.metadata,
                }
            )
        runs.append({"variant": variant_name, "summary": _summarize_rows(variant_rows), "rows": variant_rows})

    report = {"config": asdict(config), "runs": runs}
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    primary_scores = [row["scores"]["primary"] for row in rows]
    latencies = [row["latency_sec"] for row in rows]
    filter_ratios = [row["compression"]["estimated_filter_ratio"] for row in rows]
    kv_proxy = [row["kv_cache_proxy_bytes"] for row in rows]
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row["scores"]["primary"])
    return {
        "examples": len(rows),
        "mean_primary_score": sum(primary_scores) / len(primary_scores),
        "mean_latency_sec": sum(latencies) / len(latencies),
        "mean_filter_ratio": sum(filter_ratios) / len(filter_ratios),
        "mean_kv_cache_proxy_bytes": sum(kv_proxy) / len(kv_proxy),
        "score_by_task": {
            task: sum(values) / len(values)
            for task, values in sorted(by_task.items())
        },
    }


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Run LongBench-style MemoirAI evaluation.")
    parser.add_argument("--backend", default="dummy", choices=["dummy", "transformers"])
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset-source", default="local", choices=["local", "hf"])
    parser.add_argument("--dataset-name", default="THUDM/LongBench")
    parser.add_argument("--local-dataset-path", default="real_eval/data/sample_longbench_subset.jsonl")
    parser.add_argument("--output-path", default="real_eval/reports/latest_report.json")
    parser.add_argument("--max-examples-per-task", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-names", nargs="+", default=["hotpotqa", "gov_report", "lcc"])
    parser.add_argument("--turboquant-bits-key", type=int, default=3)
    parser.add_argument("--turboquant-bits-value", type=int, default=4)
    args = parser.parse_args()
    return EvalConfig(
        backend=args.backend,
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        local_dataset_path=args.local_dataset_path,
        output_path=args.output_path,
        max_examples_per_task=args.max_examples_per_task,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        task_names=args.task_names,
        turboquant_bits_key=args.turboquant_bits_key,
        turboquant_bits_value=args.turboquant_bits_value,
    )


if __name__ == "__main__":
    report = run_benchmark(parse_args())
    for run in report["runs"]:
        summary = run["summary"]
        print(
            f"{run['variant']}: "
            f"score={summary.get('mean_primary_score', 0.0):.3f} "
            f"latency={summary.get('mean_latency_sec', 0.0):.4f}s "
            f"filter={summary.get('mean_filter_ratio', 0.0):.3f}"
        )
