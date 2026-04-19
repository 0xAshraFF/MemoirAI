from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TASK_TYPES = {
    "hotpotqa": "qa",
    "2wikimqa": "qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "lcc": "code",
    "repobench-p": "code",
}


@dataclass
class LongBenchExample:
    task: str
    prompt: str
    answers: list[str]
    metadata: dict
    task_type: str


def _coerce_answers(record: dict) -> list[str]:
    answers = record.get("answers") or record.get("answer") or []
    if isinstance(answers, str):
        return [answers]
    return [str(item) for item in answers]


def _build_prompt(record: dict, task: str) -> str:
    context = record.get("context") or record.get("article") or record.get("passage") or ""
    question = record.get("question") or record.get("input") or record.get("instruction") or ""
    if task in {"gov_report", "qmsum"}:
        return (
            "You are solving a LongBench summarization task.\n\n"
            f"Document:\n{context}\n\n"
            "Write a concise faithful summary."
        ).strip()
    if task in {"lcc", "repobench-p"}:
        prefix = record.get("context") or record.get("input") or ""
        return (
            "You are solving a LongBench code completion task.\n\n"
            f"Code context:\n{prefix}\n\n"
            "Continue the code with the most likely next lines."
        ).strip()
    return (
        "You are solving a LongBench long-context QA task.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer as briefly as possible."
    ).strip()


def _convert_record(record: dict, task: str) -> LongBenchExample:
    return LongBenchExample(
        task=task,
        prompt=_build_prompt(record, task),
        answers=_coerce_answers(record),
        metadata={"id": record.get("_id") or record.get("id")},
        task_type=TASK_TYPES.get(task, "qa"),
    )


def _load_local_jsonl(path: str, task_names: list[str], limit: int) -> list[LongBenchExample]:
    rows: list[LongBenchExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task = record["task"]
            if task not in task_names:
                continue
            rows.append(_convert_record(record, task))
    return _limit_per_task(rows, limit)


def _limit_per_task(rows: Iterable[LongBenchExample], limit: int) -> list[LongBenchExample]:
    counts: dict[str, int] = {}
    kept: list[LongBenchExample] = []
    for row in rows:
        counts.setdefault(row.task, 0)
        if counts[row.task] >= limit:
            continue
        counts[row.task] += 1
        kept.append(row)
    return kept


def load_longbench_subset(
    dataset_source: str,
    dataset_name: str,
    task_names: list[str],
    limit: int,
    local_dataset_path: str | None = None,
) -> list[LongBenchExample]:
    if dataset_source == "local":
        if not local_dataset_path:
            raise ValueError("local_dataset_path is required when dataset_source='local'")
        return _load_local_jsonl(local_dataset_path, task_names, limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets is not installed. Install requirements or use dataset_source='local'."
        ) from exc

    rows: list[LongBenchExample] = []
    for task in task_names:
        dataset = load_dataset(
            dataset_name,
            task,
            split="test",
            trust_remote_code=True,
        )
        for record in dataset:
            rows.append(_convert_record(record, task))
    return _limit_per_task(rows, limit)
