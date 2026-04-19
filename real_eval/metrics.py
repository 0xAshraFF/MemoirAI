from __future__ import annotations

import re
from collections import Counter


_TOKEN_RE = re.compile(r"\w+")


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def exact_match(prediction: str, answers: list[str]) -> float:
    norm_pred = _normalize(prediction)
    return float(any(norm_pred == _normalize(answer) for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    pred = _tokens(prediction)
    best = 0.0
    for answer in answers:
        gold = _tokens(answer)
        if not pred or not gold:
            score = float(pred == gold)
        else:
            overlap = Counter(pred) & Counter(gold)
            common = sum(overlap.values())
            if common == 0:
                score = 0.0
            else:
                precision = common / len(pred)
                recall = common / len(gold)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return float(best)


def rouge_l_recall(prediction: str, answers: list[str]) -> float:
    pred_tokens = _tokens(prediction)
    best = 0.0
    for answer in answers:
        gold_tokens = _tokens(answer)
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        lcs = _lcs_length(pred_tokens, gold_tokens)
        best = max(best, lcs / len(gold_tokens))
    return float(best)


def code_overlap(prediction: str, answers: list[str]) -> float:
    pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    best = 0.0
    for answer in answers:
        gold_lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if not pred_lines or not gold_lines:
            best = max(best, float(pred_lines == gold_lines))
            continue
        prefix = 0
        for pred_line, gold_line in zip(pred_lines, gold_lines):
            if pred_line != gold_line:
                break
            prefix += 1
        best = max(best, prefix / len(gold_lines))
    return float(best)


def score_prediction(task_type: str, prediction: str, answers: list[str]) -> dict:
    if task_type == "summarization":
        return {"primary": rouge_l_recall(prediction, answers), "metric": "rouge_l_recall"}
    if task_type == "code":
        return {"primary": code_overlap(prediction, answers), "metric": "code_overlap"}
    return {
        "primary": token_f1(prediction, answers),
        "metric": "token_f1",
        "exact_match": exact_match(prediction, answers),
    }


def _lcs_length(xs: list[str], ys: list[str]) -> int:
    dp = [0] * (len(ys) + 1)
    for x in xs:
        prev = 0
        for j, y in enumerate(ys, start=1):
            cur = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]
