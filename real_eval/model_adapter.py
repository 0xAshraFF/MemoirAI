from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    raw: dict


class BaseModelAdapter:
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        raise NotImplementedError


class DummyEchoAdapter(BaseModelAdapter):
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        start = time.perf_counter()
        tail = prompt.splitlines()[-1] if prompt.splitlines() else prompt
        text = f"[dummy] {tail[:max_new_tokens]}"
        return GenerationResult(text=text, latency_sec=time.perf_counter() - start, raw={})


class TransformersCausalLMAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        extra_model_kwargs: dict | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers and torch are required for backend='transformers'"
            ) from exc

        model_kwargs = extra_model_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.device = device
        self.torch = torch
        self.model.to(device)
        self.model.eval()
        self.max_input_tokens = self._resolve_max_input_tokens()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        start = time.perf_counter()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])
        available_new_tokens = max(1, self.max_input_tokens - prompt_tokens)
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=min(max_new_tokens, available_new_tokens),
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return GenerationResult(text=text, latency_sec=time.perf_counter() - start, raw={})

    def _resolve_max_input_tokens(self) -> int:
        model_limit = getattr(self.model.config, "n_positions", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        candidates = [
            value for value in (model_limit, tokenizer_limit)
            if isinstance(value, int) and value > 0 and value < 10**6
        ]
        return min(candidates) if candidates else 2048


def build_model_adapter(
    backend: str,
    model_name: str,
    device: str,
    extra_model_kwargs: dict | None = None,
) -> BaseModelAdapter:
    if backend == "dummy":
        return DummyEchoAdapter()
    if backend == "transformers":
        return TransformersCausalLMAdapter(
            model_name=model_name,
            device=device,
            extra_model_kwargs=extra_model_kwargs,
        )
    raise ValueError(f"Unsupported backend: {backend}")
