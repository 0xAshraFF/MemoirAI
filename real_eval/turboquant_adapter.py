from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    bits_key: int = 3
    bits_value: int = 4


class TurboQuantBackend:
    """
    Optional runtime hook for an external TurboQuant backend.

    If the package is unavailable, the benchmark still runs and reports a
    memory proxy plus capability flags, but quality comparisons remain driven by
    the same base model output path.
    """

    def __init__(self, config: TurboQuantConfig) -> None:
        self.config = config
        self._backend = self._detect_backend()

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    def generation_kwargs(self) -> dict:
        if not self._backend:
            return {}
        return {
            "turboquant_k_bits": self.config.bits_key,
            "turboquant_v_bits": self.config.bits_value,
        }

    def memory_multiplier(self) -> float:
        if self.config.bits_value <= 2:
            return 0.23
        return 0.31

    @staticmethod
    def _detect_backend():
        try:
            import turboquant  # type: ignore # noqa: F401
        except ModuleNotFoundError:
            return None
        return "turboquant"
