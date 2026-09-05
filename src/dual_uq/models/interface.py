"""Minimal model adapter boundary; concrete models remain in existing modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dual_uq.models.capabilities import ModelCapability


class InverseFoldingAdapter(ABC):
    model_id: str
    model_version: str

    @abstractmethod
    def capabilities(self) -> frozenset[ModelCapability]:
        raise NotImplementedError

    def check_eligibility(self, instance: Any, capability: ModelCapability) -> tuple[bool, str | None]:
        return capability in self.capabilities(), None if capability in self.capabilities() else "capability_not_declared"

    def local_scores(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def score_sequences(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def generate_multistate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
