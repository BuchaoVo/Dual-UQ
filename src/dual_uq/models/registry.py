"""Small in-process model adapter registry."""

from dual_uq.models.interface import InverseFoldingAdapter


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, InverseFoldingAdapter] = {}

    def register(self, adapter: InverseFoldingAdapter) -> None:
        if not getattr(adapter, "model_id", None):
            raise ValueError("adapter model_id is required")
        if adapter.model_id in self._adapters:
            raise ValueError(f"model already registered: {adapter.model_id}")
        self._adapters[adapter.model_id] = adapter

    def get(self, model_id: str) -> InverseFoldingAdapter:
        try:
            return self._adapters[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model adapter: {model_id}") from exc

    def capabilities(self, model_id: str):
        return self.get(model_id).capabilities()
