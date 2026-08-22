"""Single loader for the frozen JSON schema authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SchemaDefinition:
    name: str
    filename: str
    payload: dict[str, Any]

    @property
    def properties(self) -> tuple[str, ...]:
        return tuple(self.payload.get("properties", {}))

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.payload.get("required", ()))

    @property
    def primary_key(self) -> tuple[str, ...]:
        return tuple(self.payload.get("x-primary-key", ()))

    @property
    def prohibited_fields(self) -> tuple[str, ...]:
        return tuple(self.payload.get("x-prohibited-fields", ()))

    @property
    def version(self) -> str | None:
        return self.payload.get("x-version")


class SchemaRegistry:
    """Resolve semantic table names against ``schemas/*.schema.json``."""

    def __init__(self, schema_root: str | Path):
        self.root = Path(schema_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def get(self, name: str) -> SchemaDefinition:
        filename = name if name.endswith(".schema.json") else f"{name}.schema.json"
        path = self.root / filename
        if not path.is_file():
            raise KeyError(f"unknown frozen schema: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        semantic_name = filename.removesuffix(".schema.json")
        return SchemaDefinition(semantic_name, filename, payload)
