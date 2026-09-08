"""Relational validation for canonical semantic tables."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real

import pandas as pd

from dual_uq.structcal.condition_semantics import orientation_for
from dual_uq.structcal.schema_registry import SchemaRegistry


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _matches_schema(value: object, schema: dict[str, object]) -> bool:
    if "anyOf" in schema:
        return any(_matches_schema(value, branch) for branch in schema["anyOf"])
    expected = schema.get("type")
    if expected == "null":
        return _is_null(value)
    if _is_null(value):
        return False
    if expected == "string" and not isinstance(value, str):
        return False
    if expected == "boolean" and not isinstance(value, (bool,)):
        return False
    if expected == "integer" and (isinstance(value, bool) or not isinstance(value, Integral)):
        return False
    if expected == "number" and (isinstance(value, bool) or not isinstance(value, Real)):
        return False
    if expected == "array" and not isinstance(value, (list, tuple)):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        return False
    return not (isinstance(value, Real) and "minimum" in schema and value < schema["minimum"])


def validate_primary_key(frame: pd.DataFrame, key: Iterable[str]) -> None:
    key = tuple(key)
    missing = sorted(set(key) - set(frame.columns))
    if missing:
        raise ValueError(f"primary key fields missing: {missing}")
    if frame[list(key)].isna().any().any():
        raise ValueError("primary key contains null")
    if frame.duplicated(list(key)).any():
        raise ValueError(f"duplicate primary key: {key}")


def validate_frame_contract(frame: pd.DataFrame, table: str, registry: SchemaRegistry) -> None:
    schema = registry.get(table)
    missing = sorted(set(schema.required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} missing required fields: {missing}")
    extra = sorted(set(frame.columns) - set(schema.properties))
    if extra:
        raise ValueError(f"{table} has fields outside frozen schema: {extra}")
    if schema.primary_key:
        validate_primary_key(frame, schema.primary_key)
    prohibited = set(schema.prohibited_fields).intersection(frame.columns)
    if prohibited:
        raise ValueError(f"{table} contains prohibited fields: {sorted(prohibited)}")
    for column, column_schema in registry.get(table).payload.get("properties", {}).items():
        if column not in frame:
            continue
        invalid = [index for index, value in frame[column].items() if not _matches_schema(value, column_schema)]
        if invalid:
            raise ValueError(f"{table}.{column} violates frozen schema at rows {invalid[:5]}")


def validate_pair_orientation(row: dict[str, object]) -> None:
    labels = orientation_for(str(row["arm"]), row.get("state_family") and str(row["state_family"]))
    actual = (str(row["condition_1_label"]), str(row["condition_2_label"]))
    if actual != labels:
        raise ValueError(f"pair orientation {actual} != canonical {labels}")


def validate_core_relationships(tables: dict[str, pd.DataFrame], registry: SchemaRegistry) -> None:
    for name, frame in tables.items():
        validate_frame_contract(frame, name, registry)
    proteins = set(tables["proteins"]["protein_id"])
    structures = tables["structures"]
    if not set(structures["protein_id"]).issubset(proteins):
        raise ValueError("structures reference unknown proteins")
    pairs = tables["condition_pairs"]
    if not set(pairs["protein_id"]).issubset(proteins):
        raise ValueError("pairs reference unknown proteins")
    structure_ids = set(structures["structure_id"])
    if not set(pairs["condition_1_structure_id"]).issubset(structure_ids) or not set(pairs["condition_2_structure_id"]).issubset(structure_ids):
        raise ValueError("pairs reference unknown structures")
    for row in pairs.to_dict(orient="records"):
        validate_pair_orientation(row)
