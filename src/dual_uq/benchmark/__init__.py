"""Model-independent canonical benchmark contracts."""

from dual_uq.benchmark.condition_semantics import orient_pair
from dual_uq.benchmark.ids import canonical_id, pair_id, protein_id
from dual_uq.benchmark.instances import BenchmarkInstance
from dual_uq.benchmark.schema_registry import SchemaRegistry

__all__ = ["BenchmarkInstance", "SchemaRegistry", "canonical_id", "orient_pair", "pair_id", "protein_id"]
