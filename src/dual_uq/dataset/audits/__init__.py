"""Read-only engineering audits for the unified dataset domain."""

from .namespace import (
    NamespaceInventory,
    capture_namespace_inventory,
    write_namespace_inventory,
)

__all__ = [
    "NamespaceInventory",
    "capture_namespace_inventory",
    "write_namespace_inventory",
]
