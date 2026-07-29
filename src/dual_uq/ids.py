from __future__ import annotations
import hashlib

def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    payload = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"
