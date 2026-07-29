from __future__ import annotations

import json
import time

from dual_uq.afdb import get_afdb_prediction_metadata
from dual_uq.net import request_json
from dual_uq.rcsb_discovery import (
    build_polymer_entity_query,
    search_polymer_entities,
)


def timed(label: str, function):
    print(f"[START] {label}", flush=True)
    started = time.perf_counter()
    try:
        result = function()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"[FAIL]  {label} after {elapsed:.2f}s: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - started
    print(f"[OK]    {label} in {elapsed:.2f}s", flush=True)
    return result


def main() -> None:
    query = build_polymer_entity_query(
        rows=5,
        methods=["X-RAY DIFFRACTION"],
        resolution_max=3.0,
        length_min=100,
        length_max=500,
        sequence_identity_grouping=None,
    )

    identifiers = timed(
        "RCSB Search API (5 ungrouped polymer entities)",
        lambda: search_polymer_entities(query),
    )
    print(json.dumps({"identifiers": identifiers}, indent=2), flush=True)

    timed(
        "RCSB Data API entry/1AKE",
        lambda: request_json(
            "https://data.rcsb.org/rest/v1/core/entry/1AKE"
        ),
    )
    timed(
        "RCSB Data API polymer_entity/1AKE/1",
        lambda: request_json(
            "https://data.rcsb.org/rest/v1/core/polymer_entity/1AKE/1"
        ),
    )

    metadata = timed(
        "AlphaFold DB prediction/P69441",
        lambda: get_afdb_prediction_metadata("P69441"),
    )
    print(
        json.dumps(
            {
                "modelEntityId": metadata.get("modelEntityId"),
                "latestVersion": metadata.get("latestVersion"),
                "globalMetricValue": metadata.get("globalMetricValue"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
