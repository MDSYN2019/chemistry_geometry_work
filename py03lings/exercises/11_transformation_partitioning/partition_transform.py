"""Exercise 11: transform records and partition for efficient reads."""

from collections import defaultdict
from datetime import datetime
from typing import Any


def normalize_amount(value: Any) -> float:
    """Coerce amount-like values to float with 2 decimals."""
    # TODO: cast value to float and round to 2 decimals.
    return 0.0


def partition_key(event_ts: str) -> str:
    """Return partition key in format dt=YYYY-MM-DD."""
    # TODO: parse event_ts with datetime.fromisoformat and build dt key.
    return "dt=1970-01-01"


def transform_and_partition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize amounts and group rows by date partition."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        # TODO: create a copied row with normalized amount.
        # TODO: append the transformed row to grouped[partition_key(row["event_ts"])].
        pass
    return dict(grouped)


if __name__ == "__main__":
    sample_rows = [
        {"order_id": "A1", "event_ts": "2026-04-13T09:10:00", "amount": "11.234"},
        {"order_id": "A2", "event_ts": "2026-04-13T15:10:00", "amount": 5},
        {"order_id": "A3", "event_ts": "2026-04-14T01:00:00", "amount": "7.8"},
    ]
    print(transform_and_partition(sample_rows))
