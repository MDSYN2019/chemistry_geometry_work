"""Solution 11: transform records and partition for efficient reads."""

from collections import defaultdict
from datetime import datetime
from typing import Any


def normalize_amount(value: Any) -> float:
    return round(float(value), 2)


def partition_key(event_ts: str) -> str:
    day = datetime.fromisoformat(event_ts).date().isoformat()
    return f"dt={day}"


def transform_and_partition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        transformed = dict(row)
        transformed["amount"] = normalize_amount(row["amount"])
        grouped[partition_key(str(row["event_ts"]))].append(transformed)
    return dict(grouped)


if __name__ == "__main__":
    sample_rows = [
        {"order_id": "A1", "event_ts": "2026-04-13T09:10:00", "amount": "11.234"},
        {"order_id": "A2", "event_ts": "2026-04-13T15:10:00", "amount": 5},
        {"order_id": "A3", "event_ts": "2026-04-14T01:00:00", "amount": "7.8"},
    ]
    print(transform_and_partition(sample_rows))
