"""Solution 10: model clean data contracts for ingestion."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    source_system: str
    event_ts: datetime
    payload_size_bytes: int


REQUIRED_KEYS = {"event_id", "source_system", "event_ts", "payload_size_bytes"}


def parse_event(raw: dict[str, Any]) -> EventRecord:
    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    event_ts = datetime.fromisoformat(str(raw["event_ts"]))
    return EventRecord(
        event_id=str(raw["event_id"]),
        source_system=str(raw["source_system"]),
        event_ts=event_ts,
        payload_size_bytes=int(raw["payload_size_bytes"]),
    )


if __name__ == "__main__":
    sample = {
        "event_id": "evt-101",
        "source_system": "billing-api",
        "event_ts": "2026-04-14T00:00:00",
        "payload_size_bytes": 512,
    }
    print(parse_event(sample))
