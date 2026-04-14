"""Exercise 10: model clean data contracts for ingestion."""

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
    """Build a validated EventRecord from a raw dictionary."""
    # TODO: verify all REQUIRED_KEYS are present; raise ValueError if missing.
    # TODO: parse ISO timestamp (e.g. "2026-04-14T00:00:00") using datetime.fromisoformat.
    # TODO: return EventRecord and ensure payload_size_bytes is an int.
    return EventRecord(
        event_id="",
        source_system="",
        event_ts=datetime(1970, 1, 1),
        payload_size_bytes=0,
    )


if __name__ == "__main__":
    sample = {
        "event_id": "evt-101",
        "source_system": "billing-api",
        "event_ts": "2026-04-14T00:00:00",
        "payload_size_bytes": 512,
    }
    print(parse_event(sample))
