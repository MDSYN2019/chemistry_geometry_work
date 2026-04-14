"""Exercise 13: map requirements to cloud-native architecture decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataProductSpec:
    latency_minutes: int
    daily_volume_gb: int
    needs_realtime_alerts: bool


def choose_ingestion(spec: DataProductSpec) -> str:
    """Return 'streaming' or 'batch'."""
    # TODO: choose streaming when realtime alerts are needed or latency <= 5.
    return "batch"


def choose_storage(spec: DataProductSpec) -> str:
    """Return 'warehouse' or 'data_lake'."""
    # TODO: choose data_lake when daily_volume_gb >= 500, else warehouse.
    return "warehouse"


def architecture_plan(spec: DataProductSpec) -> dict[str, str]:
    # TODO: return dict with ingestion, processing, storage, consumption.
    # processing should be 'incremental' for streaming, else 'scheduled'.
    # consumption should be 'bi+alerts' if realtime alerts needed, else 'bi'.
    return {}


if __name__ == "__main__":
    spec = DataProductSpec(latency_minutes=3, daily_volume_gb=120, needs_realtime_alerts=True)
    print(architecture_plan(spec))
