"""Solution 13: map requirements to cloud-native architecture decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataProductSpec:
    latency_minutes: int
    daily_volume_gb: int
    needs_realtime_alerts: bool


def choose_ingestion(spec: DataProductSpec) -> str:
    if spec.needs_realtime_alerts or spec.latency_minutes <= 5:
        return "streaming"
    return "batch"


def choose_storage(spec: DataProductSpec) -> str:
    return "data_lake" if spec.daily_volume_gb >= 500 else "warehouse"


def architecture_plan(spec: DataProductSpec) -> dict[str, str]:
    ingestion = choose_ingestion(spec)
    return {
        "ingestion": ingestion,
        "processing": "incremental" if ingestion == "streaming" else "scheduled",
        "storage": choose_storage(spec),
        "consumption": "bi+alerts" if spec.needs_realtime_alerts else "bi",
    }


if __name__ == "__main__":
    spec = DataProductSpec(latency_minutes=3, daily_volume_gb=120, needs_realtime_alerts=True)
    print(architecture_plan(spec))
