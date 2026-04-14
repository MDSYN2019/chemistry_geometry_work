"""Exercise 12: orchestrate pipeline stages with simple SLA checks."""

from datetime import datetime


def run_stage(stage_name: str, duration_secs: int) -> dict[str, int | str]:
    return {"stage": stage_name, "duration_secs": duration_secs, "status": "success"}


def total_duration(stage_results: list[dict[str, int | str]]) -> int:
    # TODO: sum duration_secs for all stage results.
    return 0


def within_sla(stage_results: list[dict[str, int | str]], sla_secs: int) -> bool:
    # TODO: return True when total_duration is <= sla_secs.
    return False


def build_run_report(stage_results: list[dict[str, int | str]], sla_secs: int) -> dict[str, object]:
    """Produce a run summary for downstream consumers."""
    # TODO: create and return dict with keys: started_at_utc, total_secs, sla_secs, sla_met.
    return {}


if __name__ == "__main__":
    stages = [
        run_stage("ingestion", 120),
        run_stage("transformation", 180),
        run_stage("publish", 40),
    ]
    print(build_run_report(stages, sla_secs=400))
