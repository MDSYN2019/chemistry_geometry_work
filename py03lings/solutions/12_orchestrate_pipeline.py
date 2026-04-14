"""Solution 12: orchestrate pipeline stages with simple SLA checks."""

from datetime import datetime


def run_stage(stage_name: str, duration_secs: int) -> dict[str, int | str]:
    return {"stage": stage_name, "duration_secs": duration_secs, "status": "success"}


def total_duration(stage_results: list[dict[str, int | str]]) -> int:
    return sum(int(stage["duration_secs"]) for stage in stage_results)


def within_sla(stage_results: list[dict[str, int | str]], sla_secs: int) -> bool:
    return total_duration(stage_results) <= sla_secs


def build_run_report(stage_results: list[dict[str, int | str]], sla_secs: int) -> dict[str, object]:
    return {
        "started_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "total_secs": total_duration(stage_results),
        "sla_secs": sla_secs,
        "sla_met": within_sla(stage_results, sla_secs),
    }


if __name__ == "__main__":
    stages = [
        run_stage("ingestion", 120),
        run_stage("transformation", 180),
        run_stage("publish", 40),
    ]
    print(build_run_report(stages, sla_secs=400))
