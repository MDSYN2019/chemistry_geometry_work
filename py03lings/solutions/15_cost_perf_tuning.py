"""Solution 15: reason about partition pruning, indexing, and caching."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPattern:
    filters_on_partition: bool
    filters_on_indexed_column: bool
    repeat_per_hour: int


def estimated_scan_gb(total_table_gb: float, pattern: QueryPattern) -> float:
    scan = total_table_gb
    if pattern.filters_on_partition:
        scan *= 0.10
    if pattern.filters_on_indexed_column:
        scan *= 0.50
    return round(scan, 2)


def should_enable_cache(pattern: QueryPattern) -> bool:
    return pattern.repeat_per_hour >= 4


def cost_optimization_plan(total_table_gb: float, pattern: QueryPattern) -> dict[str, object]:
    scan = estimated_scan_gb(total_table_gb, pattern)
    return {
        "scan_gb": scan,
        "enable_cache": should_enable_cache(pattern),
        "recommendation": "Partition by date and index high-selectivity keys to reduce scan cost.",
    }


if __name__ == "__main__":
    pattern = QueryPattern(filters_on_partition=True, filters_on_indexed_column=True, repeat_per_hour=8)
    print(cost_optimization_plan(total_table_gb=1000.0, pattern=pattern))
