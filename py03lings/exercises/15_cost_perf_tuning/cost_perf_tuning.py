"""Exercise 15: reason about partition pruning, indexing, and caching."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPattern:
    filters_on_partition: bool
    filters_on_indexed_column: bool
    repeat_per_hour: int


def estimated_scan_gb(total_table_gb: float, pattern: QueryPattern) -> float:
    """Estimate read volume after partition/index pruning."""
    # TODO: start with total_table_gb.
    # TODO: if filters_on_partition, reduce scan to 10%.
    # TODO: if filters_on_indexed_column, reduce current scan to 50%.
    # TODO: round to 2 decimals.
    return total_table_gb


def should_enable_cache(pattern: QueryPattern) -> bool:
    # TODO: return True when repeat_per_hour >= 4.
    return False


def cost_optimization_plan(total_table_gb: float, pattern: QueryPattern) -> dict[str, object]:
    # TODO: include keys scan_gb, enable_cache, recommendation.
    # recommendation should be one short sentence mentioning partitioning and indexing.
    return {}


if __name__ == "__main__":
    pattern = QueryPattern(filters_on_partition=True, filters_on_indexed_column=True, repeat_per_hour=8)
    print(cost_optimization_plan(total_table_gb=1000.0, pattern=pattern))
