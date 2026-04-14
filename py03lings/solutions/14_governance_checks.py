"""Solution 14: implement lightweight governance checks."""

from typing import Any


def has_required_metadata(dataset: dict[str, Any]) -> bool:
    required = {"owner", "pii_classification", "retention_days"}
    return required.issubset(dataset.keys())


def build_lineage_edge(upstream: str, downstream: str) -> tuple[str, str]:
    return upstream, downstream


def can_read_dataset(user_roles: set[str], dataset_acl: set[str]) -> bool:
    return not user_roles.isdisjoint(dataset_acl)


if __name__ == "__main__":
    dataset = {"owner": "data-platform", "pii_classification": "internal", "retention_days": 365}
    print(has_required_metadata(dataset))
    print(build_lineage_edge("raw.orders", "mart.daily_orders"))
    print(can_read_dataset({"analyst", "finance"}, {"finance", "admin"}))
