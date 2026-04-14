"""Exercise 14: implement lightweight governance checks."""

from typing import Any


def has_required_metadata(dataset: dict[str, Any]) -> bool:
    """Require owner, pii_classification, and retention_days keys."""
    # TODO: return True only if all required metadata keys are present.
    return False


def build_lineage_edge(upstream: str, downstream: str) -> tuple[str, str]:
    # TODO: return a lineage tuple (upstream, downstream).
    return ("", "")


def can_read_dataset(user_roles: set[str], dataset_acl: set[str]) -> bool:
    """Allow read when user has at least one required role."""
    # TODO: return True if there is any overlap between user_roles and dataset_acl.
    return False


if __name__ == "__main__":
    dataset = {"owner": "data-platform", "pii_classification": "internal", "retention_days": 365}
    print(has_required_metadata(dataset))
    print(build_lineage_edge("raw.orders", "mart.daily_orders"))
    print(can_read_dataset({"analyst", "finance"}, {"finance", "admin"}))
