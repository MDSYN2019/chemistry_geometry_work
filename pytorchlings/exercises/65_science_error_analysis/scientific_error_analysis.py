"""Exercise 65: scientific error analysis for model communication.

Goal:
- compute slice metrics by chemistry-like regime
- turn raw outputs into concise, shareable diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SliceReport:
    name: str
    count: int
    mae: float


def make_predictions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (y_true, y_pred, oxygen_fraction)."""
    y_true = torch.tensor([0.10, 0.20, 0.18, 0.32, 0.28, 0.41, 0.37, 0.52, 0.50, 0.62])
    y_pred = torch.tensor([0.12, 0.25, 0.16, 0.30, 0.22, 0.44, 0.40, 0.49, 0.57, 0.66])
    oxygen_fraction = torch.tensor([0.0, 0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_pred, oxygen_fraction


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # TODO: implement mean absolute error.
    return 0.0


def build_slice_reports(y_true: torch.Tensor, y_pred: torch.Tensor, oxygen_fraction: torch.Tensor) -> list[SliceReport]:
    """Create low/mid/high oxygen slice reports.

    low:  oxygen_fraction < 0.3
    mid:  0.3 <= oxygen_fraction < 0.7
    high: oxygen_fraction >= 0.7
    """
    reports: list[SliceReport] = []
    # TODO: create boolean masks for each slice.
    # TODO: compute count + MAE per slice and append SliceReport entries.
    return reports


def make_markdown_summary(reports: list[SliceReport], overall_mae: float) -> str:
    lines = ["# Validation Error Summary", "", f"Overall MAE: {overall_mae:.4f}", "", "| Slice | Count | MAE |", "|---|---:|---:|"]
    for r in reports:
        lines.append(f"| {r.name} | {r.count} | {r.mae:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    y_true, y_pred, oxygen_fraction = make_predictions()
    overall = mae(y_true, y_pred)
    reports = build_slice_reports(y_true, y_pred, oxygen_fraction)
    print(make_markdown_summary(reports, overall))
