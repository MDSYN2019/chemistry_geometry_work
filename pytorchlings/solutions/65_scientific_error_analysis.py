from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SliceReport:
    name: str
    count: int
    mae: float


def make_predictions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_true = torch.tensor([0.10, 0.20, 0.18, 0.32, 0.28, 0.41, 0.37, 0.52, 0.50, 0.62])
    y_pred = torch.tensor([0.12, 0.25, 0.16, 0.30, 0.22, 0.44, 0.40, 0.49, 0.57, 0.66])
    oxygen_fraction = torch.tensor([0.0, 0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_pred, oxygen_fraction


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred)).item())


def build_slice_reports(y_true: torch.Tensor, y_pred: torch.Tensor, oxygen_fraction: torch.Tensor) -> list[SliceReport]:
    reports: list[SliceReport] = []

    masks = {
        "low oxygen": oxygen_fraction < 0.3,
        "mid oxygen": (oxygen_fraction >= 0.3) & (oxygen_fraction < 0.7),
        "high oxygen": oxygen_fraction >= 0.7,
    }

    for name, mask in masks.items():
        y_t = y_true[mask]
        y_p = y_pred[mask]
        reports.append(SliceReport(name=name, count=int(mask.sum().item()), mae=mae(y_t, y_p)))
    return reports
