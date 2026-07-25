"""Small, dependency-free example of leakage-safe seasonal baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from statistics import mean
from typing import Iterable, Sequence

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = 7 * PERIODS_PER_DAY


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    demand_mw: float


def generate_example(days: int = 56) -> list[Observation]:
    """Generate a deterministic series with daily and weekly seasonality."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    observations = []
    for index in range(days * PERIODS_PER_DAY):
        period = index % PERIODS_PER_DAY
        weekday = (start + timedelta(minutes=30 * index)).weekday()
        morning_peak = max(0, 1 - abs(period - 16) / 8) * 4_000
        evening_peak = max(0, 1 - abs(period - 35) / 9) * 6_000
        weekend_reduction = 3_500 if weekday >= 5 else 0
        slow_trend = index * 0.25
        demand = 23_000 + morning_peak + evening_peak - weekend_reduction + slow_trend
        observations.append(
            Observation(start + timedelta(minutes=30 * index), demand)
        )
    return observations


def seasonal_prediction(
    observations: Sequence[Observation], target_index: int, lag: int
) -> float:
    """Predict a target from an earlier observation, rejecting future access."""
    source_index = target_index - lag
    if lag <= 0 or source_index < 0:
        raise ValueError("seasonal lag must refer to an available earlier value")
    return observations[source_index].demand_mw


def equivalent_period_prediction(
    observations: Sequence[Observation], target_index: int, weeks: int = 4
) -> float:
    """Average the same half-hour in each of the preceding complete weeks."""
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    indices = [target_index - week * PERIODS_PER_WEEK for week in range(1, weeks + 1)]
    if min(indices) < 0:
        raise ValueError("not enough history for equivalent-period baseline")
    return mean(observations[index].demand_mw for index in indices)


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted, strict=True))
    if not pairs:
        raise ValueError("metrics require at least one pair")
    return mean(abs(observed - forecast) for observed, forecast in pairs)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted, strict=True))
    if not pairs:
        raise ValueError("metrics require at least one pair")
    return sqrt(mean((observed - forecast) ** 2 for observed, forecast in pairs))


def evaluate_last_week(observations: Sequence[Observation], test_start: int) -> dict[str, float]:
    """Evaluate only the chronological suffix beginning at ``test_start``."""
    if test_start < PERIODS_PER_WEEK or test_start >= len(observations):
        raise ValueError("test_start must leave a non-empty test set and one week of history")
    actual = [row.demand_mw for row in observations[test_start:]]
    predicted = [
        seasonal_prediction(observations, index, PERIODS_PER_WEEK)
        for index in range(test_start, len(observations))
    ]
    return {"mae_mw": mae(actual, predicted), "rmse_mw": rmse(actual, predicted)}


def main() -> None:
    observations = generate_example()
    test_start = len(observations) - PERIODS_PER_WEEK
    actual = [row.demand_mw for row in observations[test_start:]]
    baselines = {
        "yesterday": [
            seasonal_prediction(observations, index, PERIODS_PER_DAY)
            for index in range(test_start, len(observations))
        ],
        "last_week": [
            seasonal_prediction(observations, index, PERIODS_PER_WEEK)
            for index in range(test_start, len(observations))
        ],
        "four_week_average": [
            equivalent_period_prediction(observations, index)
            for index in range(test_start, len(observations))
        ],
    }
    print("Chronological holdout: final 7 days (336 half-hours)")
    for name, predicted in baselines.items():
        print(f"{name:>17}: MAE={mae(actual, predicted):8.2f} MW  RMSE={rmse(actual, predicted):8.2f} MW")


if __name__ == "__main__":
    main()
