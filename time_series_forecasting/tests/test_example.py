import unittest

from time_series_forecasting.example import (
    PERIODS_PER_DAY,
    PERIODS_PER_WEEK,
    equivalent_period_prediction,
    evaluate_last_week,
    generate_example,
    mae,
    rmse,
    seasonal_prediction,
)


class ForecastExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = generate_example()

    def test_generator_has_expected_half_hours(self) -> None:
        self.assertEqual(len(self.observations), 56 * PERIODS_PER_DAY)
        self.assertEqual(
            self.observations[1].timestamp - self.observations[0].timestamp,
            self.observations[2].timestamp - self.observations[1].timestamp,
        )

    def test_seasonal_prediction_uses_requested_past_row(self) -> None:
        target = PERIODS_PER_WEEK + 10
        self.assertEqual(
            seasonal_prediction(self.observations, target, PERIODS_PER_WEEK),
            self.observations[10].demand_mw,
        )

    def test_seasonal_prediction_rejects_unavailable_history(self) -> None:
        with self.assertRaises(ValueError):
            seasonal_prediction(self.observations, 10, PERIODS_PER_WEEK)

    def test_equivalent_period_requires_complete_history(self) -> None:
        with self.assertRaises(ValueError):
            equivalent_period_prediction(self.observations, PERIODS_PER_WEEK)

    def test_metrics(self) -> None:
        self.assertEqual(mae([1.0, 4.0], [2.0, 2.0]), 1.5)
        self.assertAlmostEqual(rmse([1.0, 4.0], [2.0, 2.0]), (2.5) ** 0.5)

    def test_chronological_evaluation_returns_nonnegative_metrics(self) -> None:
        result = evaluate_last_week(
            self.observations, len(self.observations) - PERIODS_PER_WEEK
        )
        self.assertGreaterEqual(result["mae_mw"], 0)
        self.assertGreaterEqual(result["rmse_mw"], result["mae_mw"])


if __name__ == "__main__":
    unittest.main()
