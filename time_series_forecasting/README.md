# GB demand forecasting practice lab

This lab turns the NESO day-ahead demand idea into a sequence of small,
testable exercises.  The included example deliberately uses only Python's
standard library and generated data, so it can be run before downloading a
large dataset.  Replace the generated observations with curated NESO rows
once the mechanics are understood.

## Run the worked example

From the repository root:

```bash
python time_series_forecasting/example.py
python -m unittest discover -s time_series_forecasting/tests -v
```

The example generates eight weeks of half-hourly demand, makes a chronological
split, predicts the final week with yesterday and last-week baselines, and
prints MAE/RMSE.  It also demonstrates a four-week equivalent-period average.
No random split or future target value is used.

## Production question to answer first

Write down the forecast issue time.  If all 48 values for tomorrow are issued
at 16:00 today, tomorrow's `lag_1` and `lag_2` targets do **not** exist at issue
time.  They are valid during one-step-ahead training but leak information in a
direct day-ahead backtest.  Calendar features, yesterday/last-week demand, and
data or forecasts published before the issue time are safe candidates.  Keep
both `event_time` and `available_at` in the feature provenance.

NESO settlement periods are also not always interchangeable with 48 naïve
local half-hours: clock-change days may have 46 or 50 periods.  Preserve the
source settlement date and period, define a documented UTC conversion policy,
and test both daylight-saving transitions.

## Suggested project milestones

Each milestone should end with code, tests, a short decision record, and a
sample command rather than only a notebook.

### 1. Ingestion and canonical data

1. Download one annual historic-demand CSV and profile its headers, types,
   nulls, duplicates, and date range.
2. Map source columns (for example `ND`, `TSD`, wind, and solar) to stable
   canonical names without silently accepting unknown schemas.
3. Create `demand_observations` at settlement-date/period grain.  Decide
   whether the natural key or a UTC timestamp is authoritative on DST days.
4. Make ingestion idempotent.  Re-ingesting an identical file must change no
   rows; a retrospectively corrected row must be recorded and auditable.
5. Hash the exact raw file and curated training extract.  Store source URL,
   retrieval time, row count, checksum, and schema version in a manifest.
6. Add incremental CKAN/API retrieval with pagination, retries, a watermark,
   response caching, and respectful rate limiting.

**Acceptance checks:** unique canonical key; expected periods per settlement
day; no impossible negative demand; deterministic output hash for identical
input; clear quarantine reasons for rejected rows.

### 2. Exploration and time-series diagnostics

7. Plot median demand and quantile bands by settlement period.
8. Compare weekday/weekend, month/season, and holiday profiles.
9. Locate morning/evening peaks and quantify how their time and magnitude
   change through the year.
10. Measure autocorrelation at 1, 2, 48, 96, and 336 half-hours.
11. Find gaps, duplicated intervals, constant runs, outliers, and DST days.
12. Compare annual distributions and rolling means to identify structural
   change.  Write hypotheses rather than removing unusual values immediately.

### 3. Point-in-time-correct features

13. Implement calendar features plus lag 48, lag 96, and lag 336.
14. Implement same-period averages over the preceding two and four weeks.
15. Add rolling mean/std features using only observations available before the
   forecast issue time.
16. Represent every feature with `event_time`, `available_at`, source, and
   transformation version.  Add a test that fails if `available_at` is later
   than the simulated issue time.
17. Add holidays as a versioned calendar input.  Decide how Christmas-like
   days should borrow history.
18. Compare historical wind/solar features with genuine day-ahead weather or
   generation forecasts; never substitute future outturns.

### 4. Baselines and evaluation

19. Reproduce the included yesterday, last-week, and four-week baselines on
   the curated data.
20. Add a profile baseline: median by settlement period, weekday class, and
   season, fitted only on the training window.
21. Implement MAE, RMSE, bias, MAPE with a documented zero policy, and peak
   MAE.  Report MW as well as percentages.
22. Break errors down by horizon (1–48), settlement period, weekday, month,
   peak/off-peak, holiday, and demand quantile.
23. Join NESO historic day-ahead forecasts using target time **and issuance
   time/days-ahead**.  Compare all models only on their common target set.
24. Add forecast skill relative to last-week demand and use paired daily errors
   or confidence intervals so small metric differences are not overclaimed.

### 5. Models and backtesting

25. Fit regularised linear regression and inspect coefficients/residuals.
26. Fit LightGBM or XGBoost with a reproducible seed and constrained feature
   contract.  Tune only against validation/backtest folds.
27. Implement expanding-window monthly backtests.  Log training cutoff,
   simulated issue time, target interval, and model version for every fold.
28. Compare expanding and rolling training windows under structural change.
29. Only after strong tabular evaluation, try NeuralProphet, N-BEATS/N-HiTS,
   or PatchTST using the identical folds and target set.
30. Test probabilistic forecasts (for example 10th/50th/90th quantiles) with
   pinball loss, interval coverage, and interval width.

### 6. Reproducible training and prediction

31. Package `ingest`, `validate`, `features`, `train`, `predict`, and
   `evaluate` as CLI commands.  Make each safe to rerun.
32. Save model artefact, dataset hash, Git commit, dependency lock hash,
   feature contract, training bounds, hyperparameters, metrics, and runtime.
33. Create a forecast manifest that proves all expected targets were emitted
   exactly once and records the issue time and horizon definition.
34. Store predictions append-only.  Reconcile actuals later without rewriting
   what the model originally predicted.
35. Build a clean-environment test that recreates a known model and predictions
   from the recorded inputs.

### 7. Monitoring, retraining, and promotion

36. Emit pipeline metrics for freshness, row counts, rejected rows, missing
   periods, stage duration, and 48-target forecast completeness.
37. When actuals arrive, monitor MAE/RMSE/bias, peak errors, error by horizon,
   seasonal-baseline skill, and rolling 7/28-day changes.
38. Separate data drift, feature drift, concept/performance drift, and pipeline
   failure alerts.  Give each alert an owner and runbook.
39. Implement a monthly candidate-training policy and a separate degradation
   trigger.  Neither trigger should deploy automatically.
40. Gate promotion on data quality, minimum backtest coverage, improvement
   across multiple slices, latency, and artefact reproducibility.  Support
   explicit approval and rollback.

## General time-series extensions

Use the same framework on other domains to show transferable understanding:

41. **Multiple seasonalities:** forecast hourly bicycle demand with daily and
    weekly cycles; compare Fourier/calendar features with seasonal lags.
42. **Intermittent demand:** forecast spare-parts demand and compare naïve,
    Croston-style, and count models using inventory-relevant metrics.
43. **Hierarchies:** forecast demand by region and national total, then
    reconcile bottom-up and top-down forecasts so totals agree.
44. **Exogenous variables:** forecast building load using weather forecasts;
    simulate forecast error rather than using observed future temperature.
45. **Missingness:** inject gaps and compare forward fill, seasonal imputation,
    model-native missing values, and explicit missingness flags.
46. **Regime change:** introduce a level shift and compare expanding windows,
    rolling windows, decay weights, and change-point alerts.
47. **Anomalies:** distinguish bad sensors from real demand spikes and show how
    each treatment changes training and monitoring.
48. **Probabilistic decisions:** translate quantile forecasts into an asymmetric
    over/under-forecast cost and evaluate calibration separately from cost.
49. **Online evaluation:** reproduce delayed labels and retrospective source
    corrections without mutating the original score history.
50. **Capacity and reliability:** load-test batch inference, make publication
    atomic, retry safely, and prove that a partial run cannot be promoted.

## Definition of done for the portfolio project

A reviewer can run one command on a small fixture; trace every forecast to its
data, features, code, and artefact; reproduce chronological evaluation against
seasonal and NESO baselines; see leakage and DST tests; and understand how a
candidate is monitored, approved, promoted, and rolled back.
