"""Solution 61: Optuna search over non-PyTorch baselines."""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def make_data():
    x, y = make_classification(
        n_samples=700,
        n_features=20,
        n_informative=10,
        n_redundant=3,
        class_sep=1.2,
        flip_y=0.02,
        random_state=42,
    )
    return x, y


def objective(trial, x, y) -> float:
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 12)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=123,
        n_jobs=-1,
    )

    scores = cross_val_score(model, x, y, cv=5, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def run_study(n_trials: int = 30):
    import optuna

    x, y = make_data()
    sampler = optuna.samplers.RandomSampler(seed=1234)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="rf_baseline_tuning")
    study.optimize(lambda trial: objective(trial, x, y), n_trials=n_trials)
    return study


if __name__ == "__main__":
    study = run_study(n_trials=10)
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)
