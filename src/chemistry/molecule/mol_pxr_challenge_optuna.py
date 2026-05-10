"""
Classical ML baseline for molecular property prediction with Optuna tuning.

This script builds fingerprint descriptors from SMILES and tunes a classical
model (ExtraTreesRegressor) that is often strong on tabular molecular features.

Usage
-----
python src/chemistry/molecule/mol_pxr_challenge_optuna.py \
  --train_csv data/train.csv --target pxr_value --smiles_col smiles
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


@dataclass
class FeaturizerConfig:
    fp_radius: int = 2
    fp_nbits: int = 2048


def smiles_to_features(smiles: str, cfg: FeaturizerConfig) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(cfg.fp_nbits + 6, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, cfg.fp_radius, nBits=cfg.fp_nbits)
    arr = np.zeros((cfg.fp_nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    physchem = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.RingCount(mol),
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr, physchem])


def build_feature_matrix(df: pd.DataFrame, smiles_col: str, cfg: FeaturizerConfig) -> np.ndarray:
    feats = [smiles_to_features(s, cfg) for s in df[smiles_col].astype(str).values]
    return np.vstack(feats)


def make_objective(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1600, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.25, 0.4, 0.6]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
            "n_jobs": -1,
            "random_state": seed,
        }

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_mae = []
        for tr_idx, va_idx in cv.split(X):
            model = ExtraTreesRegressor(**params)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_mae.append(mean_absolute_error(y[va_idx], pred))

        return float(np.mean(fold_mae))

    return objective


def train_best_model(X: np.ndarray, y: np.ndarray, best_params: dict, seed: int) -> ExtraTreesRegressor:
    params = dict(best_params)
    params.update({"n_jobs": -1, "random_state": seed})
    model = ExtraTreesRegressor(**params)
    model.fit(X, y)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--smiles_col", default="smiles")
    p.add_argument("--target", required=True)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--cv_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_out", default="mol_pxr_extratrees.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.train_csv)
    df = df[[args.smiles_col, args.target]].dropna().reset_index(drop=True)

    cfg = FeaturizerConfig()
    X = build_feature_matrix(df, args.smiles_col, cfg)
    y = df[args.target].astype(float).values

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="pxr_classical_ml")
    study.optimize(make_objective(X, y, args.cv_splits, args.seed), n_trials=args.n_trials)

    print("Best CV MAE:", study.best_value)
    print("Best params:", study.best_params)

    model = train_best_model(X, y, study.best_params, args.seed)

    import joblib

    artifact = {
        "model": model,
        "featurizer": cfg,
        "smiles_col": args.smiles_col,
        "target": args.target,
        "best_cv_mae": study.best_value,
    }
    joblib.dump(artifact, args.model_out)
    print(f"Saved model artifact to: {args.model_out}")


if __name__ == "__main__":
    main()
