"""
Trains the direction-probability model using walk-forward validation.

Uses sklearn's HistGradientBoostingClassifier: handles NaNs natively, fast
on tabular data, no heavyweight compiled dependency (unlike LightGBM/
XGBoost) to install in constrained environments. Swap in another sklearn-
API-compatible classifier here if you want to experiment -- nothing else
in the platform depends on which model class this is, only that it
exposes .fit(X, y) and .predict_proba(X).

The critical output of this module is `oof_predictions`: out-of-fold
probabilities where every prediction was made by a model that never saw
that bar (or the horizon of bars after it) during training. That's what
the backtester scores against -- not in-sample fit, which would just
measure how well the model memorized history.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
from pathlib import Path

from models.dataset import build_dataset, walk_forward_folds, Fold

REGISTRY_DIR = Path(__file__).parent / "registry"


@dataclass
class FoldResult:
    fold_number: int
    train_size: int
    test_size: int
    auc: float
    brier: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class TrainingResult:
    symbol: str
    oof_predictions: pd.Series      # walk-forward out-of-fold P(up-move) per timestamp
    fold_results: list[FoldResult]
    final_model: HistGradientBoostingClassifier
    feature_names: list[str]
    mean_auc: float


def _make_model(early_stopping: bool = True) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=early_stopping,
        validation_fraction=0.15,
        random_state=42,
    )


def train_symbol(
    df: pd.DataFrame,
    symbol: str,
    horizon: int = 12,
    cost_threshold_bps: float = 30.0,
    n_folds: int = 5,
) -> TrainingResult:
    X, y = build_dataset(df, horizon=horizon, cost_threshold_bps=cost_threshold_bps)
    folds = walk_forward_folds(X.index, n_folds=n_folds, embargo_bars=horizon)

    oof = pd.Series(index=X.index, dtype=float)
    fold_results = []

    for fold in folds:
        X_train, y_train = X.loc[fold.train_idx], y.loc[fold.train_idx]
        X_test, y_test = X.loc[fold.test_idx], y.loc[fold.test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            # Degenerate fold (e.g. every label the same) -- skip rather
            # than report a meaningless AUC.
            continue

        # HistGradientBoostingClassifier's early_stopping=True internally
        # carves out a stratified validation split. Early walk-forward
        # folds have small training windows, and if the minority class
        # has too few examples, sklearn's stratified split can't put >=2
        # of them in its validation slice and raises. Rather than let
        # that crash the whole training run (or silently skip an early
        # fold that's otherwise perfectly usable), fall back to fitting
        # without early stopping for just that fold -- slightly more
        # overfit-prone, but still an honest out-of-fold prediction.
        minority_count = y_train.value_counts().min()
        model = _make_model(early_stopping=bool(minority_count >= 20))
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        oof.loc[fold.test_idx] = proba

        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        fold_results.append(
            FoldResult(
                fold_number=fold.fold_number,
                train_size=len(X_train),
                test_size=len(X_test),
                auc=auc,
                brier=brier,
                train_start=str(fold.train_idx[0]),
                train_end=str(fold.train_idx[-1]),
                test_start=str(fold.test_idx[0]),
                test_end=str(fold.test_idx[-1]),
            )
        )

    valid_oof = oof.dropna()
    mean_auc = float(np.mean([f.auc for f in fold_results])) if fold_results else float("nan")

    # Final production model: retrain on ALL available data. This model is
    # what actually goes live -- but its performance is judged using the
    # out-of-fold numbers above, never by scoring it on its own training set.
    final_minority_count = y.value_counts().min()
    final_model = _make_model(early_stopping=bool(final_minority_count >= 20))
    final_model.fit(X, y)

    return TrainingResult(
        symbol=symbol,
        oof_predictions=valid_oof,
        fold_results=fold_results,
        final_model=final_model,
        feature_names=list(X.columns),
        mean_auc=mean_auc,
    )


def save_model(result: TrainingResult) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = result.symbol.replace("/", "-")
    path = REGISTRY_DIR / f"{safe_symbol}_model.joblib"
    joblib.dump(
        {"model": result.final_model, "feature_names": result.feature_names},
        path,
    )
    return path


def load_model(symbol: str) -> dict:
    safe_symbol = symbol.replace("/", "-")
    path = REGISTRY_DIR / f"{safe_symbol}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model found for {symbol} at {path}")
    return joblib.load(path)
