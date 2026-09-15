"""Grid de experimentos rastreado com MLflow.

Uso: ``uv run python -m churn.experiments``

Roda 9 combinações (n_estimators × max_depth), registra params e métricas no
MLflow. Depois: ``uv run mlflow ui`` para comparar os runs em localhost:5000.
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import settings as cfg
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)


def run_grid() -> None:
    frame = add_derived_features(
        load_clean(cfg.data_path, cfg.id_column, cfg.target)
    )
    X, y = split_features_target(frame, cfg.target, cfg.positive_label)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
    )
    enc = CategoricalEncoder().fit(x_tr)
    x_tr_e, x_te_e = enc.transform(x_tr), enc.transform(x_te)

    mlflow.set_experiment("churn")
    for n in (100, 200, 500):
        for depth in (None, 8, 16):
            with mlflow.start_run(run_name=f"rf-n{n}-d{depth}"):
                mlflow.log_params({"n_estimators": n, "max_depth": depth})
                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    random_state=cfg.random_state,
                ).fit(x_tr_e, y_tr)
                mlflow.log_metrics(evaluate(model, x_te_e, y_te))


def find_best() -> None:
    runs = mlflow.search_runs(
        experiment_names=["churn"],
        order_by=["metrics.roc_auc DESC"],
    )
    best = runs.iloc[0]
    print(
        f"Melhor run: {best['run_id'][:8]}  "
        f"n={best['params.n_estimators']}  "
        f"depth={best['params.max_depth']}  "
        f"roc_auc={float(best['metrics.roc_auc']):.4f}"
    )


if __name__ == "__main__":
    run_grid()
    find_best()
