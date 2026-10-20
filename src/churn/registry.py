"""Model Registry — Encontro 7.

Registra a melhor run no MLflow Model Registry, promove aliases e lista versões.

Uso: ``uv run python -m churn.registry``
Depois: ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``
"""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "churn"


def register_model(run_id: str, artifact_path: str = "model") -> mlflow.entities.model_registry.ModelVersion:
    """Registra o artefato de uma run no Model Registry e retorna a versão."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.register_model(model_uri, MODEL_NAME)


def promote_champion(version: int, model_name: str = MODEL_NAME) -> None:
    """Marca `@champion` na versão informada."""
    MlflowClient().set_registered_model_alias(model_name, "champion", version)


def promote_challenger(version: int, model_name: str = MODEL_NAME) -> None:
    """Marca `@challenger` na versão informada."""
    MlflowClient().set_registered_model_alias(model_name, "challenger", version)


def load_champion(model_name: str = MODEL_NAME):
    """Carrega o modelo marcado como @champion via pyfunc."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")


def list_versions(model_name: str = MODEL_NAME) -> None:
    """Imprime versões registradas com seus aliases."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"  Nenhuma versão registrada para '{model_name}'.")
        return
    for v in versions:
        aliases = getattr(v, "aliases", [])
        print(f"  v{v.version}  run={v.run_id[:8]}  status={v.status}  aliases={aliases}")


def main() -> None:
    runs = mlflow.search_runs(
        experiment_names=["churn"],
        order_by=["metrics.roc_auc DESC"],
    )
    if runs.empty:
        print("Nenhuma run encontrada. Execute churn.model ou churn.experiments primeiro.")
        return

    best = runs.iloc[0]
    run_id = best["run_id"]
    roc_auc = float(best["metrics.roc_auc"])
    print(f"Melhor run: {run_id[:8]}  roc_auc={roc_auc:.4f}")

    mv = register_model(run_id)
    print(f"Modelo registrado → versão {mv.version}")

    promote_champion(int(mv.version))
    print(f"Alias @champion → v{mv.version}")

    print("\nVersões registradas:")
    list_versions()


if __name__ == "__main__":
    main()
