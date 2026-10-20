"""API REST de predição de churn — Encontro 12.

Endpoints:
  GET  /health   → status do modelo carregado
  POST /predict  → recebe dados do cliente, retorna probabilidade de churn

Uso:
    uv run uvicorn churn.app:app --reload --port 8000
    Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import settings
from churn.features import add_derived_features


_pipeline: dict = {}


def _load_artifact(path: Path) -> dict:
    """Carrega o pickle {model, encoder} do disco."""
    if not path.exists():
        raise RuntimeError(
            f"Artefato não encontrado: {path}. Execute 'uv run python -m churn.model' primeiro."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _pipeline.update(_load_artifact(settings.artifact_path))
    yield
    _pipeline.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Serve o modelo @champion treinado no pipeline MLOps.",
    version="0.1.0",
    lifespan=lifespan,
)


class CustomerIn(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionOut(BaseModel):
    churn: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    label: str


@app.get("/health", tags=["infra"])
def health():
    """Verifica se a API está no ar e o modelo foi carregado."""
    return {"status": "ok", "model_loaded": "model" in _pipeline}


@app.post("/predict", response_model=PredictionOut, tags=["inferência"])
def predict(customer: CustomerIn):
    """Recebe os dados de um cliente e retorna a probabilidade de churn."""
    if "model" not in _pipeline:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    df = pd.DataFrame([customer.model_dump()])
    df = add_derived_features(df)

    encoder = _pipeline["encoder"]
    model = _pipeline["model"]

    x = encoder.transform(df)
    proba = float(model.predict_proba(x)[0, 1])
    churn = proba >= 0.5

    return PredictionOut(churn=churn, churn_probability=proba, label="Yes" if churn else "No")
