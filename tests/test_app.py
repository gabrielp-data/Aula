"""Testes da API FastAPI (churn.app)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn.app import app

VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.5,
    "TotalCharges": 666.0,
}


def _fake_pipeline() -> dict:
    encoder = MagicMock()
    encoder.transform.side_effect = lambda df: df.select_dtypes(include="number")
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return {"encoder": encoder, "model": model}


@pytest.fixture()
def client():
    """TestClient com carregamento de artefato mockado."""
    with patch("churn.app._load_artifact", return_value=_fake_pipeline()):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


def test_predict_retorna_estrutura(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn" in body
    assert "churn_probability" in body
    assert "label" in body


def test_predict_probabilidade_no_intervalo(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_label_consistente(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    body = resp.json()
    assert body["label"] == ("Yes" if body["churn"] else "No")


def test_predict_campo_faltando(client):
    incompleto = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    resp = client.post("/predict", json=incompleto)
    assert resp.status_code == 422
