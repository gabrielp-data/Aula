import pandas as pd
import pytest
import pandera.errors

from churn.schema import ChurnSchema


def test_schema_valida_dado_limpo(raw_frame):
    """Schema aceita um DataFrame válido sem levantar erro."""
    from churn.data import coerce_total_charges, drop_identifier
    df = coerce_total_charges(drop_identifier(raw_frame, "customerID"))
    validated = ChurnSchema.validate(df, lazy=True)
    assert len(validated) == len(df)


def test_schema_rejeita_tenure_invalido():
    """tenure fora de [0, 72] viola o contrato."""
    bad = pd.DataFrame({
        "gender": ["Male"], "SeniorCitizen": [0], "Partner": ["Yes"],
        "Dependents": ["No"], "tenure": [999],
        "PhoneService": ["Yes"], "MultipleLines": ["No"],
        "InternetService": ["DSL"], "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"], "DeviceProtection": ["No"],
        "TechSupport": ["No"], "StreamingTV": ["No"], "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [55.5], "TotalCharges": [666.0], "Churn": ["No"],
    })
    with pytest.raises(pandera.errors.SchemaErrors):
        ChurnSchema.validate(bad, lazy=True)


def test_schema_rejeita_contract_invalido():
    """Contract com valor fora do isin viola o contrato."""
    bad = pd.DataFrame({
        "gender": ["Female"], "SeniorCitizen": [0], "Partner": ["No"],
        "Dependents": ["No"], "tenure": [10],
        "PhoneService": ["Yes"], "MultipleLines": ["No"],
        "InternetService": ["No"], "OnlineSecurity": ["No internet service"],
        "OnlineBackup": ["No internet service"], "DeviceProtection": ["No internet service"],
        "TechSupport": ["No internet service"], "StreamingTV": ["No internet service"],
        "StreamingMovies": ["No internet service"],
        "Contract": ["Vitalício"],  # categoria inválida
        "PaperlessBilling": ["No"],
        "PaymentMethod": ["Mailed check"],
        "MonthlyCharges": [20.0], "TotalCharges": [200.0], "Churn": ["No"],
    })
    with pytest.raises(pandera.errors.SchemaErrors):
        ChurnSchema.validate(bad, lazy=True)
