"""Contrato de dados do dataset Telco Customer Churn (pandera).

Valide ANTES de transformar features — dado ruim para o pipeline cedo,
não depois de a métrica cair em silêncio.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    """Schema para o dataset após remoção do customerID e coerção de TotalCharges."""

    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0, le=72)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(
        isin=["Yes", "No", "No phone service"]
    )
    InternetService: Series[str] = pa.Field(
        isin=["DSL", "Fiber optic", "No"]
    )
    OnlineSecurity: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    OnlineBackup: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    DeviceProtection: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    TechSupport: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    StreamingTV: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    StreamingMovies: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0)
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        coerce = True
        strict = False
