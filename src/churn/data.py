"""Carga, limpeza e validação dos dados.

Funções puras (recebem/retornam DataFrame, sem efeito colateral) compostas por
uma orquestração fina em ``load_clean``. A validação usa o schema pandera
definido em ``schema.py`` — dado ruim para o pipeline cedo, não silenciosamente.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.schema import ChurnSchema

EXPECTED_COLUMNS: frozenset[str] = frozenset({
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
})


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV cru do disco."""
    return pd.read_csv(path)


def drop_identifier(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Remove a coluna de identificador (não é feature)."""
    return df.drop(columns=[id_column])


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges (texto, com brancos) em número.

    Os brancos correspondem a clientes com ``tenure == 0`` (novos), cujo total
    gasto é, de fato, 0 — não um número mágico como no script original.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra o ChurnSchema (pandera).

    Usa ``lazy=True`` para coletar TODAS as violações de uma vez antes de
    levantar a exceção — facilita o diagnóstico de dado ruim em produção.
    """
    return ChurnSchema.validate(df, lazy=True)


def load_clean(data_path: Path, id_column: str, target: str) -> pd.DataFrame:
    """Pipeline de dados: carrega → remove id → coerce TotalCharges → valida."""
    df = load_raw(data_path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")
    df = coerce_total_charges(drop_identifier(df, id_column))
    return validate(df)
