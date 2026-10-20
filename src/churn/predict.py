"""Inferência em lote usando o modelo @champion do Registry — Encontro 8.

Carrega o modelo empacotado via pyfunc (formato agnóstico de framework) e
aplica sobre dados novos em CSV.

Uso:
    uv run python -m churn.predict --input data/churn.csv
    uv run python -m churn.predict --input data/churn.csv --output predictions.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from churn.config import settings
from churn.data import coerce_total_charges, drop_identifier
from churn.features import add_derived_features, split_features_target
from churn.registry import load_champion


def predict(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Carrega @champion e pontua o CSV de entrada."""
    raw = pd.read_csv(input_path)
    df = add_derived_features(coerce_total_charges(drop_identifier(raw, settings.id_column)))
    features, _ = split_features_target(df, settings.target, settings.positive_label)

    model = load_champion()
    predictions = model.predict(features)

    result = raw[[settings.id_column]].copy()
    result["churn_pred"] = predictions

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Predições salvas em {output_path}")
    else:
        print(result.to_string(index=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score com o modelo @champion")
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", default=None, help="CSV de saída (opcional)")
    args = parser.parse_args()
    predict(args.input, args.output)


if __name__ == "__main__":
    main()
