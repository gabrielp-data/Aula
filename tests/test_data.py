import pandas as pd
import pytest

from churn.data import coerce_total_charges, drop_identifier, validate


def test_coerce_total_charges_converte_branco_em_zero(raw_frame):
    out = coerce_total_charges(raw_frame)
    assert out["TotalCharges"].dtype.kind == "f"
    # a linha com tenure 0 tinha TotalCharges em branco -> 0.0
    assert out.loc[raw_frame["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_coerce_e_pura(raw_frame):
    _ = coerce_total_charges(raw_frame)
    # original intacto: ainda é string/object (não float), independente da versão do pandas
    assert raw_frame["TotalCharges"].dtype.kind in ("O", "U") or hasattr(
        raw_frame["TotalCharges"].dtype, "na_value"
    )


def test_validate_falha_sem_coluna(raw_frame):
    from churn.data import coerce_total_charges, drop_identifier
    import pandera.errors
    df = coerce_total_charges(drop_identifier(raw_frame, "customerID"))
    with pytest.raises((ValueError, pandera.errors.SchemaErrors)):
        validate(df.drop(columns=["Contract"]))


def test_drop_identifier(raw_frame):
    out = drop_identifier(raw_frame, "customerID")
    assert "customerID" not in out.columns
