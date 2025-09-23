import pandas as pd
import pytest

from sample_data import load_sample_dataset
from services import fill_missing_months, compute_year_rolling, compute_slopes


def test_sample_dataset_structure():
    df = load_sample_dataset()
    required_cols = {"product_code", "product_name", "month", "sales_amount_jpy", "is_missing"}
    assert required_cols.issubset(df.columns)
    assert df["product_code"].nunique() >= 5
    assert df["month"].nunique() >= 12
    assert (df["sales_amount_jpy"] >= 0).all()
    assert df["is_missing"].dtype == bool


def test_sample_dataset_pipeline_roundtrip():
    df = load_sample_dataset()
    filled = fill_missing_months(df, policy="zero_fill")
    year_df = compute_year_rolling(filled, window=12, policy="zero_fill")
    year_df = compute_slopes(year_df, last_n=12)

    assert not year_df.empty
    assert year_df["year_sum"].notna().any()
    assert not year_df["product_code"].isna().any()
    # Ensure chronological order per product
    grouped = year_df.groupby("product_code")
    for _, group in grouped:
        months = pd.to_datetime(group["month"]).sort_values()
        assert months.is_monotonic_increasing


def test_fill_missing_months_forward_fill():
    df = pd.DataFrame(
        {
            "product_code": ["P1", "P1", "P1"],
            "product_name": ["テスト商品"] * 3,
            "month": ["2024-01", "2024-03", "2024-04"],
            "sales_amount_jpy": [100.0, 160.0, 200.0],
            "is_missing": [False, False, False],
        }
    )

    filled = fill_missing_months(df, policy="forward_fill")
    feb = filled[(filled["product_code"] == "P1") & (filled["month"] == "2024-02")]

    assert not feb.empty
    assert bool(feb["is_missing"].iloc[0])
    assert feb["sales_amount_jpy"].iloc[0] == pytest.approx(100.0)


def test_fill_missing_months_linear_interp():
    df = pd.DataFrame(
        {
            "product_code": ["P1", "P1", "P1"],
            "product_name": ["テスト商品"] * 3,
            "month": ["2024-01", "2024-03", "2024-04"],
            "sales_amount_jpy": [100.0, 160.0, 220.0],
            "is_missing": [False, False, False],
        }
    )

    filled = fill_missing_months(df, policy="linear_interp")
    feb = filled[(filled["product_code"] == "P1") & (filled["month"] == "2024-02")]

    assert not feb.empty
    assert bool(feb["is_missing"].iloc[0])
    assert feb["sales_amount_jpy"].iloc[0] == pytest.approx(130.0)
