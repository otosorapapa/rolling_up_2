import pandas as pd
import pytest

from services import sku_correlation_pivot, pairwise_correlation


def _sample_year_df():
    data = [
        ("A", "Alpha", "2023-01", 10.0),
        ("A", "Alpha", "2023-02", 20.0),
        ("A", "Alpha", "2023-03", 30.0),
        ("B", "Beta", "2023-01", 30.0),
        ("B", "Beta", "2023-02", 20.0),
        ("B", "Beta", "2023-03", 10.0),
        ("C", "Gamma", "2023-01", 5.0),
        ("C", "Gamma", "2023-02", 15.0),
        ("C", "Gamma", "2023-03", 25.0),
    ]
    return pd.DataFrame(
        data,
        columns=["product_code", "product_name", "month", "year_sum"],
    )


def test_pairwise_correlation_basic():
    df_year = _sample_year_df()
    pivot = sku_correlation_pivot(df_year)
    result = pairwise_correlation(pivot, method="pearson")
    pairs = {row.pair: row for row in result.itertuples()}

    assert pytest.approx(pairs["A×B"].r, abs=1e-6) == -1.0
    assert pytest.approx(pairs["A×B"].n) == 3
    assert pytest.approx(pairs["A×C"].r, abs=1e-6) == 1.0
    assert pytest.approx(pairs["B×C"].r, abs=1e-6) == -1.0
    assert pairs["B×C"].n == 3


def test_pairwise_correlation_respects_period_filter():
    df_year = _sample_year_df()
    pivot = sku_correlation_pivot(df_year, start="2023-02", end="2023-03")
    result = pairwise_correlation(pivot, method="pearson")
    pairs = {row.pair: row for row in result.itertuples()}

    # 2ヶ月のみを対象とするためサンプル数は2
    assert pairs["A×B"].n == 2
    assert pytest.approx(pairs["A×B"].r, abs=1e-6) == -1.0


def test_pairwise_correlation_skips_constant_series():
    df_year = _sample_year_df()
    # SKU C は一定値のため、他SKUとの相関結果から除外される
    df_year.loc[df_year["product_code"] == "C", "year_sum"] = 5.0
    pivot = sku_correlation_pivot(df_year)
    filtered = pairwise_correlation(pivot[["A", "C"]], method="pearson")
    assert filtered.empty
