import pandas as pd

from services import parse_uploaded_table


def test_parse_with_mapping_basic():
    df = pd.DataFrame(
        {
            "Month": ["2024-01", "2024-01", "2024-02"],
            "Channel": ["EC", "店舗", "EC"],
            "Item": ["Alpha", "Alpha", "Alpha"],
            "Revenue": [1000, 500, 1100],
        }
    )

    mapping = {
        "month": "Month",
        "channel": "Channel",
        "product_name": "Item",
        "sales": "Revenue",
        "product_code": None,
    }

    result = parse_uploaded_table(df, column_mapping=mapping)

    assert set(result.columns) == {
        "product_code",
        "product_name",
        "month",
        "sales_amount_jpy",
        "is_missing",
    }
    assert len(result) == 3

    jan_ec = result[(result["product_name"] == "Alpha｜EC") & (result["month"] == "2024-01")]
    assert not jan_ec.empty
    assert jan_ec["sales_amount_jpy"].iloc[0] == 1000
    assert not bool(jan_ec["is_missing"].iloc[0])

    jan_store = result[(result["product_name"] == "Alpha｜店舗") & (result["month"] == "2024-01")]
    assert not jan_store.empty
    assert jan_store["sales_amount_jpy"].iloc[0] == 500

    feb_ec = result[(result["product_name"] == "Alpha｜EC") & (result["month"] == "2024-02")]
    assert not feb_ec.empty
    assert feb_ec["sales_amount_jpy"].iloc[0] == 1100


def test_parse_with_mapping_and_product_code_deduplication():
    df = pd.DataFrame(
        {
            "年月": ["2024-01", "2024-01"],
            "チャネル": ["EC", "店舗"],
            "商品名": ["商品A", "商品A"],
            "売上額": [150, 200],
            "商品コード": ["SKU1", "SKU1"],
        }
    )

    mapping = {
        "month": "年月",
        "channel": "チャネル",
        "product_name": "商品名",
        "sales": "売上額",
        "product_code": "商品コード",
    }

    result = parse_uploaded_table(df, column_mapping=mapping)

    assert sorted(result["product_code"].unique()) == ["SKU1｜EC", "SKU1｜店舗"]
    assert set(result["product_name"].unique()) == {"商品A｜EC", "商品A｜店舗"}
    assert result["sales_amount_jpy"].sum() == 350


def test_parse_with_mapping_marks_missing_values():
    df = pd.DataFrame(
        {
            "月": ["2024-01", "2024-01", "2024-02"],
            "チャネル": ["EC", "EC", "EC"],
            "品名": ["Beta", "Beta", "Beta"],
            "金額": [None, "1234", None],
        }
    )

    mapping = {
        "month": "月",
        "channel": "チャネル",
        "product_name": "品名",
        "sales": "金額",
        "product_code": None,
    }

    result = parse_uploaded_table(df, column_mapping=mapping)

    jan_row = result[(result["product_name"] == "Beta｜EC") & (result["month"] == "2024-01")]
    assert not jan_row.empty
    assert jan_row["sales_amount_jpy"].iloc[0] == 1234.0
    assert bool(jan_row["is_missing"].iloc[0])

    feb_row = result[(result["product_name"] == "Beta｜EC") & (result["month"] == "2024-02")]
    assert not feb_row.empty
    assert pd.isna(feb_row["sales_amount_jpy"].iloc[0])
    assert bool(feb_row["is_missing"].iloc[0])
