import pandas as pd

from services import detect_data_quality_issues, parse_uploaded_table


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


def test_detect_quality_issues_flags_missing_and_negative():
    df = pd.DataFrame(
        {
            "Month": ["2024-01", None, "2024/03/01"],
            "Channel": ["EC", "", "店舗"],
            "Item": ["Alpha", "Beta", ""],
            "Revenue": [1000, "abc", -200],
        }
    )

    mapping = {
        "month": "Month",
        "channel": "Channel",
        "product_name": "Item",
        "sales": "Revenue",
        "product_code": None,
    }

    issues = detect_data_quality_issues(df, mapping)

    assert len(issues) == 5

    missing_month_issue = next(
        issue for issue in issues if issue.issue_type == "missing_month"
    )
    assert missing_month_issue.row_number == 3

    non_numeric_issue = next(
        issue for issue in issues if issue.issue_type == "non_numeric_sales"
    )
    assert "abc" in non_numeric_issue.message
    assert non_numeric_issue.row_number == 3

    negative_issue = next(
        issue for issue in issues if issue.issue_type == "negative_sales"
    )
    assert negative_issue.row_number == 4
    assert negative_issue.suggested_value == 200


def test_detect_quality_issues_suggests_clean_numeric_and_invalid_month():
    df = pd.DataFrame(
        {
            "Month": ["2024-13", "2024-02"],
            "Channel": ["EC", "店舗"],
            "Item": ["Alpha", "Beta"],
            "Revenue": ["1,234", 5000],
        }
    )

    mapping = {
        "month": "Month",
        "channel": "Channel",
        "product_name": "Item",
        "sales": "Revenue",
        "product_code": None,
    }

    issues = detect_data_quality_issues(df, mapping)

    assert len(issues) == 2

    invalid_month_issue = next(
        issue for issue in issues if issue.issue_type == "invalid_month"
    )
    assert invalid_month_issue.row_number == 2

    numeric_issue = next(
        issue for issue in issues if issue.issue_type == "non_numeric_sales"
    )
    assert numeric_issue.suggested_value == 1234
    assert "カンマ" in numeric_issue.suggestion


def test_detect_quality_issues_returns_empty_when_mapping_incomplete():
    df = pd.DataFrame(
        {
            "Month": ["2024-01"],
            "Item": ["Alpha"],
            "Revenue": [1000],
        }
    )

    mapping = {
        "month": "Month",
        "channel": None,
        "product_name": "Item",
        "sales": None,
        "product_code": None,
    }

    issues = detect_data_quality_issues(df, mapping)

    assert issues == []
