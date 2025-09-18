from pathlib import Path
from typing import List, Optional
import html

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ai_features import answer_question
from services import (
    fill_missing_months,
    compute_year_rolling,
    compute_slopes,
    aggregate_overview,
    compute_hhi,
)

APP_TITLE = "売上年計ダッシュボード"
COLOR_PALETTE = ["#1f5aa6", "#0f996e", "#f97316", "#6366f1", "#0891b2"]
px.defaults.color_discrete_sequence = COLOR_PALETTE
px.defaults.template = "plotly_white"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "locale": "ja",
}

UNIT_MAP = {"円": 1, "千円": 1_000, "百万円": 1_000_000}
DEFAULT_SETTINGS = {
    "window": 12,
    "last_n": 12,
    "missing_policy": "zero_fill",
    "currency_unit": "円",
}
SAMPLE_DATA_PATH = Path("data/sample.csv")

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;600;700&display=swap');
    :root {
        --bg-color: #f5f7fb;
        --panel-color: #ffffff;
        --accent-color: #1f5aa6;
        --accent-strong: #0f3c82;
        --success-color: #0f996e;
        --danger-color: #e11d48;
        --muted-color: #475569;
        --text-color: #0f172a;
    }
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Noto Sans JP', sans-serif;
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102347 0%, #1f4f8d 100%);
        color: #fff;
    }
    [data-testid="stSidebar"] * {
        color: #fff !important;
        font-family: 'Noto Sans JP', sans-serif;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .sidebar-summary {
        border-radius: 16px;
        background: rgba(255,255,255,0.14);
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .sidebar-summary .summary-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.9;
    }
    .sidebar-summary .summary-total {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0.2rem 0 0.6rem;
    }
    .sidebar-summary .summary-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        opacity: 0.95;
    }
    .hero {
        background: linear-gradient(135deg, #1f4f8d 0%, #2f89d6 100%);
        color: #fff;
        padding: 2.4rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 24px 48px rgba(15, 23, 42, 0.25);
    }
    .hero::after {
        content: '';
        position: absolute;
        width: 280px;
        height: 280px;
        background: rgba(255,255,255,0.18);
        border-radius: 50%;
        right: -120px;
        bottom: -120px;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
        color: #fff;
        font-weight: 700;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.92;
        max-width: 540px;
    }
    .hero-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 0.6rem;
    }
    .kpi-card {
        background: var(--panel-color);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(31,79,141,0.12);
        box-shadow: 0 16px 32px rgba(15,23,42,0.08);
        width: 100%;
        min-height: 150px;
    }
    .kpi-card .kpi-title {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted-color);
        font-weight: 600;
    }
    .kpi-card .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-top: 0.4rem;
        word-break: break-all;
    }
    .kpi-card.positive .kpi-value {
        color: var(--success-color);
    }
    .kpi-card.negative .kpi-value {
        color: var(--danger-color);
    }
    .kpi-caption {
        margin-top: 0.6rem;
        font-size: 0.85rem;
        color: var(--muted-color);
    }
    .chat-entry {
        margin-bottom: 1.1rem;
    }
    .chat-question, .chat-answer {
        background: var(--panel-color);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(31,79,141,0.16);
        box-shadow: 0 12px 24px rgba(15,23,42,0.06);
    }
    .chat-answer {
        margin-top: 0.4rem;
        border-left: 4px solid var(--accent-color);
    }
    .chat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted-color);
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .chat-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: var(--text-color);
    }
    .chat-meta {
        margin-top: 0.6rem;
        font-size: 0.75rem;
        color: var(--muted-color);
    }
    .chat-context {
        margin-top: 0.6rem;
        font-size: 0.8rem;
        color: var(--muted-color);
    }
    .step-card {
        background: var(--panel-color);
        border-radius: 18px;
        padding: 1.4rem;
        border: 1px solid rgba(31,79,141,0.10);
        box-shadow: 0 12px 24px rgba(15,23,42,0.05);
    }
    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--accent-color);
        color: #fff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-bottom: 0.8rem;
        font-size: 1rem;
    }
    .stDownloadButton button, .stButton>button {
        border-radius: 999px;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
        background: var(--accent-color);
        color: #fff;
        border: none;
        box-shadow: 0 12px 20px rgba(31,79,141,0.25);
    }
    .stDownloadButton button:hover, .stButton>button:hover {
        background: var(--accent-strong);
    }
    .stMetric {
        background: var(--panel-color);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(31,79,141,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    if "settings" not in st.session_state:
        st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    if "data_raw" not in st.session_state:
        st.session_state["data_raw"] = None
    if "data_monthly" not in st.session_state:
        st.session_state["data_monthly"] = None
    if "data_year" not in st.session_state:
        st.session_state["data_year"] = None
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []
    if "copilot_prompt" not in st.session_state:
        st.session_state["copilot_prompt"] = "前年同月比が高いSKUや、下落しているSKUを教えて"
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None


def format_int(value: Optional[float]) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def format_currency(value: Optional[float], unit: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    scale = UNIT_MAP.get(unit, 1)
    scaled = float(value) / scale
    if abs(scaled) >= 100:
        return f"{scaled:,.0f} {unit}"
    return f"{scaled:,.1f} {unit}"


def format_percentage(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%"


def find_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    for col in columns:
        col_str = str(col).strip()
        lower = col_str.lower()
        for keyword in keywords:
            if keyword in lower or keyword in col_str:
                return col
    return None


def normalize_month_values(series: pd.Series) -> pd.Series:
    attempts = [
        pd.to_datetime(series, errors="coerce"),
        pd.to_datetime(series.astype(str).str.replace("/", "-").str.replace(".", "-"), errors="coerce"),
        pd.to_datetime(
            series.astype(str).str.replace("/", "-").str.replace(".", "-") + "-01",
            errors="coerce",
        ),
    ]
    for dt in attempts:
        if dt.notna().any():
            return dt.dt.to_period("M").astype(str)
    raise ValueError("月度を日付形式に変換できませんでした。")


def load_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    raise ValueError("CSVまたはExcelファイルを指定してください。")


def prepare_long_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("データが空です。")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    columns = df.columns.tolist()

    date_col = find_column(columns, ["date", "日付", "年月", "month", "売上日", "年月日"])
    amount_col = find_column(columns, ["売上", "金額", "amount", "sales", "revenue"])
    code_col = find_column(columns, ["sku", "code", "商品コード", "品番", "product code"])
    name_col = find_column(columns, ["name", "商品名", "品名", "product name", "item"])

    if date_col is None:
        raise ValueError("月度または日付の列が見つかりませんでした。")
    if amount_col is None:
        raise ValueError("売上金額の列が見つかりませんでした。")
    if code_col is None and name_col is None:
        raise ValueError("SKUまたは商品名の列が見つかりませんでした。")

    selected_cols: List[str] = []
    for col in [code_col, name_col, date_col, amount_col]:
        if col is not None and col not in selected_cols:
            selected_cols.append(col)

    tidy = df[selected_cols].copy()
    rename_map = {}
    if code_col is not None:
        rename_map[code_col] = "product_code"
    if name_col is not None:
        rename_map[name_col] = "product_name"
    rename_map[date_col] = "date"
    rename_map[amount_col] = "sales_amount"
    tidy = tidy.rename(columns=rename_map)

    tidy["sales_amount"] = (
        tidy["sales_amount"].astype(str).str.replace(",", "").str.replace("¥", "").str.replace("円", "")
    )
    tidy["sales_amount"] = pd.to_numeric(tidy["sales_amount"], errors="coerce").fillna(0.0)

    tidy["month"] = normalize_month_values(tidy["date"])
    tidy = tidy[tidy["month"] != "NaT"].copy()

    if "product_name" in tidy.columns:
        tidy["product_name"] = tidy["product_name"].fillna("未設定").astype(str).str.strip()
    if "product_code" in tidy.columns:
        tidy["product_code"] = tidy["product_code"].fillna("").astype(str).str.strip()

    if "product_code" not in tidy.columns:
        names = tidy["product_name"].fillna("商品").astype(str)
        code_map = {name: f"SKU{idx + 1:04d}" for idx, name in enumerate(names.unique())}
        tidy["product_code"] = names.map(code_map)
    else:
        missing = tidy["product_code"].eq("")
        if missing.any():
            names = tidy.loc[missing, "product_name"].fillna("商品").astype(str)
            code_map = {name: f"SKU{idx + 1:04d}" for idx, name in enumerate(names.unique())}
            tidy.loc[missing, "product_code"] = names.map(code_map)

    if "product_name" not in tidy.columns:
        tidy["product_name"] = tidy["product_code"]
    else:
        tidy["product_name"] = tidy["product_name"].replace("", np.nan).fillna(tidy["product_code"])

    tidy = tidy.drop(columns=["date"])

    aggregated = (
        tidy.groupby(["product_code", "product_name", "month"], as_index=False)["sales_amount"]
        .sum()
        .rename(columns={"sales_amount": "sales_amount_jpy"})
    )
    aggregated["is_missing"] = False
    aggregated = aggregated.sort_values(["product_code", "month"], ignore_index=True)
    return aggregated


def get_month_options(year_df: Optional[pd.DataFrame]) -> List[str]:
    if year_df is None or year_df.empty or "month" not in year_df.columns:
        return []
    months = (
        year_df["month"].dropna().unique().tolist()
    )
    months = sorted(months)
    return months


def render_kpi_card(title: str, value: str, caption: Optional[str] = None, *, positive: Optional[bool] = None) -> None:
    classes = ["kpi-card"]
    if positive is True:
        classes.append("positive")
    elif positive is False:
        classes.append("negative")
    caption_html = f"<div class='kpi-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class='{ ' '.join(classes) }'>
            <div class='kpi-title'>{html.escape(title)}</div>
            <div class='kpi-value'>{html.escape(value)}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_summary() -> Optional[str]:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.sidebar.caption("データを読み込むと最新サマリーが表示されます。")
        return None

    months = get_month_options(year_df)
    if not months:
        st.sidebar.caption("月次データが存在しません。")
        return None

    latest_month = months[-1]
    summary = aggregate_overview(year_df, latest_month)
    unit = st.session_state["settings"].get("currency_unit", "円")

    total_text = format_currency(summary.get("total_year_sum"), unit)
    yoy_text = format_percentage(summary.get("yoy"))
    delta_text = format_currency(summary.get("delta"), unit)

    st.sidebar.markdown(
        f"""
        <div class="sidebar-summary">
            <div class="summary-title">最新 {latest_month}</div>
            <div class="summary-total">{total_text}</div>
            <div class="summary-row"><span>前年同月比</span><strong>{yoy_text}</strong></div>
            <div class="summary-row"><span>前月差</span><strong>{delta_text}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return latest_month


def escape_html(text: str) -> str:
    return html.escape(text, quote=False).replace("\n", "<br>")



def build_copilot_context(end_month: Optional[str] = None, top_n: int = 5) -> str:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        return "データが取り込まれていません。"

    months = get_month_options(year_df)
    if not months:
        return "月度情報が存在しません。"

    target_month = end_month or months[-1]
    snap = year_df[year_df["month"] == target_month].dropna(subset=["year_sum"]).copy()
    if snap.empty:
        return f"{target_month}のデータがありません。"

    summary = aggregate_overview(year_df, target_month)
    hhi_value = compute_hhi(year_df, target_month)
    unit = st.session_state["settings"].get("currency_unit", "円")

    lines = [
        f"対象月: {target_month}",
        f"年計総額: {format_currency(summary.get('total_year_sum'), unit)}",
        f"前年同月比: {format_percentage(summary.get('yoy'))}",
        f"前月差: {format_currency(summary.get('delta'), unit)}",
        f"SKU数: {snap['product_code'].nunique()}",
    ]
    if hhi_value is not None and not pd.isna(hhi_value):
        lines.append(f"HHI: {hhi_value:.3f}")

    growth_df = snap.dropna(subset=["yoy"]).sort_values("yoy", ascending=False).head(top_n)
    decline_df = snap.dropna(subset=["yoy"]).sort_values("yoy", ascending=True).head(top_n)

    if not growth_df.empty:
        lines.append("前年同月比が高いSKU:")
        for _, row in growth_df.iterrows():
            lines.append(
                f"- {row['product_name']} ({row['product_code']}): {format_percentage(row['yoy'])}"
            )
    if not decline_df.empty:
        lines.append("前年同月比が低いSKU:")
        for _, row in decline_df.iterrows():
            lines.append(
                f"- {row['product_name']} ({row['product_code']}): {format_percentage(row['yoy'])}"
            )

    return "\n".join(lines)


def render_home() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">Sales Insight</div>
            <h1>{APP_TITLE}</h1>
            <p>月次売上データをアップロードすると、前年同月比やトップSKUを即座に可視化し、次のアクションを迷わず決められます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 使い方の流れ")
    step_cols = st.columns(3)
    steps = [
        ("データ取込", "CSVまたはExcelで売上データをアップロードします。日付・SKU・金額の3列があればOKです。"),
        ("ダッシュボード", "前年同月比やトップSKU、売上推移をひと目で確認できます。"),
        ("AIコパイロット", "気になることを日本語で質問すると、AIがデータを基に回答します。"),
    ]
    for idx, (col, (title, description)) in enumerate(zip(step_cols, steps), start=1):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{idx}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 今月の状況")
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.info("まずは左のメニューからデータをアップロードしてください。サンプルデータもご利用いただけます。")
    else:
        latest_month = get_month_options(year_df)[-1]
        summary = aggregate_overview(year_df, latest_month)
        unit = st.session_state["settings"].get("currency_unit", "円")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_kpi_card("年計総額", format_currency(summary.get("total_year_sum"), unit), caption=f"{latest_month} 時点")
        with col2:
            yoy_val = summary.get("yoy")
            render_kpi_card(
                "前年同月比",
                format_percentage(yoy_val),
                caption="前年同月比成長率",
                positive=None if yoy_val is None else yoy_val >= 0,
            )
        with col3:
            delta_val = summary.get("delta")
            render_kpi_card(
                "前月差",
                format_currency(delta_val, unit),
                caption="前月比の増減",
                positive=None if delta_val is None else delta_val >= 0,
            )


def render_data_import(uploaded_file) -> None:
    st.title("データ取込")
    st.write("売上データをアップロードすると、年計ベースのダッシュボードを自動生成します。")

    if SAMPLE_DATA_PATH.exists():
        st.download_button(
            "サンプルデータをダウンロード",
            data=SAMPLE_DATA_PATH.read_bytes(),
            file_name="sample.csv",
            mime="text/csv",
            help="フォーマットの参考にご利用ください。",
        )

    st.markdown(
        """
        #### アップロード前のチェックポイント
        - 日付（または月度）、SKU（商品コード）、売上金額の列をご用意ください。
        - 日付は `yyyy-mm-dd` 形式、もしくは `yyyy-mm` の月度形式で入力できます。
        - 売上金額は数値またはカンマ区切りの文字列でも読み込めます。
        """
    )

    if uploaded_file is None:
        st.info("左のサイドバーからファイルを選択すると、ここに結果が表示されます。")
        return

    try:
        with st.spinner("データを読み込んでいます…"):
            raw_df = load_dataframe(uploaded_file)
            long_df = prepare_long_df(raw_df)
            settings = st.session_state["settings"]
            monthly_df = fill_missing_months(long_df, policy=settings.get("missing_policy", "zero_fill"))
            year_df = compute_year_rolling(
                monthly_df,
                window=settings.get("window", 12),
                policy=settings.get("missing_policy", "zero_fill"),
            )
            year_df = compute_slopes(year_df, last_n=settings.get("last_n", 12))

            st.session_state["data_raw"] = raw_df
            st.session_state["data_monthly"] = monthly_df
            st.session_state["data_year"] = year_df
            st.session_state["uploaded_file_name"] = uploaded_file.name

        st.success("データを読み込みました")
    except ValueError as exc:
        st.error(f"入力データを解釈できませんでした: {exc}")
        return
    except Exception as exc:  # pragma: no cover - unexpected issues should still surface
        st.exception(exc)
        return

    month_span = (
        st.session_state["data_monthly"]["month"].min(),
        st.session_state["data_monthly"]["month"].max(),
    )
    sku_count = st.session_state["data_monthly"]["product_code"].nunique()
    record_count = len(st.session_state["data_monthly"])

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi_card("期間", f"{month_span[0]} 〜 {month_span[1]}", caption="対象となる月度範囲")
    with c2:
        render_kpi_card("SKU数", format_int(sku_count), caption="ユニークな商品数")
    with c3:
        render_kpi_card("レコード数", format_int(record_count), caption="集計後の行数")

    st.subheader("プレビュー")
    st.dataframe(
        st.session_state["data_monthly"].head(100),
        use_container_width=True,
    )
    st.caption("先頭100行を表示しています。年計指標はダッシュボードからご確認ください。")


def render_dashboard() -> None:
    st.title("ダッシュボード")
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.info("左のデータ取込からデータをアップロードしてください。")
        return

    months = get_month_options(year_df)
    if not months:
        st.info("月度データが見つかりませんでした。")
        return

    if "dashboard_month" not in st.session_state or st.session_state["dashboard_month"] not in months:
        st.session_state["dashboard_month"] = months[-1]

    selected_month = st.selectbox("表示月", months, key="dashboard_month")
    summary = aggregate_overview(year_df, selected_month)
    hhi_value = compute_hhi(year_df, selected_month)
    unit = st.session_state["settings"].get("currency_unit", "円")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "年計総額",
            format_currency(summary.get("total_year_sum"), unit),
            caption=f"{selected_month} 時点",
        )
    with col2:
        yoy_val = summary.get("yoy")
        render_kpi_card(
            "前年同月比",
            format_percentage(yoy_val),
            caption="前年同月比成長率",
            positive=None if yoy_val is None else yoy_val >= 0,
        )
    with col3:
        delta_val = summary.get("delta")
        render_kpi_card(
            "前月差",
            format_currency(delta_val, unit),
            caption="直近月との増減",
            positive=None if delta_val is None else delta_val >= 0,
        )
    with col4:
        hhi_text = "—" if hhi_value is None or pd.isna(hhi_value) else f"{hhi_value:.3f}"
        render_kpi_card("HHI", hhi_text, caption="集中度 (数値が小さいほど分散)")

    totals = year_df.groupby("month", as_index=False)["year_sum"].sum().sort_values("month")
    totals["display_value"] = totals["year_sum"] / UNIT_MAP.get(unit, 1)

    yoy_rates: List[float] = []
    year_values = totals["year_sum"].tolist()
    for idx, value in enumerate(year_values):
        prev_idx = idx - 12
        if prev_idx >= 0 and not pd.isna(year_values[prev_idx]) and year_values[prev_idx] != 0:
            yoy_rates.append((value - year_values[prev_idx]) / year_values[prev_idx])
        else:
            yoy_rates.append(np.nan)
    totals["yoy_rate"] = yoy_rates

    snap = year_df[year_df["month"] == selected_month].dropna(subset=["year_sum"]).copy()
    snap["display_value"] = snap["year_sum"] / UNIT_MAP.get(unit, 1)

    chart_col1, chart_col2 = st.columns([1.6, 1.4])
    with chart_col1:
        st.markdown("#### 売上推移")
        trend_fig = px.line(
            totals,
            x="month",
            y="display_value",
            markers=True,
            title="",
        )
        trend_fig.update_yaxes(title=f"年計売上（{unit}）", zeroline=False, gridcolor="#e2e8f0")
        trend_fig.update_xaxes(title="月", showgrid=False)
        trend_fig.update_layout(
            height=360,
            margin=dict(l=16, r=16, t=32, b=16),
            font=dict(family="Noto Sans JP", color="#0f172a"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(trend_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with chart_col2:
        st.markdown("#### トップ5 SKU")
        top5 = snap.sort_values("year_sum", ascending=False).head(5)
        if top5.empty:
            st.info("対象月のデータが不足しています。")
        else:
            top5_fig = px.bar(
                top5.sort_values("year_sum"),
                x="display_value",
                y="product_name",
                orientation="h",
                color_discrete_sequence=[COLOR_PALETTE[0]],
                text=top5["display_value"].map(lambda v: f"{v:,.1f}"),
            )
            top5_fig.update_yaxes(title="")
            top5_fig.update_xaxes(title=f"売上金額（{unit}）", gridcolor="#e2e8f0")
            top5_fig.update_layout(
                height=360,
                margin=dict(l=16, r=16, t=32, b=16),
                font=dict(family="Noto Sans JP", color="#0f172a"),
            )
            st.plotly_chart(top5_fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("#### YoY分析")
    lower_col1, lower_col2 = st.columns([1.3, 1.7])
    with lower_col1:
        yoy_df = totals.dropna(subset=["yoy_rate"])
        if yoy_df.empty:
            st.info("前年同月比を計算できる期間が不足しています。")
        else:
            yoy_fig = px.line(
                yoy_df,
                x="month",
                y="yoy_rate",
                markers=True,
                color_discrete_sequence=[COLOR_PALETTE[1]],
            )
            yoy_fig.update_yaxes(title="前年同月比", tickformat=".1%", gridcolor="#e2e8f0")
            yoy_fig.update_xaxes(title="月", showgrid=False)
            yoy_fig.update_layout(
                height=300,
                margin=dict(l=16, r=16, t=32, b=16),
                font=dict(family="Noto Sans JP", color="#0f172a"),
            )
            st.plotly_chart(yoy_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with lower_col2:
        growth = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(5)
        )
        decline = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=True)
            .head(5)
        )
        subcols = st.columns(2)
        with subcols[0]:
            st.markdown("##### YoY 上位")
            if growth.empty:
                st.caption("対象データなし")
            else:
                display_df = growth[["product_name", "product_code", "yoy"]].copy()
                display_df["前年同月比"] = display_df["yoy"].map(format_percentage)
                st.dataframe(
                    display_df[["product_name", "product_code", "前年同月比"]],
                    hide_index=True,
                    use_container_width=True,
                )
        with subcols[1]:
            st.markdown("##### YoY 下位")
            if decline.empty:
                st.caption("対象データなし")
            else:
                display_df = decline[["product_name", "product_code", "yoy"]].copy()
                display_df["前年同月比"] = display_df["yoy"].map(format_percentage)
                st.dataframe(
                    display_df[["product_name", "product_code", "前年同月比"]],
                    hide_index=True,
                    use_container_width=True,
                )


def render_ai_copilot(latest_month: Optional[str]) -> None:
    st.title("AIコパイロット")
    st.write("データを基にAIが自然言語でサポートします。気になる点を質問してみましょう。")

    year_df = st.session_state.get("data_year")
    months = get_month_options(year_df)
    if months:
        if "copilot_month" not in st.session_state or st.session_state["copilot_month"] not in months:
            st.session_state["copilot_month"] = latest_month or months[-1]
    else:
        st.session_state["copilot_month"] = None

    with st.expander("AIへの質問を入力", expanded=True):
        target_month = None
        if months:
            target_month = st.selectbox("分析対象月", months, key="copilot_month")
        question = st.text_area("質問", key="copilot_prompt", height=120)
        col_send, col_clear = st.columns([1, 0.8])
        send = col_send.button("送信", type="primary")
        clear = col_clear.button("履歴をクリア")

    if clear:
        st.session_state["copilot_history"] = []
        st.info("会話履歴をクリアしました。")

    if send:
        q = question.strip()
        if not q:
            st.warning("質問を入力してください。")
        else:
            context = build_copilot_context(target_month or latest_month)
            with st.spinner("AIが分析しています..."):
                answer = answer_question(q, context)
            st.session_state["copilot_history"].append(
                {
                    "question": q,
                    "answer": answer,
                    "month": target_month or latest_month,
                    "context": context,
                }
            )
            st.success("AIからの回答を表示しました。")

    history = st.session_state.get("copilot_history", [])
    if not history:
        st.info("質問を送信すると、ここに会話履歴が表示されます。")
        return

    st.markdown("### 会話履歴")
    for entry in reversed(history):
        question_html = escape_html(entry.get("question", ""))
        answer_html = escape_html(entry.get("answer", ""))
        context_html = escape_html(entry.get("context", ""))
        st.markdown(
            f"""
            <div class="chat-entry">
                <div class="chat-question">
                    <div class="chat-label">あなた</div>
                    <div class="chat-text">{question_html}</div>
                </div>
                <div class="chat-answer">
                    <div class="chat-label">AI</div>
                    <div class="chat-text">{answer_html}</div>
                    <div class="chat-meta">対象月: {entry.get('month') or '-'} </div>
                    <div class="chat-context">コンテキスト:<br>{context_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_settings() -> None:
    st.title("設定")
    st.write("データ処理と表示の初期設定を変更できます。設定を保存すると最新データに再計算が適用されます。")

    settings = st.session_state.get("settings", {}).copy()

    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider("年計の計算期間（月）", 3, 24, value=settings.get("window", 12))
            last_n = st.slider("傾向分析に使う月数", 3, 24, value=settings.get("last_n", 12))
        with col2:
            missing_policy = st.selectbox(
                "欠測値の扱い",
                options=["zero_fill", "mark_missing"],
                index=["zero_fill", "mark_missing"].index(settings.get("missing_policy", "zero_fill")),
                format_func=lambda x: "欠測を0で埋める" if x == "zero_fill" else "欠測がある期間は除外",
            )
            currency_unit = st.selectbox(
                "表示単位",
                options=list(UNIT_MAP.keys()),
                index=list(UNIT_MAP.keys()).index(settings.get("currency_unit", "円")),
            )
        submitted = st.form_submit_button("保存")

    if not submitted:
        return

    st.session_state["settings"] = {
        "window": window,
        "last_n": last_n,
        "missing_policy": missing_policy,
        "currency_unit": currency_unit,
    }

    if st.session_state.get("data_monthly") is not None:
        with st.spinner("設定を反映しています…"):
            year_df = compute_year_rolling(
                st.session_state["data_monthly"],
                window=window,
                policy=missing_policy,
            )
            year_df = compute_slopes(year_df, last_n=last_n)
            st.session_state["data_year"] = year_df
    st.success("設定を保存しました。")



def main() -> None:
    init_session_state()

    st.sidebar.markdown(
        f"<div class='sidebar-title'>{APP_TITLE}</div>",
        unsafe_allow_html=True,
    )
    menu = st.sidebar.radio(
        "メニュー",
        ["ホーム", "データ取込", "ダッシュボード", "AIコパイロット", "設定"],
    )

    latest_month = render_sidebar_summary()
    if st.session_state.get("uploaded_file_name"):
        st.sidebar.caption(f"データソース: {st.session_state['uploaded_file_name']}")
    st.sidebar.divider()

    uploaded_file = None
    if menu == "データ取込":
        uploaded_file = st.sidebar.file_uploader(
            "売上データをアップロード",
            type=["csv", "xlsx"],
            help="売上データをアップロードしてください（例：yyyy-mm-dd, SKU, 売上金額）",
            key="sidebar_uploader",
        )

    if menu == "ホーム":
        render_home()
    elif menu == "データ取込":
        render_data_import(uploaded_file)
    elif menu == "ダッシュボード":
        render_dashboard()
    elif menu == "AIコパイロット":
        render_ai_copilot(latest_month)
    elif menu == "設定":
        render_settings()


if __name__ == "__main__":
    main()

