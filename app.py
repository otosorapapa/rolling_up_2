import html
import io
import json
import math
import re
import textwrap
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Set, Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ai_features import (
    summarize_dataframe,
    generate_comment,
    explain_analysis,
    generate_actions,
    answer_question,
    generate_anomaly_brief,
)
from core.i18n import (
    get_available_languages,
    get_current_language,
    init_language,
    language_name,
    t,
)

# McKinsey inspired pastel palette
MCKINSEY_PALETTE = [
    "#123a5f",  # deep navy
    "#2d6f8e",  # steel blue
    "#4f9ab8",  # aqua accent
    "#71b7d4",  # sky blue
    "#a9d0e7",  # frost blue
    "#dbe8f5",  # airy pastel
]
# Apply palette across figures
px.defaults.color_discrete_sequence = MCKINSEY_PALETTE

init_language()
current_language = get_current_language()

PLOTLY_CONFIG = {
    "locale": "ja",
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "autoScale2d",
        "resetViewMapbox",
        "toggleSpikelines",
        "select2d",
        "lasso2d",
        "zoom3d",
        "orbitRotation",
        "tableRotation",
    ],
    "toImageButtonOptions": {"format": "png", "filename": "年計比較"},
}
PLOTLY_CONFIG["locale"] = "ja" if current_language == "ja" else "en"


UPLOAD_FIELD_DEFS = [
    {
        "key": "month",
        "label": "年月",
        "description": "YYYY-MM 形式（例: 2024-01）",
        "required": True,
    },
    {
        "key": "channel",
        "label": "チャネル",
        "description": "販路・流通区分（例: EC, 店舗）",
        "required": True,
    },
    {
        "key": "product_name",
        "label": "商品名",
        "description": "商品を識別できる名称",
        "required": True,
    },
    {
        "key": "sales",
        "label": "売上額",
        "description": "数値（円）",
        "required": True,
    },
    {
        "key": "product_code",
        "label": "商品コード（任意）",
        "description": "SKU コードなどがあれば割当を推奨",
        "required": False,
    },
    {
        "key": "category",
        "label": "商品カテゴリー（任意）",
        "description": "品目分類・カテゴリ名など",
        "required": False,
    },
    {
        "key": "customer",
        "label": "主要顧客（任意）",
        "description": "主要顧客名または顧客セグメント",
        "required": False,
    },
    {
        "key": "region",
        "label": "地域（任意）",
        "description": "販売地域・エリア名",
        "required": False,
    },
]

UPLOAD_REQUIRED_KEYS = [field["key"] for field in UPLOAD_FIELD_DEFS if field["required"]]

UPLOAD_FIELD_KEYWORDS: Dict[str, List[str]] = {
    "month": ["年月", "yearmonth", "ym", "month", "date", "期間", "会計月"],
    "channel": ["チャネル", "channel", "販路", "流通", "店舗区分", "経路"],
    "product_name": ["商品名", "product", "item", "品名", "sku名", "名称", "name"],
    "sales": ["売上", "sales", "金額", "revenue", "amount", "売上額", "売上高", "net"],
    "product_code": ["商品コード", "sku", "code", "id", "品番", "productcode"],
    "category": ["カテゴリ", "カテゴリー", "category", "品目", "部門", "分類"],
    "customer": ["顧客", "customer", "keyaccount", "得意先", "主要顧客", "販社"],
    "region": ["地域", "リージョン", "region", "エリア", "地方", "ゾーン"],
}


def _normalize_header(label: object) -> str:
    text = str(label).strip().lower()
    for src, dst in (("（", "("), ("）", ")"), ("　", " ")):
        text = text.replace(src, dst)
    text = re.sub(r"[\s_/\\-]", "", text)
    return text


def _looks_like_month(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    patterns = [
        r"^\d{4}[-/]?(0[1-9]|1[0-2])$",
        r"^\d{4}[-/]?(0[1-9]|1[0-2])[-/]?(0[1-9]|[12]\d|3[01])$",
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def suggest_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = list(df.columns)
    normalized = {col: _normalize_header(col) for col in columns}
    suggestions: Dict[str, Optional[str]] = {field["key"]: None for field in UPLOAD_FIELD_DEFS}
    used: Set[object] = set()

    # Prefer datetime-like columns for month detection
    for col in columns:
        if col in used:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_period_dtype(series):
            suggestions["month"] = col
            used.add(col)
            break

    for field in UPLOAD_FIELD_DEFS:
        key = field["key"]
        if suggestions.get(key):
            continue
        keywords = UPLOAD_FIELD_KEYWORDS.get(key, [])
        for exact in (True, False):
            selected: Optional[object] = None
            for kw in keywords:
                kw_norm = _normalize_header(kw)
                for col in columns:
                    if col in used:
                        continue
                    norm_label = normalized[col]
                    if exact:
                        if norm_label == kw_norm:
                            selected = col
                            break
                    else:
                        if kw_norm and kw_norm in norm_label:
                            selected = col
                            break
                if selected is not None:
                    break
            if selected is not None:
                suggestions[key] = selected
                used.add(selected)
                break

    if suggestions.get("month") is None:
        for col in columns:
            if col in used:
                continue
            sample = df[col].dropna().head(6)
            if not sample.empty and sample.apply(_looks_like_month).all():
                suggestions["month"] = col
                used.add(col)
                break

    if suggestions.get("sales") is None:
        numeric_candidates = [
            col
            for col in columns
            if col not in used and pd.api.types.is_numeric_dtype(df[col])
        ]
        if numeric_candidates:
            suggestions["sales"] = numeric_candidates[0]
            used.add(numeric_candidates[0])

    if suggestions.get("product_name") is None:
        for col in columns:
            if col in used:
                continue
            suggestions["product_name"] = col
            used.add(col)
            break

    if suggestions.get("channel") is None:
        for col in columns:
            if col in used:
                continue
            suggestions["channel"] = col
            used.add(col)
            break

    return suggestions


def render_column_mapping_tool(
    columns: List[object],
    mapping: Dict[str, Optional[str]],
    *,
    key: str,
) -> Optional[Dict[str, Optional[str]]]:
    unassigned_token = f"__{key}_unassigned__"

    id_to_column: Dict[str, object] = {}
    label_by_id: Dict[str, str] = {}
    value_to_id: Dict[object, str] = {}
    for idx, column in enumerate(columns):
        column_id = f"{key}_col_{idx}"
        id_to_column[column_id] = column
        label = str(column)
        label_by_id[column_id] = label
        if column not in value_to_id:
            value_to_id[column] = column_id

    mapping_ids = {
        field["key"]: value_to_id.get(mapping.get(field["key"]))
        for field in UPLOAD_FIELD_DEFS
    }

    state_ids_key = f"{key}__selected_ids"
    if state_ids_key not in st.session_state:
        st.session_state[state_ids_key] = mapping_ids.copy()
        for field in UPLOAD_FIELD_DEFS:
            widget_key = f"{key}__select__{field['key']}"
            st.session_state[widget_key] = (
                mapping_ids[field["key"]] or unassigned_token
            )
    elif st.session_state[state_ids_key] != mapping_ids:
        st.session_state[state_ids_key] = mapping_ids.copy()
        for field in UPLOAD_FIELD_DEFS:
            widget_key = f"{key}__select__{field['key']}"
            st.session_state[widget_key] = (
                mapping_ids[field["key"]] or unassigned_token
            )

    option_ids = [unassigned_token] + list(id_to_column.keys())

    def format_option(option_id: str) -> str:
        if option_id == unassigned_token:
            return "未割当"
        return label_by_id.get(option_id, option_id)

    if label_by_id:
        st.markdown(
            "<div style='margin-bottom:0.75rem'>"
            + "".join(
                f"<span style='display:inline-block;padding:4px 10px;margin:2px;border-radius:999px;"
                "background:#eef4fb;border:1px solid #c7d7eb;font-size:12px;color:#0b2f4c'>"
                f"{html.escape(label)}</span>"
                for label in label_by_id.values()
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("アップロードされた列がありません。")

    selected_ids: Dict[str, Optional[str]] = {}
    for field in UPLOAD_FIELD_DEFS:
        widget_key = f"{key}__select__{field['key']}"
        desired_default = mapping_ids[field["key"]] or unassigned_token
        if widget_key not in st.session_state:
            st.session_state[widget_key] = desired_default
        elif st.session_state[widget_key] not in option_ids:
            st.session_state[widget_key] = unassigned_token

        label = field["label"] + ("（必須）" if field["required"] else "（任意）")
        selected = st.selectbox(
            label,
            options=option_ids,
            format_func=format_option,
            key=widget_key,
            help=field["description"],
        )
        st.caption(field["description"])
        selected_ids[field["key"]] = (
            None if selected == unassigned_token else selected
        )

    st.session_state[state_ids_key] = selected_ids.copy()

    resolved_mapping: Dict[str, Optional[str]] = {}
    for field in UPLOAD_FIELD_DEFS:
        mapped_id = selected_ids.get(field["key"])
        resolved_mapping[field["key"]] = id_to_column.get(mapped_id)

    changed = any(
        resolved_mapping.get(field["key"]) != mapping.get(field["key"])
        for field in UPLOAD_FIELD_DEFS
    )
    return resolved_mapping if changed else None


@st.cache_data(show_spinner=False)
def build_upload_template() -> bytes:
    template_df = pd.DataFrame(
        {
            "年月": [
                "2024-01",
                "2024-01",
                "2024-02",
                "2024-02",
                "2024-03",
                "2024-03",
            ],
            "チャネル": ["EC", "店舗", "EC", "店舗", "EC", "店舗"],
            "商品名": [
                "サンプル栄養ドリンクA",
                "サンプル炭酸飲料B",
                "サンプル栄養ドリンクA",
                "サンプル炭酸飲料B",
                "サンプル栄養ドリンクA",
                "サンプル炭酸飲料B",
            ],
            "売上額": [1250000, 980000, 1300000, 1010000, 1280000, 990000],
            "商品コード": [
                "TMP-A",
                "TMP-B",
                "TMP-A",
                "TMP-B",
                "TMP-A",
                "TMP-B",
            ],
        }
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        template_df.to_excel(writer, sheet_name="売上データ", index=False)
        workbook = writer.book
        sheet = writer.sheets["売上データ"]
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#dbe8f5", "border": 1, "font_color": "#0b2f4c"}
        )
        for col_idx, column_name in enumerate(template_df.columns):
            sheet.write(0, col_idx, column_name, header_format)
        sheet.freeze_panes(1, 0)
        sheet.set_column("A:A", 14)
        sheet.set_column("B:B", 12)
        sheet.set_column("C:C", 26)
        sheet.set_column("D:D", 14)
        sheet.set_column("E:E", 16)

        readme = workbook.add_worksheet("README")
        readme.set_column("A:A", 60)
        readme.write(0, 0, "RollingUp データ取込テンプレート")
        readme.write(2, 0, "必要列")
        readme.write(3, 0, "- 年月: YYYY-MM 形式（例: 2024-01）")
        readme.write(4, 0, "- チャネル: 販売チャネル名（例: EC, 店舗）")
        readme.write(5, 0, "- 商品名: 商品名称 / SKU 名")
        readme.write(6, 0, "- 売上額: 金額（円）")
        readme.write(8, 0, "任意列")
        readme.write(9, 0, "- 商品コード: SKU コード。未設定の場合は自動で付番されます。")
        readme.write(11, 0, "メモ")
        readme.write(12, 0, "- 同一商品が複数チャネルに存在する場合はチャネル別に集計されます。")
        readme.write(13, 0, "- 日次・週次データは年月で集計してからアップロードしてください。")

    buffer.seek(0)
    return buffer.getvalue()

APP_TITLE = t("header.title", language=current_language)
st.set_page_config(
    page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded"
)


@st.cache_data(ttl=600)
def _ai_sum_df(df: pd.DataFrame) -> str:
    return summarize_dataframe(df)


@st.cache_data(ttl=600)
def _ai_explain(d: dict) -> str:
    return explain_analysis(d)


@st.cache_data(ttl=600)
def _ai_comment(t: str) -> str:
    return generate_comment(t)


@st.cache_data(ttl=600)
def _ai_actions(metrics: Dict[str, float], focus: str) -> str:
    return generate_actions(metrics, focus)


@st.cache_data(ttl=600)
def _ai_answer(question: str, context: str) -> str:
    return answer_question(question, context)


@st.cache_data(ttl=600)
def _ai_anomaly_report(df: pd.DataFrame) -> str:
    return generate_anomaly_brief(df)


from services import (
    parse_uploaded_table,
    fill_missing_months,
    compute_year_rolling,
    compute_slopes,
    abc_classification,
    compute_hhi,
    build_alerts,
    aggregate_overview,
    build_indexed_series,
    latest_yearsum_snapshot,
    resolve_band,
    filter_products_by_band,
    get_yearly_series,
    top_growth_codes,
    trend_last6,
    slopes_snapshot,
    shape_flags,
    detect_linear_anomalies,
)
from sample_data import load_sample_dataset
from core.chart_card import toolbar_sku_detail, build_chart_card
from core.plot_utils import apply_elegant_theme, render_plotly_with_spinner
from core.correlation import (
    corr_table,
    fisher_ci,
    fit_line,
    maybe_log1p,
    narrate_top_insights,
    winsorize_frame,
)
from core.product_clusters import render_correlation_category_module
from core.theme import base_theme_css, elegant_theme_css

# McKinsey inspired light theme
st.markdown(base_theme_css(), unsafe_allow_html=True)

# ===== Elegant（品格）UI ON/OFF & Language Selector =====
if "elegant_on" not in st.session_state:
    st.session_state["elegant_on"] = True

with st.container():
    control_left, control_right = st.columns([3, 1])
    with control_left:
        elegant_on = st.toggle(
            t("header.elegant_toggle.label"),
            value=st.session_state.get("elegant_on", True),
            help=t("header.elegant_toggle.help"),
            key="elegant_ui_toggle",
        )
        st.session_state["elegant_on"] = elegant_on
    with control_right:
        language_codes = get_available_languages()
        if language_codes:
            current_value = st.session_state.get("language")
            if current_value not in language_codes:
                st.session_state["language"] = language_codes[0]
        else:
            language_codes = [get_current_language()]
        st.selectbox(
            t("header.language_selector.label"),
            options=language_codes,
            key="language",
            format_func=lambda code: language_name(code),
        )

elegant_on = st.session_state.get("elegant_on", True)

# ===== 品格UI CSS（配色/余白/フォント/境界の見直し） =====
if elegant_on:
    st.markdown(elegant_theme_css(), unsafe_allow_html=True)

# ---------------- Session State ----------------
if "data_monthly" not in st.session_state:
    st.session_state.data_monthly = None  # long-form DF
if "data_year" not in st.session_state:
    st.session_state.data_year = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "window": 12,
        "last_n": 12,
        "missing_policy": "zero_fill",
        "yoy_threshold": -0.10,
        "delta_threshold": -300000.0,
        "slope_threshold": -1.0,
        "currency_unit": "円",
    }
if "notes" not in st.session_state:
    st.session_state.notes = {}  # product_code -> str
if "tags" not in st.session_state:
    st.session_state.tags = {}  # product_code -> List[str]
if "saved_views" not in st.session_state:
    st.session_state.saved_views = {}  # name -> dict
if "compare_params" not in st.session_state:
    st.session_state.compare_params = {}
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None
if "copilot_answer" not in st.session_state:
    st.session_state.copilot_answer = ""
if "copilot_context" not in st.session_state:
    st.session_state.copilot_context = ""
if "copilot_focus" not in st.session_state:
    st.session_state.copilot_focus = "全体サマリー"
if "tour_active" not in st.session_state:
    st.session_state.tour_active = True
if "tour_step_index" not in st.session_state:
    st.session_state.tour_step_index = 0
if "tour_completed" not in st.session_state:
    st.session_state.tour_completed = False
if "sample_data_notice" not in st.session_state:
    st.session_state.sample_data_notice = False

# track user interactions and global filters
if "click_log" not in st.session_state:
    st.session_state.click_log = {}
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "upload_mapping" not in st.session_state:
    st.session_state.upload_mapping = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

# currency unit scaling factors
UNIT_MAP = {"円": 1, "千円": 1_000, "百万円": 1_000_000}


def log_click(name: str):
    """Increment click count for command bar actions."""
    st.session_state.click_log[name] = st.session_state.click_log.get(name, 0) + 1


def render_app_hero():
    st.markdown(
        f"""
        <div class=\"mck-hero\">
            <div class=\"mck-hero__eyebrow\">{t("header.eyebrow")}</div>
            <h1>{t("header.title")}</h1>
            <p>{t("header.description")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_current_tour_step() -> Optional[Dict[str, str]]:
    if not st.session_state.get("tour_active", True):
        return None
    if not TOUR_STEPS:
        return None
    idx = max(0, min(st.session_state.get("tour_step_index", 0), len(TOUR_STEPS) - 1))
    return TOUR_STEPS[idx]


def render_tour_banner() -> None:
    if not TOUR_STEPS:
        return

    total = len(TOUR_STEPS)
    idx = max(0, min(st.session_state.get("tour_step_index", 0), total - 1))
    st.session_state.tour_step_index = idx
    active = st.session_state.get("tour_active", True)

    banner = st.container()
    with banner:
        banner_class = "tour-banner" if active else "tour-banner tour-banner--muted"
        st.markdown(f"<div class='{banner_class}'>", unsafe_allow_html=True)
        if active:
            step = TOUR_STEPS[idx]
            section_label = step.get("section", "")
            section_index = step.get("section_index", idx + 1)
            section_total = step.get("section_total", total)
            title_text = step.get("title") or step.get("heading") or step.get("label") or ""
            description_text = step.get("description", "")
            details_text = step.get("details", "")

            if section_label:
                st.markdown(
                    f"<div class='tour-banner__section'>{html.escape(section_label)}<span>{section_index} / {section_total}</span></div>",
                    unsafe_allow_html=True,
                )

            if title_text:
                st.markdown(
                    f"<div class='tour-banner__title'>{html.escape(title_text)}</div>",
                    unsafe_allow_html=True,
                )
            if description_text:
                st.markdown(
                    f"<p class='tour-banner__desc'>{html.escape(description_text)}</p>",
                    unsafe_allow_html=True,
                )
            if details_text:
                st.markdown(
                    f"<p class='tour-banner__details'>{html.escape(details_text)}</p>",
                    unsafe_allow_html=True,
                )

            section_progress_label = (
                f"{section_label} {section_index} / {section_total}"
                if section_label
                else f"STEP {idx + 1} / {total}"
            )
            progress_percent = ((idx + 1) / total) * 100 if total else 0
            progress_html = f"""
<div class='tour-progress'>
  <div class='tour-progress__meta'>
    <span>{html.escape(section_progress_label)}</span>
    <span>STEP {idx + 1} / {total}</span>
  </div>
  <div class='tour-progress__track' role='progressbar' aria-valuemin='1' aria-valuemax='{total}' aria-valuenow='{idx + 1}'>
    <div class='tour-progress__bar' style='width: {progress_percent:.2f}%;'></div>
  </div>
</div>
"""
            st.markdown(progress_html, unsafe_allow_html=True)

            st.markdown("<div class='tour-banner__nav'>", unsafe_allow_html=True)
            prev_col, next_col, finish_col = st.columns(3)
            prev_clicked = prev_col.button(
                "前へ",
                key="tour_prev",
                use_container_width=True,
                disabled=idx == 0,
            )
            next_clicked = next_col.button(
                "次へ",
                key="tour_next",
                use_container_width=True,
                disabled=idx >= total - 1,
            )
            finish_clicked = finish_col.button(
                "終了",
                key="tour_finish",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if prev_clicked and idx > 0:
                new_idx = idx - 1
                st.session_state.tour_step_index = new_idx
                st.session_state.tour_pending_nav = TOUR_STEPS[new_idx]["nav_key"]
                st.session_state.tour_completed = False
                st.experimental_rerun()

            if next_clicked and idx < total - 1:
                new_idx = idx + 1
                st.session_state.tour_step_index = new_idx
                st.session_state.tour_pending_nav = TOUR_STEPS[new_idx]["nav_key"]
                st.session_state.tour_completed = False
                st.experimental_rerun()

            if finish_clicked:
                st.session_state.tour_active = False
                st.session_state.tour_completed = True
                st.session_state.pop("tour_pending_nav", None)
                st.experimental_rerun()
        else:
            completed = st.session_state.get("tour_completed", False)
            last_step = TOUR_STEPS[idx] if 0 <= idx < total else None
            section_label = last_step.get("section", "") if last_step else ""
            section_index = last_step.get("section_index", 0) if last_step else 0
            section_total = last_step.get("section_total", 0) if last_step else 0
            title_text = (
                last_step.get("title")
                or last_step.get("heading")
                or last_step.get("label")
                or ""
                if last_step
                else ""
            )

            if section_label:
                st.markdown(
                    f"<div class='tour-banner__section'>{html.escape(section_label)}<span>{section_index} / {section_total}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<p class='tour-banner__progress'>チュートリアルツアー</p>",
                unsafe_allow_html=True,
            )

            if completed and idx == total - 1:
                desc_text = "基礎編から応用編までのツアーを完了しました。必要なときにいつでも振り返りできます。"
            elif last_step:
                desc_text = (
                    f"前回は{section_label}の「{title_text}」まで進みました。途中から続きが再開できます。"
                )
            else:
                desc_text = "再開ボタンでいつでもハイライトを確認できます。"

            st.markdown(
                f"<p class='tour-banner__desc'>{html.escape(desc_text)}</p>",
                unsafe_allow_html=True,
            )

            if last_step:
                section_progress_label = (
                    f"{section_label} {section_index} / {section_total}"
                    if section_label
                    else f"STEP {idx + 1} / {total}"
                )
                progress_percent = ((idx + 1) / total) * 100 if total else 0
                progress_html = f"""
<div class='tour-progress'>
  <div class='tour-progress__meta'>
    <span>{html.escape(section_progress_label)}</span>
    <span>STEP {idx + 1} / {total}</span>
  </div>
  <div class='tour-progress__track' role='progressbar' aria-valuemin='1' aria-valuemax='{total}' aria-valuenow='{idx + 1}'>
    <div class='tour-progress__bar' style='width: {progress_percent:.2f}%;'></div>
  </div>
</div>
"""
                st.markdown(progress_html, unsafe_allow_html=True)

            st.markdown(
                "<div class='tour-banner__nav tour-banner__nav--resume'>",
                unsafe_allow_html=True,
            )
            resume_col, restart_col = st.columns(2)
            resume_clicked = resume_col.button(
                "再開",
                key="tour_resume",
                use_container_width=True,
            )
            restart_clicked = restart_col.button(
                "最初から",
                key="tour_restart",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if resume_clicked:
                st.session_state.tour_active = True
                st.session_state.tour_completed = False
                if last_step and last_step.get("nav_key") in NAV_KEYS:
                    st.session_state.tour_pending_nav = last_step["nav_key"]
                st.experimental_rerun()

            if restart_clicked:
                st.session_state.tour_active = True
                st.session_state.tour_completed = False
                st.session_state.tour_step_index = 0
                if TOUR_STEPS:
                    st.session_state.tour_pending_nav = TOUR_STEPS[0]["nav_key"]
                st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def apply_tour_highlight(step: Optional[Dict[str, str]]) -> None:
    payload = {
        "key": step.get("key") if step else "",
        "navKey": step.get("nav_key") if step else "",
        "label": step.get("label") if step else "",
        "heading": step.get("heading") if step else "",
    }
    script = f"""
    <script>
    const STEP = {json.dumps(payload, ensure_ascii=False)};
    const normalize = (text) => (text || '').replace(/\s+/g, ' ').trim();
    const doc = window.parent.document;
    const run = () => {{
        const root = doc.documentElement;
        if (STEP.key) {{
            root.setAttribute('data-tour-key', STEP.key);
        }} else {{
            root.removeAttribute('data-tour-key');
        }}

        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {{
            sidebar.querySelectorAll('.tour-highlight-nav').forEach((el) => el.classList.remove('tour-highlight-nav'));
            let target = null;
            if (STEP.navKey) {{
                target = sidebar.querySelector(`label[data-nav-key="${{STEP.navKey}}"]`);
            }}
            if (!target && STEP.label) {{
                const labels = Array.from(sidebar.querySelectorAll('label'));
                target = labels.find((el) => normalize(el.innerText) === normalize(STEP.label));
            }}
            if (target) {{
                target.classList.add('tour-highlight-nav');
                target.scrollIntoView({{ block: 'nearest' }});
            }}
        }}

        doc.querySelectorAll('.tour-highlight-heading').forEach((el) => el.classList.remove('tour-highlight-heading'));
        if (STEP.heading) {{
            const headings = Array.from(doc.querySelectorAll('h1, h2, h3'));
            const targetHeading = headings.find((el) => normalize(el.innerText) === normalize(STEP.heading));
            if (targetHeading) {{
                const container = targetHeading.closest('.mck-section-header') || targetHeading.parentElement;
                if (container) {{
                    container.classList.add('tour-highlight-heading');
                    container.scrollIntoView({{ block: 'start', behavior: 'smooth' }});
                }}
            }}
        }}

        const hints = Array.from(doc.querySelectorAll('div, span')).filter((el) => normalize(el.textContent).includes('→キーで次へ'));
        hints.forEach((el) => el.remove());
    }};
    setTimeout(run, 120);
    </script>
    """
    components.html(script, height=0)
def section_header(
    title: str, subtitle: Optional[str] = None, icon: Optional[str] = None
):
    icon_html = f"<span class='mck-section-icon'>{icon}</span>" if icon else ""
    subtitle_html = (
        f"<p class='mck-section-subtitle'>{subtitle}</p>" if subtitle else ""
    )
    st.markdown(
        f"""
        <div class=\"mck-section-header\">
            {icon_html}
            <div>
                <h2>{title}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clip_text(value: str, width: int = 220) -> str:
    if not value:
        return ""
    return textwrap.shorten(value, width=width, placeholder="…")


# ---------------- Helpers ----------------
def require_data():
    if st.session_state.data_year is None or st.session_state.data_monthly is None:
        st.stop()


def month_options(df: pd.DataFrame) -> List[str]:
    return sorted(df["month"].dropna().unique().tolist())


def end_month_selector(
    df: pd.DataFrame,
    key: str = "end_month",
    label: str = "終端月（年計の計算対象）",
    sidebar: bool = False,
):
    """Month selector that can be rendered either in the main area or sidebar."""

    mopts = month_options(df)
    widget = st.sidebar if sidebar else st
    if not mopts:
        widget.caption("対象となる月がありません。")
        return None
    return widget.selectbox(
        label,
        mopts,
        index=(len(mopts) - 1) if mopts else 0,
        key=key,
    )


def download_excel(df: pd.DataFrame, filename: str) -> bytes:
    import xlsxwriter  # noqa

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()


def download_pdf_overview(kpi: dict, top_df: pd.DataFrame, filename: str) -> bytes:
    # Minimal PDF using reportlab (text only)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "年計KPIサマリー")
    y -= 24
    c.setFont("Helvetica", 11)
    for k, v in kpi.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TOP10（年計）")
    y -= 18
    c.setFont("Helvetica", 10)
    cols = ["product_code", "product_name", "year_sum"]
    for _, row in top_df[cols].head(10).iterrows():
        c.drawString(
            40,
            y,
            f"{row['product_code']}  {row['product_name']}  {int(row['year_sum']):,}",
        )
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def format_percent(
    val: Optional[float], *, decimals: int = 1, signed: bool = True
) -> str:
    """Format a ratio (0-1) as a percentage string."""

    if val is None:
        return "—"
    try:
        if pd.isna(val):
            return "—"
    except TypeError:
        pass

    pct = float(val) * 100.0
    formatted = f"{pct:.{decimals}f} %"
    if signed and pct > 0:
        formatted = "+" + formatted
    return formatted


def format_amount(val: Optional[float], unit: str) -> str:
    """Format a numeric value according to currency unit."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    scale = UNIT_MAP.get(unit, 1)
    return f"{format_int(val / scale)} {unit}".strip()


def format_amount_delta(val: Optional[float], unit: str) -> str:
    """Format a numeric delta with explicit sign."""

    if val is None:
        return "—"
    try:
        if pd.isna(val):
            return "—"
    except TypeError:
        pass

    scale = UNIT_MAP.get(unit, 1)
    scaled = float(val) / scale
    sign = "+" if scaled > 0 else ""
    return f"{sign}{format_int(scaled)} {unit}".strip()


def format_int(val: float | int) -> str:
    """Format a number with commas and no decimal part."""
    try:
        return f"{int(round(val)):,}"
    except (TypeError, ValueError):
        return "0"


def infer_channel_label(name: Optional[str]) -> str:
    """Infer channel label from product name that may contain a delimiter."""

    if not isinstance(name, str):
        return "全チャネル"
    text = name.strip()
    if "｜" in text:
        candidate = text.rsplit("｜", 1)[-1].strip()
        if candidate:
            return candidate
    return "全チャネル"


def shift_year_month(month: Optional[str], offset: int) -> Optional[str]:
    """Shift a YYYY-MM string by the given number of months."""

    if month is None:
        return None
    text = str(month)
    try:
        dt = datetime.strptime(text, "%Y-%m")
    except ValueError:
        return None
    shifted = dt + relativedelta(months=offset)
    return shifted.strftime("%Y-%m")


def apply_dimensional_filters(
    df: Optional[pd.DataFrame], filters: Dict[str, List[str]]
) -> Optional[pd.DataFrame]:
    """Filter a dataframe by the provided column -> values mapping."""

    if df is None or df.empty:
        return df
    if not filters:
        return df

    mask = pd.Series(True, index=df.index)
    for column, selected in filters.items():
        if not selected:
            continue
        if column not in df.columns:
            continue
        selected_str = [str(val) for val in selected]
        mask = mask & df[column].astype(str).isin(selected_str)
    return df[mask].copy()


def compute_segment_summary(
    year_df: Optional[pd.DataFrame], group_cols: List[str], end_month: str
) -> pd.DataFrame:
    """Aggregate year_sum totals and YoY for the specified grouping columns."""

    if year_df is None or year_df.empty:
        return pd.DataFrame()
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if not group_cols:
        return pd.DataFrame()

    missing_cols = [col for col in group_cols if col not in year_df.columns]
    if missing_cols:
        return pd.DataFrame()

    snap = (
        year_df[year_df["month"] == end_month]
        .dropna(subset=["year_sum"])
        .copy()
    )
    if snap.empty:
        return pd.DataFrame()

    summary = (
        snap.groupby(group_cols, dropna=False, as_index=False)["year_sum"].sum()
        .sort_values("year_sum", ascending=False)
    )
    total = float(summary["year_sum"].sum())
    if total > 0:
        summary["share"] = summary["year_sum"] / total
    else:
        summary["share"] = np.nan

    prev_month = shift_year_month(end_month, -12)
    if prev_month and (year_df["month"] == prev_month).any():
        prev = (
            year_df[year_df["month"] == prev_month]
            .groupby(group_cols, dropna=False)["year_sum"].sum()
            .reset_index()
            .rename(columns={"year_sum": "year_sum_prev"})
        )
        summary = summary.merge(prev, on=group_cols, how="left")
        summary["growth_rate"] = np.where(
            summary["year_sum_prev"].gt(0),
            (summary["year_sum"] - summary["year_sum_prev"]) / summary["year_sum_prev"],
            np.nan,
        )
    else:
        summary["year_sum_prev"] = np.nan
        summary["growth_rate"] = np.nan

    return summary


def render_segment_summary_tab(
    summary: pd.DataFrame, column: str, label: str, unit: str
) -> None:
    """Render a table and scatter chart for a segment summary."""

    if summary is None or summary.empty:
        st.info(f"{label}のデータが不足しています。")
        return
    if column not in summary.columns:
        st.info(f"{label}情報が存在しません。")
        return

    work = summary.copy()
    work = work.rename(columns={column: label})

    scale = UNIT_MAP.get(unit, 1)
    work[f"年計({unit})"] = work["year_sum"] / scale
    work["構成比(%)"] = work["share"] * 100
    work["YoY(%)"] = work["growth_rate"] * 100

    amount_format = "%.0f" if unit == "円" else "%.1f"
    table_df = work[[label, f"年計({unit})", "構成比(%)", "YoY(%)"]]

    st.dataframe(
        table_df,
        use_container_width=True,
        column_config={
            f"年計({unit})": st.column_config.NumberColumn(format=amount_format),
            "構成比(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "YoY(%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    chart_df = work[[label, f"年計({unit})", "YoY(%)", "share", "year_sum_prev"]].dropna(
        subset=[f"年計({unit})"]
    )
    chart_df = chart_df.dropna(subset=["YoY(%)"])
    if chart_df.empty:
        st.info("YoY を算出できるデータが不足しています。")
        return

    chart_df = chart_df.rename(columns={f"年計({unit})": "年計値", "YoY(%)": "YoY値"})
    chart_df["構成比"] = chart_df["share"] * 100
    if "year_sum_prev" in chart_df.columns:
        chart_df["前年年計"] = chart_df["year_sum_prev"] / scale

    hover_config: Dict[str, str | bool] = {
        label: True,
        "年計値": ":.2f",
        "YoY値": ":.2f",
        "構成比": ":.1f%%",
    }
    if "前年年計" in chart_df.columns:
        hover_config["前年年計"] = ":.2f"

    fig = px.scatter(
        chart_df,
        x="YoY値",
        y="年計値",
        size="構成比",
        color=label,
        hover_data=hover_config,
        size_max=45,
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="#123a5f")))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="YoY(%)",
        yaxis_title=f"年計({unit})",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#888", opacity=0.6)
    fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "dark"))
    render_plotly_with_spinner(fig, config=PLOTLY_CONFIG)

def _build_kpi_card_css(theme: str) -> str:
    if theme == "dark":
        card_bg = "rgba(17, 24, 38, 0.92)"
        border = "rgba(113, 183, 212, 0.35)"
        shadow = "0 12px 24px rgba(3, 14, 26, 0.55)"
        shadow_hover = "0 18px 32px rgba(3, 14, 26, 0.65)"
        label_color = "#BBD2E8"
        value_color = "#E9F1FF"
        caption_color = "#8FA5C2"
    else:
        card_bg = "rgba(255, 255, 255, 0.95)"
        border = "rgba(18, 58, 95, 0.18)"
        shadow = "0 6px 16px rgba(18, 58, 95, 0.08)"
        shadow_hover = "0 12px 22px rgba(18, 58, 95, 0.12)"
        label_color = "#0F2C4C"
        value_color = "#123a5f"
        caption_color = "#4F627A"

    return f"""
    <style>
      .kpi-card-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin: 0.5rem 0 1.25rem;
      }}
      .kpi-card {{
        position: relative;
        padding: 0.9rem 1rem 0.85rem;
        border-radius: 16px;
        border: 1px solid {border};
        background: {card_bg};
        box-shadow: {shadow};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }}
      .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: {shadow_hover};
      }}
      .kpi-card__label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {label_color};
        display: flex;
        align-items: center;
        gap: 0.45rem;
      }}
      .kpi-card__icon {{
        font-size: 1.15rem;
      }}
      .kpi-card__value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {value_color};
        margin-top: 0.35rem;
        line-height: 1.2;
        word-break: break-word;
      }}
      .kpi-card__caption {{
        margin-top: 0.45rem;
        font-size: 0.78rem;
        color: {caption_color};
        line-height: 1.45;
        word-break: break-word;
      }}
      .kpi-card__delta {{
        margin-top: 0.2rem;
        font-size: 0.78rem;
        font-weight: 600;
      }}
      .kpi-card__delta--positive {{ color: #1f7a53; }}
      .kpi-card__delta--negative {{ color: #c84c44; }}
      .kpi-card__delta--neutral {{ color: {caption_color}; }}
      .kpi-card--positive {{ border-color: rgba(38, 137, 91, 0.45); box-shadow: 0 8px 18px rgba(38, 137, 91, 0.18); }}
      .kpi-card--negative {{ border-color: rgba(200, 76, 68, 0.35); box-shadow: 0 8px 18px rgba(200, 76, 68, 0.18); }}
    </style>
    """


def render_kpi_cards(cards: List[Dict[str, Any]]) -> None:
    if not cards:
        return

    theme = st.session_state.get("ui_theme", "dark")
    css_theme_key = "_kpi_cards_css_theme"
    if st.session_state.get(css_theme_key) != theme:
        st.session_state[css_theme_key] = theme
        st.session_state["_kpi_cards_css_injected"] = False

    if not st.session_state.get("_kpi_cards_css_injected", False):
        st.markdown(_build_kpi_card_css(theme), unsafe_allow_html=True)
        st.session_state["_kpi_cards_css_injected"] = True

    blocks = ["<div class='kpi-card-grid'>"]
    for card in cards:
        sentiment = card.get("sentiment")
        if sentiment not in {"positive", "negative", "neutral"}:
            delta_val = card.get("delta")
            sentiment = "neutral"
            if isinstance(delta_val, (int, float, np.floating)) and not pd.isna(delta_val):
                if delta_val > 0:
                    sentiment = "positive"
                elif delta_val < 0:
                    sentiment = "negative"

        icon = card.get("icon")
        icon_html = (
            f"<span class='kpi-card__icon'>{html.escape(str(icon))}</span>"
            if icon
            else ""
        )
        label = html.escape(str(card.get("label", "")))
        value = card.get("value", "—")
        value_html = html.escape(str(value))

        caption = card.get("caption")
        caption_html = ""
        if caption:
            caption_html = (
                "<div class='kpi-card__caption'>"
                + html.escape(str(caption)).replace("\n", "<br>")
                + "</div>"
            )

        delta_text = card.get("delta_text")
        delta_html = ""
        if delta_text:
            delta_class = f"kpi-card__delta--{sentiment}"
            delta_html = (
                f"<div class='kpi-card__delta {delta_class}'>"
                + html.escape(str(delta_text))
                + "</div>"
            )

        blocks.append(
            """
            <div class='kpi-card kpi-card--{sentiment}'>
              <div class='kpi-card__label'>{icon_html}{label}</div>
              <div class='kpi-card__value'>{value_html}</div>
              {delta_html}
              {caption_html}
            </div>
            """.format(
                sentiment=sentiment,
                icon_html=icon_html,
                label=label,
                value_html=value_html,
                delta_html=delta_html,
                caption_html=caption_html,
            )
        )

    blocks.append("</div>")
    st.markdown("\n".join(blocks), unsafe_allow_html=True)


def nice_slider_step(max_value: int, target_steps: int = 40) -> int:
    """Return an intuitive step size so sliders move in round increments."""
    if max_value <= 0:
        return 1

    raw_step = max_value / target_steps
    if raw_step <= 1:
        return 1

    exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = raw_step / (10 ** exponent) if raw_step > 0 else 1

    for nice in (1, 2, 5, 10):
        if base <= nice:
            step = nice * (10 ** exponent)
            return int(step) if step >= 1 else 1

    return int(10 ** (exponent + 1))


def choose_amount_slider_unit(max_amount: int) -> tuple[int, str]:
    """Choose a unit so the slider operates in easy-to-understand scales."""
    units = [
        (1, "円"),
        (1_000, "千円"),
        (10_000, "万円"),
        (1_000_000, "百万円"),
        (100_000_000, "億円"),
    ]

    if max_amount <= 0:
        return units[0]

    for scale, label in units:
        if max_amount / scale <= 300:
            return scale, label

    return units[-1]


def int_input(label: str, value: int) -> int:
    """Text input for integer values displayed with thousands separators."""
    text = st.text_input(label, format_int(value))
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return value


def render_sidebar_summary() -> Optional[str]:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.sidebar.caption("データを取り込むと最新サマリーが表示されます。")
        return None

    months = month_options(year_df)
    if not months:
        st.sidebar.caption("月次データが存在しません。")
        return None

    end_m = months[-1]
    unit = st.session_state.settings.get("currency_unit", "円")
    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)
    sku_cnt = int(year_df["product_code"].nunique())
    rec_cnt = int(len(year_df))

    total_txt = format_amount(kpi.get("total_year_sum"), unit)
    yoy_val = kpi.get("yoy")
    yoy_txt = f"{yoy_val * 100:.1f}%" if yoy_val is not None else "—"
    delta_txt = format_amount(kpi.get("delta"), unit)
    hhi_txt = f"{hhi_val:.3f}" if hhi_val is not None else "—"

    st.sidebar.markdown(
        f"""
        <div class=\"mck-sidebar-summary\">
            <strong>最新月:</strong> {end_m}<br>
            <strong>年計総額:</strong> {total_txt}<br>
            <strong>YoY:</strong> {yoy_txt}<br>
            <strong>Δ:</strong> {delta_txt}<br>
            <strong>HHI:</strong> {hhi_txt}<br>
            <strong>SKU数:</strong> {sku_cnt:,}<br>
            <strong>レコード:</strong> {rec_cnt:,}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return end_m


def build_copilot_context(
    focus: str, end_month: Optional[str] = None, top_n: int = 5
) -> str:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        return "データが取り込まれていません。"

    months = month_options(year_df)
    if not months:
        return "月度情報が存在しません。"

    end_m = end_month or months[-1]
    snap = (
        year_df[year_df["month"] == end_m]
        .dropna(subset=["year_sum"])
        .copy()
    )
    if snap.empty:
        return f"{end_m}の年計スナップショットが空です。"

    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)

    def fmt_amt(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{format_int(val)}円"

    def fmt_pct(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{val * 100:.1f}%"

    lines = [
        f"対象月: {end_m}",
        f"年計総額: {fmt_amt(kpi.get('total_year_sum'))}",
        f"年計YoY: {fmt_pct(kpi.get('yoy'))}",
        f"前月差Δ: {fmt_amt(kpi.get('delta'))}",
        f"SKU数: {snap['product_code'].nunique():,}",
    ]
    if hhi_val is not None:
        lines.append(f"HHI: {hhi_val:.3f}")

    if focus == "伸びているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(top_n)
        )
        label = "伸長SKU"
    elif focus == "苦戦しているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=True)
            .head(top_n)
        )
        label = "苦戦SKU"
    else:
        subset = snap.sort_values("year_sum", ascending=False).head(top_n)
        label = "主要SKU"

    if not subset.empty:
        bullets = []
        for _, row in subset.iterrows():
            name = row.get("product_name") or row.get("product_code")
            yoy_txt = fmt_pct(row.get("yoy"))
            delta_txt = fmt_amt(row.get("delta"))
            bullets.append(
                f"{name} (年計 {fmt_amt(row.get('year_sum'))}, YoY {yoy_txt}, Δ {delta_txt})"
            )
        lines.append(f"{label}: " + " / ".join(bullets))

    worst = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=True)
        .head(1)
    )
    best = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=False)
        .head(1)
    )
    if not best.empty:
        b = best.iloc[0]
        lines.append(
            f"YoY最高: {(b['product_name'] or b['product_code'])} ({fmt_pct(b['yoy'])})"
        )
    if not worst.empty:
        w = worst.iloc[0]
        lines.append(
            f"YoY最低: {(w['product_name'] or w['product_code'])} ({fmt_pct(w['yoy'])})"
        )

    return " ｜ ".join(lines)


def marker_step(dates, target_points=24):
    n = len(pd.unique(dates))
    return max(1, round(n / target_points))


NAME_MAP = {
    "year_sum": "年計（12ヶ月累計）",
    "yoy": "YoY（前年同月比）",
    "delta": "Δ（前月差）",
    "slope6m": "直近6ヶ月の傾き",
    "std6m": "直近6ヶ月の変動",
    "slope_beta": "直近Nの傾き",
    "hhi_share": "HHI寄与度",
}



# ---------------- Sidebar ----------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-app-brand">
        <div class="sidebar-app-brand__title">{APP_TITLE}</div>
        <p class="sidebar-app-brand__caption">メニューは色分けされ、各機能の役割がひと目で分かります。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title(t("sidebar.navigation_title"))

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .sidebar-app-brand{
      background:linear-gradient(135deg, rgba(255,255,255,0.24), rgba(255,255,255,0.06));
      border-radius:18px;
      padding:1rem 1.1rem;
      border:1px solid rgba(255,255,255,0.2);
      box-shadow:0 14px 32px rgba(7,32,54,0.32);
      margin-bottom:1.1rem;
    }
    [data-testid="stSidebar"] .sidebar-app-brand__title{
      font-size:1.18rem;
      font-weight:800;
      letter-spacing:.08em;
      margin:0 0 .35rem;
      color:#ffffff;
    }
    [data-testid="stSidebar"] .sidebar-app-brand__caption{
      font-size:0.9rem;
      line-height:1.55;
      color:rgba(255,255,255,0.86);
      margin:0;
    }
    [data-testid="stSidebar"] .sidebar-legend{
      background:rgba(255,255,255,0.08);
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.2);
      padding:0.75rem 0.85rem;
      margin:0 0 0.9rem;
      box-shadow:0 8px 18px rgba(7,32,54,0.28);
    }
    [data-testid="stSidebar"] .sidebar-legend__title{
      font-size:0.78rem;
      letter-spacing:.12em;
      text-transform:uppercase;
      margin:0 0 0.55rem;
      color:rgba(255,255,255,0.72);
      font-weight:700;
    }
    [data-testid="stSidebar"] .sidebar-legend__items{
      display:flex;
      flex-wrap:wrap;
      gap:0.4rem;
    }
    [data-testid="stSidebar"] .sidebar-legend__item{
      display:inline-flex;
      align-items:center;
      gap:0.35rem;
      padding:0.25rem 0.6rem;
      border-radius:999px;
      background:rgba(255,255,255,0.12);
      color:#ffffff;
      font-size:0.82rem;
      font-weight:600;
      box-shadow:0 4px 10px rgba(7,32,54,0.22);
    }
    [data-testid="stSidebar"] .sidebar-legend__item::before{
      content:"";
      width:0.55rem;
      height:0.55rem;
      border-radius:50%;
      background:var(--legend-color,#71b7d4);
      box-shadow:0 0 0 3px rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] .sidebar-legend__hint{
      margin:0.6rem 0 0;
      font-size:0.78rem;
      color:rgba(255,255,255,0.7);
    }
    [data-testid="stSidebar"] label.nav-pill{
      display:flex;
      align-items:flex-start;
      gap:0.75rem;
      padding:0.85rem 0.95rem;
      border-radius:16px;
      border:1px solid rgba(255,255,255,0.16);
      background:rgba(255,255,255,0.06);
      margin-bottom:0.5rem;
      box-shadow:0 14px 26px rgba(7,32,54,0.28);
      position:relative;
      transition:transform .12s ease, border-color .12s ease, background-color .12s ease, box-shadow .12s ease;
    }
    [data-testid="stSidebar"] label.nav-pill:hover{
      transform:translateY(-2px);
      border-color:rgba(255,255,255,0.4);
      background:rgba(255,255,255,0.12);
      box-shadow:0 18px 32px rgba(7,32,54,0.34);
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__icon{
      width:2.6rem;
      height:2.6rem;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:1.35rem;
      background:rgba(var(--nav-accent-rgb,71,183,212),0.18);
      border:2px solid rgba(var(--nav-accent-rgb,71,183,212),0.45);
      box-shadow:0 10px 20px rgba(var(--nav-accent-rgb,71,183,212),0.35);
      color:#ffffff;
      flex-shrink:0;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__body{
      display:flex;
      flex-direction:column;
      gap:0.2rem;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__badge{
      display:inline-flex;
      align-items:center;
      justify-content:flex-start;
      gap:0.3rem;
      font-size:0.75rem;
      font-weight:700;
      padding:0.18rem 0.55rem;
      border-radius:999px;
      background:rgba(var(--nav-accent-rgb,71,183,212),0.28);
      color:#ffffff;
      width:max-content;
      letter-spacing:.06em;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__badge:empty{
      display:none;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__title{
      font-size:1rem;
      font-weight:700;
      color:#f8fbff !important;
      line-height:1.2;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__desc{
      font-size:0.85rem;
      line-height:1.35;
      color:rgba(255,255,255,0.82) !important;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__desc:empty{
      display:none;
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active{
      border-color:rgba(var(--nav-accent-rgb,71,183,212),0.65);
      background:rgba(var(--nav-accent-rgb,71,183,212),0.25);
      box-shadow:0 20px 36px rgba(var(--nav-accent-rgb,71,183,212),0.48);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__icon{
      background:rgba(var(--nav-accent-rgb,71,183,212),0.35);
      border-color:rgba(var(--nav-accent-rgb,71,183,212),0.85);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__badge{
      background:rgba(var(--nav-accent-rgb,71,183,212),0.55);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__title{
      color:#ffffff !important;
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__desc{
      color:rgba(255,255,255,0.92) !important;
    }
    .has-tooltip{
      position:relative;
    }
    .has-tooltip::after,
    .has-tooltip::before{
      pointer-events:none;
      opacity:0;
      transition:opacity .15s ease, transform .15s ease;
    }
    .has-tooltip[data-tooltip=""]::after,
    .has-tooltip[data-tooltip=""]::before{
      display:none;
    }
    .has-tooltip::after{
      content:attr(data-tooltip);
      position:absolute;
      left:50%;
      bottom:calc(100% + 8px);
      transform:translate(-50%, 0);
      background:rgba(11,23,38,0.92);
      color:#ffffff;
      padding:0.45rem 0.7rem;
      border-radius:10px;
      font-size:0.78rem;
      line-height:1.4;
      max-width:260px;
      text-align:center;
      box-shadow:0 12px 28px rgba(7,32,54,0.38);
      white-space:pre-wrap;
      z-index:60;
    }
    .has-tooltip::before{
      content:"";
      position:absolute;
      left:50%;
      bottom:calc(100% + 2px);
      transform:translate(-50%, 0);
      border:6px solid transparent;
      border-top-color:rgba(11,23,38,0.92);
      z-index:60;
    }
    .has-tooltip:hover::after,
    .has-tooltip:hover::before,
    .has-tooltip:focus-visible::after,
    .has-tooltip:focus-visible::before{
      opacity:1;
      transform:translate(-50%, -4px);
    }
    .tour-step-guide{
      display:flex;
      flex-wrap:wrap;
      gap:0.9rem;
      margin:0 0 1.2rem;
    }
    .tour-step-guide__item{
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:0.45rem;
      padding:0.75rem 0.9rem;
      border-radius:14px;
      border:1px solid var(--border);
      background:var(--panel);
      box-shadow:0 12px 26px rgba(11,44,74,0.14);
      min-width:120px;
      position:relative;
      transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }
    .tour-step-guide__item:hover{
      transform:translateY(-3px);
      box-shadow:0 18px 32px rgba(11,44,74,0.18);
    }
    .tour-step-guide__item[data-active="true"]{
      border-color:rgba(15,76,129,0.55);
      box-shadow:0 20px 40px rgba(15,76,129,0.22);
    }
    .tour-step-guide__item:focus-visible{
      outline:3px solid rgba(15,76,129,0.35);
      outline-offset:3px;
    }
    .tour-step-guide__icon{
      width:48px;
      height:48px;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:1.45rem;
      background:rgba(15,76,129,0.1);
      color:var(--accent-strong);
      box-shadow:0 10px 20px rgba(15,76,129,0.12);
    }
    .tour-step-guide__label{
      font-size:0.95rem;
      font-weight:700;
      color:var(--accent-strong);
      text-align:center;
      line-height:1.3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SIDEBAR_CATEGORY_STYLES = {
    "basic": {"label": "基本データ", "color": "#2d6f8e"},
    "insight": {"label": "深掘り分析", "color": "#71b7d4"},
    "risk": {"label": "リスク分析", "color": "#f2994a"},
    "management": {"label": "運用・共有", "color": "#b28cf5"},
}
SIDEBAR_CATEGORY_ORDER = ["basic", "insight", "risk", "management"]

SIDEBAR_PAGES = [
    {
        "key": "dashboard",
        "page": "ダッシュボード",
        "icon": "🏠",
        "title": "ホーム",
        "tagline": "分析ダッシュボード",
        "tooltip": "主要KPIとトレンドを俯瞰できるダッシュボードです。",
        "category": "basic",
    },
    {
        "key": "ranking",
        "page": "ランキング",
        "icon": "📊",
        "title": "ランキング",
        "tagline": "指標別トップ・ボトム",
        "tooltip": "指定月の上位・下位SKUを指標別に比較して勢いを捉えます。",
        "category": "insight",
    },
    {
        "key": "compare",
        "page": "比較ビュー",
        "icon": "🔁",
        "title": "比較ビュー",
        "tagline": "SKU横断の推移比較",
        "tooltip": "複数SKUの推移を重ね合わせ、変化の違いを見比べます。",
        "category": "insight",
    },
    {
        "key": "detail",
        "page": "SKU詳細",
        "icon": "🧾",
        "title": "SKU詳細",
        "tagline": "個別SKUの深掘り",
        "tooltip": "個別SKUの時系列やAIサマリーで背景を確認します。",
        "category": "insight",
    },
    {
        "key": "correlation",
        "page": "相関分析",
        "icon": "🔗",
        "title": "相関分析",
        "tagline": "指標のつながり分析",
        "tooltip": "散布図と相関係数で指標同士やSKU間の関係を把握します。",
        "category": "insight",
    },
    {
        "key": "category",
        "page": "併買カテゴリ",
        "icon": "🛍️",
        "title": "併買カテゴリ",
        "tagline": "併買パターンの探索",
        "tooltip": "購買ネットワークのクラスタリングでクロスセル候補を探します。",
        "category": "insight",
    },
    {
        "key": "import",
        "page": "データ取込",
        "icon": "📥",
        "title": "データ取込",
        "tagline": "CSV/Excelアップロード",
        "tooltip": "CSVやExcelの月次データを取り込み、分析用データを整えます。",
        "category": "basic",
    },
    {
        "key": "anomaly",
        "page": "異常検知",
        "icon": "⚠️",
        "title": "異常検知",
        "tagline": "異常値とリスク検知",
        "tooltip": "回帰残差を基にした異常値スコアでリスク兆候を洗い出します。",
        "category": "risk",
    },
    {
        "key": "alert",
        "page": "アラート",
        "icon": "🚨",
        "title": "アラート",
        "tagline": "しきい値ベースの監視",
        "tooltip": "設定した条件に該当するSKUをリスト化し、対応優先度を整理します。",
        "category": "risk",
    },
    {
        "key": "settings",
        "page": "設定",
        "icon": "⚙️",
        "title": "設定",
        "tagline": "集計条件の設定",
        "tooltip": "年計ウィンドウや通貨単位など、分析前提を調整します。",
        "category": "management",
    },
    {
        "key": "saved",
        "page": "保存ビュー",
        "icon": "💾",
        "title": "保存ビュー",
        "tagline": "条件の保存と共有",
        "tooltip": "現在の設定や比較条件を保存し、ワンクリックで再現します。",
        "category": "management",
    },
]

SIDEBAR_PAGE_LOOKUP = {page["key"]: page for page in SIDEBAR_PAGES}
NAV_KEYS = [page["key"] for page in SIDEBAR_PAGES]
NAV_TITLE_LOOKUP = {page["key"]: page["title"] for page in SIDEBAR_PAGES}
page_lookup = {page["key"]: page["page"] for page in SIDEBAR_PAGES}


def _hex_to_rgb_string(color: str) -> str:
    stripped = color.lstrip("#")
    if len(stripped) == 6:
        try:
            r, g, b = (int(stripped[i : i + 2], 16) for i in (0, 2, 4))
            return f"{r}, {g}, {b}"
        except ValueError:
            pass
    return "71, 183, 212"


NAV_HOVER_LOOKUP: Dict[str, str] = {}
nav_client_data: List[Dict[str, str]] = []
for page in SIDEBAR_PAGES:
    category_info = SIDEBAR_CATEGORY_STYLES.get(page["category"], {})
    color = category_info.get("color", "#71b7d4")
    hover_lines = [page.get("title", "").strip()]
    tooltip_text = page.get("tooltip", "").strip()
    tagline_text = page.get("tagline", "").strip()
    if tooltip_text:
        hover_lines.append(tooltip_text)
    elif tagline_text:
        hover_lines.append(tagline_text)
    hover_text = "\n".join(filter(None, hover_lines))
    nav_client_data.append(
        {
            "key": page["key"],
            "title": page["title"],
            "tagline": page.get("tagline", ""),
            "icon": page.get("icon", ""),
            "tooltip": page.get("tooltip", ""),
            "category": page["category"],
            "category_label": category_info.get("label", ""),
            "color": color,
            "rgb": _hex_to_rgb_string(color),
            "hover_text": hover_text,
        }
    )
    NAV_HOVER_LOOKUP[page["key"]] = hover_text

used_category_keys = [
    cat for cat in SIDEBAR_CATEGORY_ORDER if any(p["category"] == cat for p in SIDEBAR_PAGES)
]
if used_category_keys:
    legend_items_html = "".join(
        f"<span class='sidebar-legend__item' style='--legend-color:{SIDEBAR_CATEGORY_STYLES[cat]['color']};'>{SIDEBAR_CATEGORY_STYLES[cat]['label']}</span>"
        for cat in used_category_keys
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-legend">
            <p class="sidebar-legend__title">色でカテゴリを表示しています</p>
            <div class="sidebar-legend__items">{legend_items_html}</div>
            <p class="sidebar-legend__hint">アイコンにカーソルを合わせると各機能の説明が表示されます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

TOUR_STEPS: List[Dict[str, str]] = [
    {
        "key": "import",
        "nav_key": "import",
        "label": SIDEBAR_PAGE_LOOKUP["import"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["import"]["page"],
        "heading": "データ取込",
        "title": "データ取込",
        "section": "基礎編",
        "description": "最初に月次売上データをアップロードし、分析ダッシュボードを有効化します。",
        "details": "テンプレートのマッピングを完了すると基礎編の残りステップをすぐに確認できます。",
    },
    {
        "key": "dashboard",
        "nav_key": "dashboard",
        "label": SIDEBAR_PAGE_LOOKUP["dashboard"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["dashboard"]["page"],
        "heading": "ダッシュボード",
        "title": "ダッシュボード",
        "section": "基礎編",
        "description": "年計KPIと総合トレンドを俯瞰し、AIサマリーで直近の動きを素早く把握します。",
        "details": "ハイライト/ランキングタブで主要SKUの変化を数クリックでチェック。",
    },
    {
        "key": "ranking",
        "nav_key": "ranking",
        "label": SIDEBAR_PAGE_LOOKUP["ranking"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["ranking"]["page"],
        "heading": "ランキング",
        "title": "ランキング",
        "section": "基礎編",
        "description": "指定月の上位・下位SKUを指標別に比較し、勢いのある商品を短時間で把握します。",
        "details": "並び順や指標を切り替えて気になるSKUを絞り込み、必要に応じてCSV/Excelで共有。",
    },
    {
        "key": "compare",
        "nav_key": "compare",
        "label": SIDEBAR_PAGE_LOOKUP["compare"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["compare"]["page"],
        "heading": "マルチ商品比較",
        "title": "比較ビュー",
        "section": "応用編",
        "description": "条件で絞った複数SKUの推移を重ね合わせ、帯やバンドで素早く切り替えます。",
        "details": "操作バーで期間や表示を選び、スモールマルチプルで個別の動きを確認。",
    },
    {
        "key": "detail",
        "nav_key": "detail",
        "label": SIDEBAR_PAGE_LOOKUP["detail"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["detail"]["page"],
        "heading": "SKU 詳細",
        "title": "SKU詳細",
        "section": "応用編",
        "description": "個別SKUの時系列と指標を確認し、メモやタグでアクションを記録します。",
        "details": "単品/複数比較モードとAIサマリーで詳細な解釈を補助。",
    },
    {
        "key": "anomaly",
        "nav_key": "anomaly",
        "label": SIDEBAR_PAGE_LOOKUP["anomaly"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["anomaly"]["page"],
        "heading": "異常検知",
        "title": "異常検知",
        "section": "応用編",
        "description": "回帰残差ベースで異常な月次を検知し、スコアの高い事象を優先的に確認します。",
        "details": "窓幅・閾値を調整し、AI異常サマリーで発生背景を把握。",
    },
    {
        "key": "correlation",
        "nav_key": "correlation",
        "label": SIDEBAR_PAGE_LOOKUP["correlation"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["correlation"]["page"],
        "heading": "相関分析",
        "title": "相関分析",
        "section": "応用編",
        "description": "指標間の関係性やSKU同士の動きを散布図と相関係数で分析します。",
        "details": "相関指標や対象SKUを選び、外れ値の注釈からインサイトを発見。",
    },
    {
        "key": "category",
        "nav_key": "category",
        "label": SIDEBAR_PAGE_LOOKUP["category"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["category"]["page"],
        "heading": "購買カテゴリ探索",
        "title": "併買カテゴリ",
        "section": "応用編",
        "description": "購買ネットワークをクラスタリングしてクロスセル候補のグルーピングを見つけます。",
        "details": "入力データや閾値・検出法を変え、ネットワーク可視化をチューニング。",
    },
    {
        "key": "alert",
        "nav_key": "alert",
        "label": SIDEBAR_PAGE_LOOKUP["alert"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["alert"]["page"],
        "heading": "アラート",
        "title": "アラート",
        "section": "応用編",
        "description": "設定した閾値に該当するリスクSKUを一覧化し、優先度の高い対応を整理します。",
        "details": "CSVダウンロードで日次の共有や監視に活用。",
    },
    {
        "key": "settings",
        "nav_key": "settings",
        "label": SIDEBAR_PAGE_LOOKUP["settings"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["settings"]["page"],
        "heading": "設定",
        "title": "設定",
        "section": "応用編",
        "description": "年計ウィンドウやアラート条件など、分析の前提を調整します。",
        "details": "変更後は再計算ボタンでデータを更新し、全ページに反映します。",
    },
    {
        "key": "saved",
        "nav_key": "saved",
        "label": SIDEBAR_PAGE_LOOKUP["saved"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["saved"]["page"],
        "heading": "保存ビュー",
        "title": "保存ビュー",
        "section": "応用編",
        "description": "現在の設定や比較条件を名前付きで保存し、ワンクリックで再現できます。",
        "details": "設定と比較条件を共有し、分析の再現性を高めます。",
    },
]


TOUR_SECTION_ORDER: List[str] = []
TOUR_SECTION_COUNTS: Dict[str, int] = {}
for step in TOUR_STEPS:
    section_name = step.get("section") or "応用編"
    if section_name not in TOUR_SECTION_COUNTS:
        TOUR_SECTION_ORDER.append(section_name)
        TOUR_SECTION_COUNTS[section_name] = 0
    TOUR_SECTION_COUNTS[section_name] += 1
    step["section"] = section_name

section_positions: Dict[str, int] = {section: 0 for section in TOUR_SECTION_ORDER}
for step in TOUR_STEPS:
    section_name = step.get("section") or "応用編"
    section_positions[section_name] = section_positions.get(section_name, 0) + 1
    step["section_index"] = section_positions[section_name]
    step["section_total"] = TOUR_SECTION_COUNTS.get(section_name, len(TOUR_STEPS))


def render_step_guide(active_nav_key: str) -> None:
    if not TOUR_STEPS:
        return

    items_html: List[str] = []
    for step in TOUR_STEPS:
        nav_key = step.get("nav_key")
        if not nav_key:
            continue

        nav_meta = SIDEBAR_PAGE_LOOKUP.get(nav_key)
        if not nav_meta:
            continue

        label_text = (
            nav_meta.get("title")
            or step.get("title")
            or step.get("label")
            or nav_key
        )
        icon_text = nav_meta.get("icon", "")

        tooltip_candidates = [
            NAV_HOVER_LOOKUP.get(nav_key, "").strip(),
            (
                f"{step.get('section', '').strip()} {step.get('section_index', 0)} / {step.get('section_total', 0)}"
                if step.get("section")
                else ""
            ),
            step.get("description", "").strip(),
            step.get("details", "").strip(),
        ]
        tooltip_parts: List[str] = []
        for candidate in tooltip_candidates:
            if candidate and candidate not in tooltip_parts:
                tooltip_parts.append(candidate)

        tooltip_text = "\n".join(tooltip_parts)
        tooltip_attr = html.escape(tooltip_text, quote=True).replace("\n", "&#10;")
        title_text = tooltip_text.replace("\n", " ") if tooltip_text else label_text
        title_attr = html.escape(title_text, quote=True)
        aria_label_text = tooltip_text.replace("\n", " ") if tooltip_text else label_text
        aria_label_attr = html.escape(aria_label_text, quote=True)

        icon_html = html.escape(icon_text)
        label_html = html.escape(label_text)
        data_active = "true" if nav_key == active_nav_key else "false"
        aria_current_attr = ' aria-current="step"' if nav_key == active_nav_key else ""

        item_html = (
            f'<div class="tour-step-guide__item has-tooltip" data-step="{nav_key}" '
            f'data-active="{data_active}" data-tooltip="{tooltip_attr}" title="{title_attr}" '
            f'tabindex="0" role="listitem" aria-label="{aria_label_attr}"{aria_current_attr}>'
            f'<span class="tour-step-guide__icon" aria-hidden="true">{icon_html}</span>'
            f'<span class="tour-step-guide__label">{label_html}</span>'
            "</div>"
        )
        items_html.append(item_html)

    if not items_html:
        return

    st.markdown(
        f'<div class="tour-step-guide" role="list" aria-label="主要ナビゲーションステップ">'
        f'{"".join(items_html)}</div>',
        unsafe_allow_html=True,
    )


if st.session_state.get("tour_active", True) and TOUR_STEPS:
    initial_idx = max(0, min(st.session_state.get("tour_step_index", 0), len(TOUR_STEPS) - 1))
    default_key = TOUR_STEPS[initial_idx]["nav_key"]
    if default_key not in NAV_KEYS:
        default_key = NAV_KEYS[0]
else:
    default_key = NAV_KEYS[0]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = default_key

if "tour_pending_nav" in st.session_state:
    pending = st.session_state.pop("tour_pending_nav")
    if pending in NAV_KEYS:
        st.session_state["nav_page"] = pending

page_key = st.sidebar.radio(
    "利用する機能を選択",
    NAV_KEYS,
    key="nav_page",
    format_func=lambda key: NAV_TITLE_LOOKUP.get(key, key),
)
page = page_lookup[page_key]

nav_script_payload = json.dumps(nav_client_data, ensure_ascii=False)
nav_script_template = """
<script>
const NAV_DATA = {payload};
(function() {
    const doc = window.parent.document;
    const apply = () => {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return false;
        const radioGroup = sidebar.querySelector('div[data-baseweb="radio"]');
        if (!radioGroup) return false;
        const labels = Array.from(radioGroup.querySelectorAll('label'));
        if (!labels.length) return false;
        const metaByKey = Object.fromEntries(NAV_DATA.map((item) => [item.key, item]));
        const updateActiveState = () => {
            labels.forEach((label) => {
                const input = label.querySelector('input[type="radio"]');
                if (!input) return;
                label.classList.toggle('nav-pill--active', input.checked);
            });
        };
        labels.forEach((label) => {
            const input = label.querySelector('input[type="radio"]');
            if (!input) return;
            const meta = metaByKey[input.value];
            if (!meta) return;
            const metaTitle = meta.title || '';
            label.dataset.navKey = meta.key;
            label.dataset.navCategory = meta.category;
            const tooltipText = (meta.hover_text || meta.tooltip || meta.tagline || '').trim();
            const ariaLabel = tooltipText
                ? (tooltipText.startsWith(metaTitle) ? tooltipText : `${metaTitle}: ${tooltipText}`)
                : metaTitle;
            label.setAttribute('title', tooltipText);
            label.setAttribute('aria-label', ariaLabel);
            label.dataset.tooltip = tooltipText;
            label.classList.add('has-tooltip');
            label.style.setProperty('--nav-accent', meta.color || '#71b7d4');
            label.style.setProperty('--nav-accent-rgb', meta.rgb || '71, 183, 212');
            if (!label.classList.contains('nav-pill')) {
                label.classList.add('nav-pill');
            }
            const spans = label.querySelectorAll('span');
            let textSpan = null;
            if (spans.length) {
                textSpan = spans[spans.length - 1];
            }
            if (textSpan) {
                textSpan.classList.add('nav-pill__body');
                if (!textSpan.querySelector('.nav-pill__title')) {
                    textSpan.innerHTML = `
                        <span class="nav-pill__badge"></span>
                        <span class="nav-pill__title"></span>
                        <span class="nav-pill__desc"></span>
                    `;
                }
                const badgeEl = textSpan.querySelector('.nav-pill__badge');
                if (badgeEl) {
                    badgeEl.textContent = meta.category_label || '';
                }
                const titleEl = textSpan.querySelector('.nav-pill__title');
                if (titleEl) {
                    titleEl.textContent = meta.title || '';
                }
                const descEl = textSpan.querySelector('.nav-pill__desc');
                if (descEl) {
                    descEl.textContent = meta.tagline || '';
                }
            }
            let iconSpan = label.querySelector('.nav-pill__icon');
            if (!iconSpan) {
                iconSpan = doc.createElement('span');
                iconSpan.className = 'nav-pill__icon';
                iconSpan.textContent = meta.icon || '';
                if (textSpan) {
                    label.insertBefore(iconSpan, textSpan);
                } else {
                    label.appendChild(iconSpan);
                }
            } else {
                iconSpan.textContent = meta.icon || '';
            }
            iconSpan.setAttribute('aria-hidden', 'true');
            input.setAttribute('aria-label', ariaLabel);
            input.setAttribute('title', tooltipText);
            if (!input.dataset.navEnhanced) {
                input.addEventListener('change', updateActiveState);
                input.dataset.navEnhanced = 'true';
            }
        });
        updateActiveState();
        return true;
    };
    const schedule = (attempt = 0) => {
        const ready = apply();
        if (!ready && attempt < 10) {
            setTimeout(() => schedule(attempt + 1), 120);
        }
    };
    schedule();
})();
</script>
"""
nav_script = nav_script_template.replace("{payload}", nav_script_payload)
components.html(nav_script, height=0)

if st.session_state.get("tour_active", True):
    for idx, step in enumerate(TOUR_STEPS):
        if step["nav_key"] == page_key:
            st.session_state.tour_step_index = idx
            break


latest_month = render_sidebar_summary()

sidebar_state: Dict[str, object] = {}
year_df = st.session_state.get("data_year")

if year_df is not None and not year_df.empty:
    if page == "ダッシュボード":
        st.sidebar.subheader("期間選択")
        period_options = [12, 24, 36]
        default_period = st.session_state.settings.get("window", 12)
        if default_period not in period_options:
            default_period = 12
        st.sidebar.selectbox(
            "集計期間",
            period_options,
            index=period_options.index(default_period),
            key="sidebar_period",
            format_func=lambda v: f"{v}ヶ月",
            on_change=lambda: log_click("期間選択"),
        )
        unit_options = list(UNIT_MAP.keys())
        default_unit = st.session_state.settings.get("currency_unit", "円")
        if default_unit not in unit_options:
            default_unit = unit_options[0]
        st.sidebar.selectbox(
            "表示単位",
            unit_options,
            index=unit_options.index(default_unit),
            key="sidebar_unit",
            on_change=lambda: log_click("表示単位"),
        )
        st.sidebar.subheader("表示月")
        sidebar_state["dashboard_end_month"] = end_month_selector(
            year_df,
            key="end_month_dash",
            label="表示月",
            sidebar=True,
        )
    elif page == "ランキング":
        st.sidebar.subheader("期間選択")
        sidebar_state["rank_end_month"] = end_month_selector(
            year_df,
            key="end_month_rank",
            label="ランキング対象月",
            sidebar=True,
        )
        st.sidebar.subheader("評価指標")
        metric_options = [
            ("年計（12カ月累計）", "year_sum"),
            ("前年同月比（YoY）", "yoy"),
            ("前月差（Δ）", "delta"),
            ("直近傾き（β）", "slope_beta"),
        ]
        selected_metric = st.sidebar.selectbox(
            "表示指標",
            metric_options,
            format_func=lambda opt: opt[0],
            key="sidebar_rank_metric",
        )
        sidebar_state["rank_metric"] = selected_metric[1]
        order_options = [
            ("降順 (大きい順)", "desc"),
            ("昇順 (小さい順)", "asc"),
        ]
        selected_order = st.sidebar.selectbox(
            "並び順",
            order_options,
            format_func=lambda opt: opt[0],
            key="sidebar_rank_order",
        )
        sidebar_state["rank_order"] = selected_order[1]
        sidebar_state["rank_hide_zero"] = st.sidebar.checkbox(
            "年計ゼロを除外",
            value=True,
            key="sidebar_rank_hide_zero",
        )
    elif page == "比較ビュー":
        st.sidebar.subheader("期間選択")
        sidebar_state["compare_end_month"] = end_month_selector(
            year_df,
            key="compare_end_month",
            label="比較対象月",
            sidebar=True,
        )
    elif page == "SKU詳細":
        st.sidebar.subheader("期間選択")
        sidebar_state["detail_end_month"] = end_month_selector(
            year_df,
            key="end_month_detail",
            label="詳細確認月",
            sidebar=True,
        )
    elif page == "相関分析":
        st.sidebar.subheader("期間選択")
        sidebar_state["corr_end_month"] = end_month_selector(
            year_df,
            key="corr_end_month",
            label="分析対象月",
            sidebar=True,
        )
    elif page == "アラート":
        st.sidebar.subheader("期間選択")
        sidebar_state["alert_end_month"] = end_month_selector(
            year_df,
            key="end_month_alert",
            label="評価対象月",
            sidebar=True,
        )

st.sidebar.divider()

with st.sidebar.expander("AIコパイロット", expanded=False):
    st.caption("最新の年計スナップショットを使って質問できます。")
    st.text_area(
        "聞きたいこと",
        key="copilot_question",
        height=90,
        placeholder="例：前年同月比が高いSKUや、下落しているSKUを教えて",
    )
    focus = st.selectbox(
        "フォーカス",
        ["全体サマリー", "伸びているSKU", "苦戦しているSKU"],
        key="copilot_focus",
    )
    if st.button("AIに質問", key="ask_ai", use_container_width=True):
        question = st.session_state.get("copilot_question", "").strip()
        if not question:
            st.warning("質問を入力してください。")
        else:
            context = build_copilot_context(focus, end_month=latest_month)
            answer = _ai_answer(question, context)
            st.session_state.copilot_answer = answer
            st.session_state.copilot_context = context
    if st.session_state.copilot_answer:
        st.markdown(
            f"<div class='mck-ai-answer'><strong>AI回答</strong><br>{st.session_state.copilot_answer}</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.copilot_context:
            st.caption("コンテキスト: " + clip_text(st.session_state.copilot_context, 220))
st.sidebar.divider()

render_app_hero()

render_tour_banner()

render_step_guide(page_key)

if st.session_state.get("sample_data_notice"):
    st.success("サンプルデータを読み込みました。ダッシュボードからすぐに分析を確認できます。")
    st.session_state.sample_data_notice = False

if (
    st.session_state.data_year is None
    or st.session_state.data_monthly is None
):
    st.info(
        "左メニューの「データ取込」からCSVまたはExcelファイルをアップロードしてください。\n\n"
        "時間がない場合は下のサンプルデータを使ってすぐに操作感を確認できます。"
    )
    st.caption(
        "フェルミ推定ではサンプル体験により1時間以上かかる初期設定を15分程度に短縮できます。"
    )
    if st.button(
        "サンプルデータを表示する",
        type="primary",
        help="サンプルの年計データを読み込み、すべてのダッシュボード機能を体験できます。",
    ):
        sample_df = load_sample_dataset()
        settings = st.session_state.settings
        long_df = fill_missing_months(
            sample_df, policy=settings.get("missing_policy", "zero_fill")
        )
        year_df = compute_year_rolling(
            long_df,
            window=int(settings.get("window", 12)),
            policy=settings.get("missing_policy", "zero_fill"),
        )
        year_df = compute_slopes(
            year_df,
            last_n=int(settings.get("last_n", 12)),
        )
        st.session_state.data_monthly = long_df
        st.session_state.data_year = year_df
        st.session_state.sample_data_notice = True
        st.experimental_rerun()

# ---------------- Pages ----------------

# 1) データ取込
if page == "データ取込":
    section_header(
        "データ取込", "ファイルのマッピングと品質チェックを行います。", icon="📥"
    )

    st.markdown(
        """
        <style>
          .upload-guide{background:rgba(255,255,255,0.85);border:1px solid #c6d4e6;border-radius:16px;padding:18px;box-shadow:0 8px 20px rgba(18,58,95,0.08);}
          .upload-guide h4{margin-top:0;margin-bottom:0.6rem;font-weight:700;color:#0f4c81;}
          .upload-guide ul{padding-left:1.2rem;margin-top:0;margin-bottom:0.8rem;}
          .upload-guide li{margin-bottom:0.35rem;}
          .upload-guide__note{font-size:0.82rem;color:#4b5c6c;margin:0.4rem 0 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    guide_left, guide_right = st.columns([3, 2])
    with guide_left:
        st.markdown(
            """
            <div class="upload-guide">
              <h4>📋 必要列</h4>
              <ul>
                <li><strong>年月</strong>: YYYY-MM 形式（例: 2024-01）</li>
                <li><strong>チャネル</strong>: EC / 店舗などの販売チャネル</li>
                <li><strong>商品名</strong>: SKU やサービス名称</li>
                <li><strong>売上額</strong>: 金額（円・税区分は任意）</li>
              </ul>
              <h4>🔍 推奨フォーマット</h4>
              <ul>
                <li>年月は <code>YYYY-MM</code> もしくは日付形式で指定してください。</li>
                <li>チャネルと商品名はテキスト列で入力してください。</li>
                <li>売上額は数値列で、マイナス値（返品）にも対応しています。</li>
              </ul>
              <p class="upload-guide__note">※ 同一商品が複数チャネルに存在する場合はチャネル別に集計します。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_right:
        template_bytes = build_upload_template()
        st.download_button(
            "Excelテンプレートをダウンロード",
            data=template_bytes,
            file_name="rollingup_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
        )
        st.caption("テンプレートには推奨列とサンプル値があらかじめ入力されています。")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        file = st.file_uploader("ファイル選択", type=["xlsx", "csv"])
    with col_u2:
        missing_policy_options = [
            (
                "zero_fill",
                "ゼロ補完（推奨）→ 実売ゼロの欠測に適合 ／ 一時欠測は過小評価の恐れ",
            ),
            (
                "forward_fill",
                "前月値補完 → トレンド継続を想定 ／ 急激な変化を反映しづらい",
            ),
            (
                "linear_interp",
                "線形補完 → 前後の値から滑らかに補間 ／ 季節性の山谷が平坦化",
            ),
            (
                "mark_missing",
                "欠測含む窓は非計上 → 実測値のみで評価 ／ 分析期間が短くなる",
            ),
        ]
        policy_keys = [key for key, _ in missing_policy_options]
        policy_labels = {key: label for key, label in missing_policy_options}
        current_policy = st.session_state.settings.get("missing_policy", "zero_fill")
        try:
            default_index = policy_keys.index(current_policy)
        except ValueError:
            default_index = 0
        st.session_state.settings["missing_policy"] = st.selectbox(
            "欠測月ポリシー",
            options=policy_keys,
            format_func=lambda x: policy_labels.get(x, x),
            index=default_index,
        )

    if file is not None:
        try:
            with st.spinner("ファイルを読み込んでいます…"):
                if file.name.lower().endswith(".csv"):
                    df_raw = pd.read_csv(file)
                else:
                    df_raw = pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error(f"読込エラー: {e}")
            st.stop()

        st.caption("アップロードプレビュー（先頭100行）")
        st.dataframe(df_raw.head(100), use_container_width=True)

        columns_signature = tuple(df_raw.columns.tolist())
        if st.session_state.upload_signature != columns_signature:
            st.session_state.upload_signature = columns_signature
            st.session_state.upload_mapping = None

        suggestions = suggest_column_mapping(df_raw)
        if st.session_state.upload_mapping is None:
            st.session_state.upload_mapping = dict(suggestions)

        header_col, reset_col = st.columns([3, 1])
        with header_col:
            st.subheader("列の自動マッピング")
            st.caption("必要項目に列をドラッグ＆ドロップで割り当ててください。")
        with reset_col:
            if st.button("推奨マッピングにリセット"):
                st.session_state.upload_mapping = dict(suggestions)

        current_mapping = {
            field["key"]: st.session_state.upload_mapping.get(field["key"])
            for field in UPLOAD_FIELD_DEFS
        }
        mapping_update = render_column_mapping_tool(
            df_raw.columns.tolist(), current_mapping, key="upload_mapping_tool"
        )
        if mapping_update is not None:
            st.session_state.upload_mapping = mapping_update

        column_mapping = {
            field["key"]: st.session_state.upload_mapping.get(field["key"])
            for field in UPLOAD_FIELD_DEFS
        }

        summary_rows = []
        missing_required = []
        for field in UPLOAD_FIELD_DEFS:
            mapped_col = column_mapping.get(field["key"])
            if mapped_col:
                display_value = str(mapped_col)
            else:
                display_value = "未割当" if field["required"] else "未割当（任意）"
                if field["required"]:
                    missing_required.append(field["label"])
            summary_rows.append({"必要項目": field["label"], "割当列": display_value})

        st.table(pd.DataFrame(summary_rows))
        if missing_required:
            st.warning("未割当の必須項目があります: " + ", ".join(missing_required))

        convert_disabled = bool(missing_required)
        convert_help = (
            "すべての必須項目を割り当ててください。"
            if convert_disabled
            else "マッピング内容でデータを取り込みます。"
        )

        if st.button("変換＆取込", type="primary", disabled=convert_disabled, help=convert_help):
            try:
                with st.spinner("年計データを計算中…"):
                    long_df = parse_uploaded_table(
                        df_raw,
                        column_mapping=column_mapping,
                    )
                    long_df = fill_missing_months(
                        long_df, policy=st.session_state.settings["missing_policy"]
                    )
                    # Compute year rolling & slopes
                    year_df = compute_year_rolling(
                        long_df,
                        window=st.session_state.settings["window"],
                        policy=st.session_state.settings["missing_policy"],
                    )
                    year_df = compute_slopes(
                        year_df, last_n=st.session_state.settings["last_n"]
                    )

                    st.session_state.data_monthly = long_df
                    st.session_state.data_year = year_df

                st.success(
                    "取込完了。ダッシュボードへ移動して可視化を確認してください。"
                )

                st.subheader("品質チェック（欠測月/非数値/重複）")
                # 欠測月
                miss_rate = (long_df["is_missing"].sum(), len(long_df))
                st.write(f"- 欠測セル数: {miss_rate[0]:,} / {miss_rate[1]:,}")
                # 月レンジ
                st.write(
                    f"- データ期間: {long_df['month'].min()} 〜 {long_df['month'].max()}"
                )
                # SKU数
                st.write(f"- SKU数: {long_df['product_code'].nunique():,}")
                st.write(f"- レコード数: {len(long_df):,}")

                st.download_button(
                    "年計テーブルをCSVでダウンロード",
                    data=st.session_state.data_year.to_csv(index=False).encode(
                        "utf-8-sig"
                    ),
                    file_name="year_rolling.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.exception(e)

# 2) ダッシュボード
elif page == "ダッシュボード":
    require_data()
    section_header("ダッシュボード", "年計KPIと成長トレンドを俯瞰します。", icon="📈")

    period_value = st.session_state.get(
        "sidebar_period", st.session_state.settings.get("window", 12)
    )
    unit_value = st.session_state.get(
        "sidebar_unit", st.session_state.settings.get("currency_unit", "円")
    )

    # update settings and filter log
    st.session_state.settings["window"] = period_value
    st.session_state.settings["currency_unit"] = unit_value
    st.session_state.filters.update(
        {
            "period": period_value,
            "currency_unit": unit_value,
        }
    )

    year_df_full = st.session_state.data_year
    monthly_df_full = st.session_state.data_monthly

    filter_dimensions = [
        ("channel", "チャネル"),
        ("category", "商品カテゴリー"),
        ("customer_segment", "主要顧客"),
    ]
    filter_cols = st.columns(len(filter_dimensions))
    selected_filters: Dict[str, List[str]] = {}
    for idx, (col_key, label) in enumerate(filter_dimensions):
        default_values = st.session_state.filters.get(col_key, [])
        options: List[str] = []
        if year_df_full is not None and col_key in year_df_full.columns:
            series = year_df_full[col_key].dropna()
            options = sorted(pd.Series(series).astype(str).unique().tolist())
            default_values = [value for value in default_values if value in options]
            with filter_cols[idx]:
                selected = st.multiselect(
                    label,
                    options=options,
                    default=default_values,
                    placeholder="選択しない場合は全体",
                )
        else:
            with filter_cols[idx]:
                st.info(f"{label}情報がありません。")
            selected = []
        selected_filters[col_key] = selected
        st.session_state.filters[col_key] = selected

    active_filters = {k: v for k, v in selected_filters.items() if v}
    year_df = apply_dimensional_filters(year_df_full, active_filters)
    monthly_df = apply_dimensional_filters(monthly_df_full, active_filters)

    if active_filters:
        filter_labels = {
            "channel": "チャネル",
            "category": "商品カテゴリー",
            "customer_segment": "主要顧客",
        }
        summary_parts = [
            f"{filter_labels[key]}: {', '.join(values)}"
            for key, values in active_filters.items()
        ]
        st.caption("適用中のフィルタ ▶ " + " / ".join(summary_parts))
    else:
        year_df = year_df_full
        monthly_df = monthly_df_full

    if year_df is None or year_df.empty:
        st.warning("選択された条件に合致するデータがありません。フィルタを見直してください。")
        st.stop()

    end_m = sidebar_state.get("dashboard_end_month") or latest_month

    # KPIと基礎集計
    kpi = aggregate_overview(year_df, end_m)
    hhi = compute_hhi(year_df, end_m)
    unit = st.session_state.settings["currency_unit"]

    snap = (
        year_df[year_df["month"] == end_m]
        .dropna(subset=["year_sum"])
        .copy()
        .sort_values("year_sum", ascending=False)
    )

    totals = year_df.groupby("month", as_index=False)["year_sum"].sum()
    totals = totals.sort_values("month").reset_index(drop=True)
    totals["month_dt"] = pd.to_datetime(totals["month"], format="%Y-%m", errors="coerce")
    totals["year_sum_disp"] = totals["year_sum"] / UNIT_MAP[unit]
    totals["delta"] = totals["year_sum"].diff()
    yoy_vals: List[float] = []
    total_values = totals["year_sum"].tolist()
    for idx, val in enumerate(total_values):
        prev_idx = idx - 12
        yoy_val = np.nan
        if prev_idx >= 0:
            prev_val = total_values[prev_idx]
            if pd.notna(prev_val) and prev_val != 0 and pd.notna(val):
                yoy_val = (val - prev_val) / prev_val
        yoy_vals.append(yoy_val)
    totals["yoy_ratio"] = yoy_vals
    totals_line_df = totals.dropna(subset=["year_sum_disp", "month_dt"]).copy()

    monthly_totals = (
        monthly_df.groupby("month", as_index=False)["sales_amount_jpy"].sum()
        if monthly_df is not None
        else pd.DataFrame(columns=["month", "sales_amount_jpy"])
    )
    if not monthly_totals.empty:
        monthly_totals = monthly_totals.sort_values("month").reset_index(drop=True)
        monthly_totals["month_dt"] = pd.to_datetime(
            monthly_totals["month"], format="%Y-%m", errors="coerce"
        )
        monthly_totals["amount_disp"] = (
            monthly_totals["sales_amount_jpy"] / UNIT_MAP[unit]
        )

    yoy_chart_df = totals.dropna(subset=["yoy_ratio", "month_dt"]).copy()
    if not yoy_chart_df.empty:
        yoy_chart_df["yoy_pct"] = yoy_chart_df["yoy_ratio"] * 100
        yoy_chart_df["trend"] = np.where(
            yoy_chart_df["yoy_pct"] >= 0, "増加", "減少"
        )

    segment_summaries = {
        "channel": compute_segment_summary(year_df, ["channel"], end_m),
        "category": compute_segment_summary(year_df, ["category"], end_m),
        "customer_segment": compute_segment_summary(year_df, ["customer_segment"], end_m),
    }

    channel_totals = segment_summaries.get("channel")
    if channel_totals is None:
        channel_totals = pd.DataFrame(columns=["channel", "year_sum", "share"])
    channel_chart_df = pd.DataFrame(columns=["channel", "year_sum", "share"])
    total_channel_amount = 0.0
    if not channel_totals.empty and "channel" in channel_totals.columns:
        channel_chart_df = channel_totals[["channel", "year_sum", "share"]].copy()
        channel_chart_df = channel_chart_df.sort_values("year_sum", ascending=False)
        total_channel_amount = float(channel_chart_df["year_sum"].sum())
        if len(channel_chart_df) > 6:
            top_channels = channel_chart_df.head(5)
            others_sum = channel_chart_df.iloc[5:]["year_sum"].sum()
            others_share = (
                others_sum / total_channel_amount if total_channel_amount > 0 else np.nan
            )
            channel_chart_df = pd.concat(
                [
                    top_channels,
                    pd.DataFrame(
                        {
                            "channel": ["その他"],
                            "year_sum": [others_sum],
                            "share": [others_share],
                        }
                    ),
                ],
                ignore_index=True,
            )
        if total_channel_amount > 0:
            channel_chart_df["share"] = channel_chart_df["year_sum"] / total_channel_amount

    top_channel_label = "—"
    top_channel_caption = "データ不足"
    if not channel_totals.empty:
        top_channel_row = channel_totals.iloc[0]
        top_channel_label = str(top_channel_row.get("channel", "—"))
        channel_caption_parts: List[str] = []
        amount_txt = format_amount(top_channel_row.get("year_sum"), unit)
        if amount_txt != "—":
            channel_caption_parts.append(f"売上 {amount_txt}")
        share_txt = (
            format_percent(top_channel_row.get("share"), decimals=1, signed=False)
            if total_channel_amount > 0
            else "—"
        )
        if share_txt != "—":
            channel_caption_parts.append(f"シェア {share_txt}")
        growth_txt = format_percent(
            top_channel_row.get("growth_rate"), decimals=1
        )
        if growth_txt != "—":
            channel_caption_parts.append(f"YoY {growth_txt}")
        if channel_caption_parts:
            top_channel_caption = "\n".join(channel_caption_parts)
        else:
            top_channel_caption = "—"

    top_sku_label = "—"
    top_sku_caption = "データ不足"
    top_sku_delta = None
    if not snap.empty:
        top_row = snap.iloc[0]
        top_sku_label = str(
            top_row.get("product_name")
            or top_row.get("product_code")
            or "—"
        )
        sku_caption_parts: List[str] = []
        amount_txt = format_amount(top_row.get("year_sum"), unit)
        if amount_txt != "—":
            sku_caption_parts.append(f"売上 {amount_txt}")
        yoy_val = top_row.get("yoy")
        yoy_txt = format_percent(yoy_val) if yoy_val is not None else "—"
        if yoy_txt != "—":
            sku_caption_parts.append(f"YoY {yoy_txt}")
        delta_txt = format_amount_delta(top_row.get("delta"), unit)
        if delta_txt != "—":
            sku_caption_parts.append(f"Δ {delta_txt}")
        if sku_caption_parts:
            top_sku_caption = "\n".join(sku_caption_parts)
        else:
            top_sku_caption = "—"
        if yoy_val is not None and not pd.isna(yoy_val):
            top_sku_delta = float(yoy_val)

    yoy_value = kpi.get("yoy") if kpi else None
    delta_value = kpi.get("delta") if kpi else None
    total_value_text = format_amount(kpi.get("total_year_sum"), unit)
    yoy_text = format_percent(yoy_value) if yoy_value is not None else "—"
    delta_text = format_amount_delta(delta_value, unit)
    hhi_display = "—"
    if hhi is not None and not pd.isna(hhi):
        hhi_display = f"{hhi:.3f}"

    kpi_cards = [
        {
            "label": "年間売上",
            "value": total_value_text,
            "caption": f"{end_m} 時点 12ヶ月移動累計",
            "icon": "💰",
        },
        {
            "label": "前年同期比",
            "value": yoy_text,
            "caption": "年計ベース YoY",
            "icon": "📊",
            "delta": yoy_value,
        },
        {
            "label": "前月差",
            "value": delta_text,
            "caption": "年計の前月差分",
            "icon": "↕",
            "delta": delta_value,
        },
        {
            "label": "トップチャネル",
            "value": top_channel_label,
            "caption": top_channel_caption,
            "icon": "🏬",
        },
        {
            "label": "トップSKU",
            "value": top_sku_label,
            "caption": top_sku_caption,
            "icon": "🥇",
            "delta": top_sku_delta,
        },
        {
            "label": "HHI(集中度)",
            "value": hhi_display,
            "caption": "1に近いほど集中",
            "icon": "🧮",
        },
    ]

    tab_highlight, tab_ranking = st.tabs(["ハイライト", "ランキング / エクスポート"])

    with tab_highlight:
        render_kpi_cards(kpi_cards)

        ai_on = st.toggle(
            "AIサマリー",
            value=False,
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
            key="dash_ai_summary",
        )
        with st.expander("AIサマリー", expanded=ai_on):
            if ai_on:
                with st.spinner("AI要約を生成中…"):
                    kpi_text = _ai_explain(
                        {
                            "年計総額": kpi["total_year_sum"],
                            "年計YoY": kpi["yoy"],
                            "前月差Δ": kpi["delta"],
                        }
                    )
                    snap_ai = snap[["year_sum", "yoy", "delta"]].head(100)
                    stat_text = _ai_sum_df(snap_ai)
                    st.info(f"**AI説明**：{kpi_text}\n\n**AI要約**：{stat_text}")
                    actions = _ai_actions(
                        {
                            "total_year_sum": float(kpi.get("total_year_sum") or 0.0),
                            "yoy": float(kpi.get("yoy") or 0.0),
                            "delta": float(kpi.get("delta") or 0.0),
                            "hhi": float(hhi or 0.0),
                        },
                        focus=end_m,
                    )
                    st.success(f"**AI推奨アクション**：{actions}")
                    st.caption(_ai_comment("直近の年計トレンドと上位SKUの動向"))

        st.markdown("#### KPIトレンド")
        trend_left, trend_right = st.columns(2)
        with trend_left:
            if totals_line_df.empty:
                st.info("12ヶ月移動累計を算出するのに十分なデータがありません。")
            else:
                fig_trend = px.line(
                    totals_line_df,
                    x="month_dt",
                    y="year_sum_disp",
                    title="12ヶ月移動累計",
                    markers=True,
                )
                fig_trend.update_yaxes(title=f"年計({unit})", tickformat="~,d")
                fig_trend.update_xaxes(title="月", tickformat="%Y-%m")
                fig_trend.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
                fig_trend = apply_elegant_theme(
                    fig_trend, theme=st.session_state.get("ui_theme", "dark")
                )
                render_plotly_with_spinner(fig_trend, config=PLOTLY_CONFIG)
                st.caption("凡例クリックで系列の表示切替、ダブルクリックで単独表示。")
        with trend_right:
            monthly_plot = monthly_totals.dropna(subset=["month_dt"]).copy()
            if monthly_plot.empty:
                st.info("月次推移を表示するには十分なデータがありません。")
            else:
                fig_monthly = px.bar(
                    monthly_plot,
                    x="month_dt",
                    y="amount_disp",
                    title="月次売上推移",
                )
                fig_monthly.update_yaxes(title=f"月次売上({unit})", tickformat="~,d")
                fig_monthly.update_xaxes(title="月")
                fig_monthly.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
                fig_monthly = apply_elegant_theme(
                    fig_monthly, theme=st.session_state.get("ui_theme", "dark")
                )
                render_plotly_with_spinner(fig_monthly, config=PLOTLY_CONFIG)

        st.markdown("#### 成長率とチャネル構成")
        growth_left, growth_right = st.columns(2)
        with growth_left:
            yoy_plot = yoy_chart_df.dropna(subset=["month_dt"]).copy()
            if yoy_plot.empty:
                st.info("前年同期比を算出できる期間が不足しています。")
            else:
                fig_yoy = px.bar(
                    yoy_plot,
                    x="month_dt",
                    y="yoy_pct",
                    color="trend",
                    color_discrete_map={"増加": "#2d6f8e", "減少": "#c84c44"},
                    title="前年同期比の推移",
                )
                fig_yoy.update_yaxes(title="YoY(%)", tickformat=".1f")
                fig_yoy.update_xaxes(title="月")
                fig_yoy.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5)
                fig_yoy.update_layout(
                    showlegend=False, height=340, margin=dict(l=10, r=10, t=45, b=10)
                )
                fig_yoy = apply_elegant_theme(
                    fig_yoy, theme=st.session_state.get("ui_theme", "dark")
                )
                render_plotly_with_spinner(fig_yoy, config=PLOTLY_CONFIG)
        with growth_right:
            channel_plot = channel_chart_df.copy()
            if channel_plot.empty or channel_plot["year_sum"].sum() <= 0:
                st.info("チャネル別の売上構成を算出できません。")
            else:
                fig_channel = px.pie(
                    channel_plot,
                    values="year_sum",
                    names="channel",
                    title="チャネル別年計比率",
                    hole=0.45,
                )
                fig_channel.update_traces(
                    textposition="inside", texttemplate="%{label}<br>%{percent:.1%}"
                )
                fig_channel.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
                fig_channel = apply_elegant_theme(
                    fig_channel, theme=st.session_state.get("ui_theme", "dark")
                )
                render_plotly_with_spinner(fig_channel, config=PLOTLY_CONFIG)

        st.markdown("#### セグメント比較（収益性 × 成長率）")
        seg_tab_labels = ["チャネル", "商品カテゴリー", "主要顧客"]
        seg_keys = ["channel", "category", "customer_segment"]
        seg_tabs = st.tabs(seg_tab_labels)
        for seg_tab, seg_key, seg_label in zip(seg_tabs, seg_keys, seg_tab_labels):
            with seg_tab:
                render_segment_summary_tab(
                    segment_summaries.get(seg_key, pd.DataFrame()),
                    seg_key,
                    seg_label,
                    unit,
                )
        st.caption("バブルの大きさは構成比。0%線より右側は成長、左側は減速領域です。")

        st.markdown("#### ドリルダウン（セグメント ▶ 地域 ▶ 商品）")
        drill_options = {
            "チャネル": "channel",
            "商品カテゴリー": "category",
            "主要顧客": "customer_segment",
        }
        drill_choice = st.selectbox(
            "ドリルダウン軸",
            list(drill_options.keys()),
            key="dashboard_drill_axis",
        )
        drill_column = drill_options[drill_choice]
        drill_base = snap.copy()
        if drill_base.empty:
            st.info("対象データが不足しています。フィルタ条件を調整してください。")
        else:
            if "region" not in drill_base.columns:
                drill_base["region"] = "地域情報なし"
            drill_base["region"] = drill_base["region"].fillna("地域情報なし")
            if "base_product_name" not in drill_base.columns:
                drill_base["base_product_name"] = drill_base["product_name"]

            treemap_df = drill_base.rename(
                columns={
                    drill_column: drill_choice,
                    "region": "地域",
                    "base_product_name": "商品",
                    "year_sum": "年計",
                    "yoy": "YoY",
                }
            )
            treemap_df["商品"] = treemap_df["商品"].fillna("無名商品")
            treemap_df[drill_choice] = treemap_df[drill_choice].fillna("未分類")
            treemap_df["地域"] = treemap_df["地域"].fillna("地域情報なし")
            fig_treemap = px.treemap(
                treemap_df,
                path=[drill_choice, "地域", "商品"],
                values="年計",
                color="YoY",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            fig_treemap.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            fig_treemap = apply_elegant_theme(
                fig_treemap, theme=st.session_state.get("ui_theme", "dark")
            )
            render_plotly_with_spinner(fig_treemap, config=PLOTLY_CONFIG)
            st.caption("セルをクリックで下位階層にズーム。ダブルクリックで一段階戻れます。")

            region_summary = compute_segment_summary(
                year_df, [drill_column, "region"], end_m
            )
            if region_summary.empty:
                st.info("地域別の要因分析に必要なデータが不足しています。")
            else:
                display_region = region_summary.rename(
                    columns={drill_column: drill_choice, "region": "地域"}
                )
                display_region[f"年計({unit})"] = (
                    display_region["year_sum"] / UNIT_MAP[unit]
                )
                display_region["構成比(%)"] = display_region["share"] * 100
                display_region["YoY(%)"] = display_region["growth_rate"] * 100
                display_region = display_region.sort_values(
                    "year_sum", ascending=False
                )
                st.dataframe(
                    display_region[
                        [
                            drill_choice,
                            "地域",
                            f"年計({unit})",
                            "構成比(%)",
                            "YoY(%)",
                        ]
                    ].head(20),
                    use_container_width=True,
                    column_config={
                        f"年計({unit})": st.column_config.NumberColumn(
                            format="%.0f" if unit == "円" else "%.1f"
                        ),
                        "構成比(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "YoY(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    with tab_ranking:
        st.markdown(f"#### ランキング（{end_m} 時点 年計）")
        snap_disp = snap.copy()
        snap_disp["year_sum"] = snap_disp["year_sum"] / UNIT_MAP[unit]
        st.dataframe(
            snap_disp[["product_code", "product_name", "year_sum", "yoy", "delta"]].head(
                20
            ),
            use_container_width=True,
        )
        st.download_button(
            "この表をCSVでダウンロード",
            data=snap.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ranking_{end_m}.csv",
            mime="text/csv",
        )

        pdf_bytes = download_pdf_overview(
            {
                "total_year_sum": int(kpi["total_year_sum"])
                if kpi["total_year_sum"] is not None
                else 0,
                "yoy": round(kpi["yoy"], 4) if kpi["yoy"] is not None else None,
                "delta": int(kpi["delta"]) if kpi["delta"] is not None else None,
            },
            snap,
            filename=f"overview_{end_m}.pdf",
        )
        st.download_button(
            "会議用PDF（KPI+Top10）を出力",
            data=pdf_bytes,
            file_name=f"overview_{end_m}.pdf",
            mime="application/pdf",
        )

# 3) ランキング
elif page == "ランキング":
    require_data()
    section_header("ランキング", "上位と下位のSKUを瞬時に把握します。", icon="🏆")
    end_m = sidebar_state.get("rank_end_month") or latest_month
    metric = sidebar_state.get("rank_metric", "year_sum")
    order = sidebar_state.get("rank_order", "desc")
    hide_zero = sidebar_state.get("rank_hide_zero", True)

    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    snap = st.session_state.data_year[
        st.session_state.data_year["month"] == end_m
    ].copy()
    total = len(snap)
    zero_cnt = int((snap["year_sum"] == 0).sum())
    if hide_zero:
        snap = snap[snap["year_sum"] > 0]
    snap = snap.dropna(subset=[metric])
    snap = snap.sort_values(metric, ascending=(order == "asc"))
    st.caption(f"除外 {zero_cnt} 件 / 全 {total} 件")

    fig_bar = px.bar(snap.head(20), x="product_name", y=metric)
    fig_bar = apply_elegant_theme(
        fig_bar, theme=st.session_state.get("ui_theme", "dark")
    )
    render_plotly_with_spinner(fig_bar, config=PLOTLY_CONFIG)

    with st.expander("AIサマリー", expanded=ai_on):
        if ai_on and not snap.empty:
            st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]].head(200)))
            st.caption(_ai_comment("上位と下位の入替やYoYの極端値に注意"))

    st.dataframe(
        snap[
            ["product_code", "product_name", "year_sum", "yoy", "delta", "slope_beta"]
        ].head(100),
        use_container_width=True,
    )

    st.download_button(
        "CSVダウンロード",
        data=snap.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ranking_{metric}_{end_m}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Excelダウンロード",
        data=download_excel(snap, f"ranking_{metric}_{end_m}.xlsx"),
        file_name=f"ranking_{metric}_{end_m}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # 4) 比較ビュー（マルチ商品バンド）
elif page == "比較ビュー":
    require_data()
    section_header("マルチ商品比較", "条件を柔軟に切り替えてSKUを重ね合わせます。", icon="🔍")
    params = st.session_state.compare_params
    year_df = st.session_state.data_year
    end_m = sidebar_state.get("compare_end_month") or latest_month

    snapshot = latest_yearsum_snapshot(year_df, end_m)
    snapshot["display_name"] = snapshot["product_name"].fillna(snapshot["product_code"])

    search = st.text_input("検索ボックス", "")
    if search:
        snapshot = snapshot[
            snapshot["display_name"].str.contains(search, case=False, na=False)
        ]
    # ---- 操作バー＋グラフ密着カード ----

    band_params_initial = params.get("band_params", {})
    band_params = band_params_initial
    amount_slider_cfg = None
    max_amount = int(snapshot["year_sum"].max()) if not snapshot.empty else 0
    low0 = int(
        band_params_initial.get(
            "low_amount", int(snapshot["year_sum"].min()) if not snapshot.empty else 0
        )
    )
    high0 = int(band_params_initial.get("high_amount", max_amount))

    st.markdown(
        """
<style>
.chart-card { position: relative; margin:.25rem 0 1rem; border-radius:12px;
  border:1px solid var(--color-primary); background:var(--card-bg,#fff); }
.chart-toolbar { position: sticky; top: -1px; z-index: 5;
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
  padding:.35rem .6rem; background: linear-gradient(180deg, rgba(0,58,112,.08), rgba(0,58,112,.02));
  border-bottom:1px solid var(--color-primary); }
/* Streamlit標準の下マージンを除去（ここが距離の主因） */
.chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
.chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
.chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:#003a70; }
.chart-toolbar .stSlider label { color:#003a70; }
.chart-body { padding:.15rem .4rem .4rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="chart-card" id="line-compare">', unsafe_allow_html=True
    )

    st.markdown('<div class="chart-toolbar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1.1, 1.0, 0.9])
    with c1:
        period = st.radio(
            "期間", ["12ヶ月", "24ヶ月", "36ヶ月"], horizontal=True, index=1
        )
    with c2:
        node_mode = st.radio(
            "ノード表示",
            ["自動", "主要ノードのみ", "すべて", "非表示"],
            horizontal=True,
            index=0,
        )
    with c3:
        hover_mode = st.radio(
            "ホバー", ["個別", "同月まとめ"], horizontal=True, index=0
        )
    with c4:
        op_mode = st.radio("操作", ["パン", "ズーム", "選択"], horizontal=True, index=0)
    with c5:
        peak_on = st.checkbox("ピーク表示", value=False)

    c6, c7, c8 = st.columns([2.0, 1.9, 1.6])
    with c6:
        band_mode = st.radio(
            "バンド",
            ["金額指定", "商品指定(2)", "パーセンタイル", "順位帯", "ターゲット近傍"],
            horizontal=True,
            index=[
                "金額指定",
                "商品指定(2)",
                "パーセンタイル",
                "順位帯",
                "ターゲット近傍",
            ].index(params.get("band_mode", "金額指定")),
        )
    with c7:
        if band_mode == "金額指定":
            if not snapshot.empty:
                unit_scale, unit_label = choose_amount_slider_unit(max_amount)
                slider_max = int(
                    math.ceil(
                        max(
                            max_amount,
                            band_params_initial.get("high_amount", high0),
                        )
                        / unit_scale
                    )
                )
                slider_max = max(slider_max, 1)

                default_low = int(
                    round(band_params_initial.get("low_amount", low0) / unit_scale)
                )
                default_high = int(
                    round(band_params_initial.get("high_amount", high0) / unit_scale)
                )
                default_low = max(0, min(default_low, slider_max))
                default_high = max(default_low, min(default_high, slider_max))

                step = nice_slider_step(slider_max)

                amount_slider_cfg = dict(
                    label=f"金額レンジ（{unit_label}単位）",
                    min_value=0,
                    max_value=slider_max,
                    value=(default_low, default_high),
                    step=step,
                    unit_scale=unit_scale,
                    unit_label=unit_label,
                    max_amount=max_amount,
                )
            else:
                band_params = {"low_amount": low0, "high_amount": high0}
        elif band_mode == "商品指定(2)":
            if not snapshot.empty:
                opts = (
                    snapshot["product_code"].fillna("")
                    + " | "
                    + snapshot["display_name"].fillna("")
                ).tolist()
                opts = [o for o in opts if o.strip() != "|"]
                prod_a = st.selectbox("商品A", opts, index=0)
                prod_b = st.selectbox("商品B", opts, index=1 if len(opts) > 1 else 0)
                band_params = {
                    "prod_a": prod_a.split(" | ")[0],
                    "prod_b": prod_b.split(" | ")[0],
                }
            else:
                band_params = band_params_initial
        elif band_mode == "パーセンタイル":
            if not snapshot.empty:
                p_low = band_params_initial.get("p_low", 0)
                p_high = band_params_initial.get("p_high", 100)
                p_low, p_high = st.slider(
                    "百分位(%)", 0, 100, (int(p_low), int(p_high))
                )
                band_params = {"p_low": p_low, "p_high": p_high}
            else:
                band_params = {
                    "p_low": band_params_initial.get("p_low", 0),
                    "p_high": band_params_initial.get("p_high", 100),
                }
        elif band_mode == "順位帯":
            if not snapshot.empty:
                max_rank = int(snapshot["rank"].max()) if not snapshot.empty else 1
                r_low = band_params_initial.get("r_low", 1)
                r_high = band_params_initial.get("r_high", max_rank)
                r_low, r_high = st.slider(
                    "順位", 1, max_rank, (int(r_low), int(r_high))
                )
                band_params = {"r_low": r_low, "r_high": r_high}
            else:
                band_params = {
                    "r_low": band_params_initial.get("r_low", 1),
                    "r_high": band_params_initial.get("r_high", 1),
                }
        else:
            opts = (
                snapshot["product_code"] + " | " + snapshot["display_name"]
            ).tolist()
            tlabel = st.selectbox("基準商品", opts, index=0) if opts else ""
            tcode = tlabel.split(" | ")[0] if tlabel else ""
            by_default = band_params_initial.get("by", "amt")
            by_index = 0 if by_default == "amt" else 1
            by = st.radio("幅指定", ["金額", "%"], horizontal=True, index=by_index)
            if by == "金額":
                width_default = 100000
                width = int_input(
                    "幅", int(band_params_initial.get("width", width_default))
                )
                band_params = {"target_code": tcode, "by": "amt", "width": int(width)}
            else:
                width_default = 0.1
                width = st.number_input(
                    "幅",
                    value=float(band_params_initial.get("width", width_default)),
                    step=width_default / 10 if width_default else 0.01,
                )
                band_params = {"target_code": tcode, "by": "pct", "width": width}
    with c8:
        quick = st.radio(
            "クイック絞り込み",
            ["なし", "Top5", "Top10", "最新YoY上位", "直近6M伸長上位"],
            horizontal=True,
            index=0,
        )
    c9, c10, c11, c12 = st.columns([1.2, 1.5, 1.5, 1.5])
    with c9:
        enable_label_avoid = st.checkbox("ラベル衝突回避", value=True)
    with c10:
        label_gap_px = st.slider("ラベル最小間隔(px)", 8, 24, 12)
    with c11:
        label_max = st.slider("ラベル最大件数", 5, 20, 12)
    with c12:
        alternate_side = st.checkbox("ラベル左右交互配置", value=True)
    c13, c14, c15, c16, c17 = st.columns([1.0, 1.4, 1.2, 1.2, 1.2])
    with c13:
        unit = st.radio("単位", ["円", "千円", "百万円"], horizontal=True, index=1)
    with c14:
        n_win = st.slider(
            "傾きウィンドウ（月）",
            0,
            12,
            6,
            1,
            help="0=自動（系列の全期間で判定）",
        )
    with c15:
        cmp_mode = st.radio("傾き条件", ["以上", "未満"], horizontal=True)
    with c16:
        thr_type = st.radio(
            "しきい値の種類", ["円/月", "%/月", "zスコア"], horizontal=True
        )
    with c17:
        if thr_type == "円/月":
            thr_val = int_input("しきい値", 0)
        else:
            thr_val = st.number_input("しきい値", value=0.0, step=0.01, format="%.2f")
    c18, c19, c20 = st.columns([1.6, 1.2, 1.8])
    with c18:
        sens = st.slider("形状抽出の感度", 0.0, 1.0, 0.5, 0.05)
    with c19:
        z_thr = st.slider("急勾配 zスコア", 0.0, 3.0, 0.0, 0.1)
    with c20:
        shape_pick = st.radio(
            "形状抽出",
            ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"],
            horizontal=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-body">', unsafe_allow_html=True)
    ai_summary_container = st.container()

    if amount_slider_cfg:
        low_scaled, high_scaled = st.slider(
            amount_slider_cfg["label"],
            min_value=amount_slider_cfg["min_value"],
            max_value=amount_slider_cfg["max_value"],
            value=amount_slider_cfg["value"],
            step=amount_slider_cfg["step"],
        )
        low = int(low_scaled * amount_slider_cfg["unit_scale"])
        high = int(high_scaled * amount_slider_cfg["unit_scale"])
        high = min(high, amount_slider_cfg["max_amount"])
        low = min(low, high)
        st.caption(f"選択中: {format_int(low)}円 〜 {format_int(high)}円")
        band_params = {"low_amount": low, "high_amount": high}
    elif band_mode == "金額指定":
        band_params = {"low_amount": low0, "high_amount": high0}

    params = {
        "end_month": end_m,
        "band_mode": band_mode,
        "band_params": band_params,
        "quick": quick,
    }
    st.session_state.compare_params = params

    mode_map = {
        "金額指定": "amount",
        "商品指定(2)": "two_products",
        "パーセンタイル": "percentile",
        "順位帯": "rank",
        "ターゲット近傍": "target_near",
    }
    low, high = resolve_band(snapshot, mode_map[band_mode], band_params)
    codes = filter_products_by_band(snapshot, low, high)

    if quick == "Top5":
        codes = snapshot.nlargest(5, "year_sum")["product_code"].tolist()
    elif quick == "Top10":
        codes = snapshot.nlargest(10, "year_sum")["product_code"].tolist()
    elif quick == "最新YoY上位":
        codes = (
            snapshot.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(10)["product_code"]
            .tolist()
        )
    elif quick == "直近6M伸長上位":
        codes = top_growth_codes(year_df, end_m, window=6, top=10)

    snap = slopes_snapshot(year_df, n=n_win)
    if thr_type == "円/月":
        key, v = "slope_yen", float(thr_val)
    elif thr_type == "%/月":
        key, v = "slope_ratio", float(thr_val)
    else:
        key, v = "slope_z", float(thr_val)
    mask = (snap[key] >= v) if cmp_mode == "以上" else (snap[key] <= v)
    codes_by_slope = set(snap.loc[mask, "product_code"])

    eff_n = n_win if n_win > 0 else 12
    shape_df = shape_flags(
        year_df,
        window=max(6, eff_n * 2),
        alpha_ratio=0.02 * (1.0 - sens),
        amp_ratio=0.06 * (1.0 - sens),
    )
    codes_steep = set(snap.loc[snap["slope_z"].abs() >= z_thr, "product_code"])
    codes_mtn = set(shape_df.loc[shape_df["is_mountain"], "product_code"])
    codes_val = set(shape_df.loc[shape_df["is_valley"], "product_code"])
    shape_map = {
        "（なし）": None,
        "急勾配": codes_steep,
        "山（への字）": codes_mtn,
        "谷（逆への字）": codes_val,
    }
    codes_by_shape = shape_map[shape_pick] or set(snap["product_code"])

    codes_from_band = set(codes)
    target_codes = list(codes_from_band & codes_by_slope & codes_by_shape)

    scale = {"円": 1, "千円": 1_000, "百万円": 1_000_000}[unit]
    snapshot_disp = snapshot.copy()
    snapshot_disp["year_sum_disp"] = snapshot_disp["year_sum"] / scale
    hist_fig = px.histogram(snapshot_disp, x="year_sum_disp")
    hist_fig.update_xaxes(title_text=f"年計（{unit}）")

    df_long, _ = get_yearly_series(year_df, target_codes)
    df_long["month"] = pd.to_datetime(df_long["month"])
    df_long["display_name"] = df_long["product_name"].fillna(df_long["product_code"])

    main_codes = target_codes
    max_lines = 30
    if len(main_codes) > max_lines:
        top_order = (
            snapshot[snapshot["product_code"].isin(main_codes)]
            .sort_values("year_sum", ascending=False)["product_code"]
            .tolist()
        )
        main_codes = top_order[:max_lines]

    df_main = df_long[df_long["product_code"].isin(main_codes)]

    with ai_summary_container:
        ai_on = st.toggle(
            "AIサマリー",
            value=False,
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
        )
        with st.expander("AIサマリー", expanded=ai_on):
            if ai_on and not df_main.empty:
                pos = len(codes_steep)
                mtn = len(codes_mtn & set(main_codes))
                val = len(codes_val & set(main_codes))
                explain = _ai_explain(
                    {
                        "対象SKU数": len(main_codes),
                        "中央値(年計)": float(
                            snapshot_disp.loc[
                                snapshot_disp["product_code"].isin(main_codes),
                                "year_sum_disp",
                            ].median()
                        ),
                        "急勾配数": pos,
                        "山数": mtn,
                        "谷数": val,
                    }
                )
                st.info(f"**AI比較コメント**：{explain}")

    tb_common = dict(
        period=period,
        node_mode=node_mode,
        hover_mode=hover_mode,
        op_mode=op_mode,
        peak_on=peak_on,
        unit=unit,
        enable_avoid=enable_label_avoid,
        gap_px=label_gap_px,
        max_labels=label_max,
        alt_side=alternate_side,
        slope_conf=None,
        forecast_method="なし",
        forecast_window=12,
        forecast_horizon=6,
        forecast_k=2.0,
        forecast_robust=False,
        anomaly="OFF",
    )
    fig = build_chart_card(
        df_main,
        selected_codes=None,
        multi_mode=True,
        tb=tb_common,
        band_range=(low, high),
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

    st.caption(
        "凡例クリックで表示切替、ダブルクリックで単独表示。ドラッグでズーム/パン、右上メニューからPNG/CSV取得可。"
    )
    st.markdown(
        """
傾き（円/月）：直近 n ヶ月の回帰直線の傾き。+は上昇、−は下降。

%/月：傾き÷平均年計。規模によらず比較可能。

zスコア：全SKUの傾き分布に対する標準化。|z|≥1.5で急勾配の目安。

山/谷：前半と後半の平均変化率の符号が**＋→−（山）／−→＋（谷）かつ振幅が十分**。
"""
    )

    snap_export = snapshot[snapshot["product_code"].isin(main_codes)].copy()
    snap_export[f"year_sum_{unit}"] = snap_export["year_sum"] / scale
    snap_export = snap_export.drop(columns=["year_sum"])
    st.download_button(
        "CSVエクスポート",
        data=snap_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"band_snapshot_{end_m}.csv",
        mime="text/csv",
    )
    try:
        png_bytes = fig.to_image(format="png")
        st.download_button(
            "PNGエクスポート",
            data=png_bytes,
            file_name=f"band_overlay_{end_m}.png",
            mime="image/png",
        )
    except Exception:
        pass

    with st.expander("分布（オプション）", expanded=False):
        hist_fig = apply_elegant_theme(
            hist_fig, theme=st.session_state.get("ui_theme", "dark")
        )
        render_plotly_with_spinner(hist_fig, config=PLOTLY_CONFIG)

    # ---- Small Multiples ----
    df_nodes = df_main.iloc[0:0].copy()
    HALO = "#ffffff" if st.get_option("theme.base") == "dark" else "#222222"
    SZ = 6
    dtick = "M1"
    drag = {"ズーム": "zoom", "パン": "pan", "選択": "select"}[op_mode]

    st.subheader("スモールマルチプル")
    share_y = st.checkbox("Y軸共有", value=False)
    show_keynode_labels = st.checkbox("キーノードラベル表示", value=False)
    per_page = st.radio("1ページ表示枚数", [8, 12], horizontal=True, index=0)
    total_pages = max(1, math.ceil(len(main_codes) / per_page))
    page_idx = st.number_input("ページ", min_value=1, max_value=total_pages, value=1)
    start = (page_idx - 1) * per_page
    page_codes = main_codes[start : start + per_page]
    col_count = 4
    cols = st.columns(col_count)
    ymax = (
        df_long[df_long["product_code"].isin(main_codes)]["year_sum"].max()
        / UNIT_MAP[unit]
        if share_y
        else None
    )
    for i, code in enumerate(page_codes):
        g = df_long[df_long["product_code"] == code]
        disp = g["display_name"].iloc[0] if not g.empty else code
        palette = fig.layout.colorway or px.colors.qualitative.Safe
        fig_s = px.line(
            g,
            x="month",
            y="year_sum",
            color_discrete_sequence=[palette[i % len(palette)]],
            custom_data=["display_name"],
        )
        fig_s.update_traces(
            mode="lines",
            line=dict(width=1.5),
            opacity=0.8,
            showlegend=False,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
        )
        fig_s.update_xaxes(tickformat="%Y-%m", dtick=dtick, title_text="月（YYYY-MM）")
        fig_s.update_yaxes(
            tickformat="~,d",
            range=[0, ymax] if ymax else None,
            title_text=f"売上 年計（{unit}）",
        )
        fig_s.update_layout(font=dict(family="Noto Sans JP, Meiryo, Arial", size=12))
        fig_s.update_layout(
            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.92)", font=dict(color="#fff", size=12)
            )
        )
        fig_s.update_layout(dragmode=drag)
        if hover_mode == "個別":
            fig_s.update_layout(hovermode="closest")
        else:
            fig_s.update_layout(hovermode="x unified", hoverlabel=dict(align="left"))
        last_val = (
            g.sort_values("month")["year_sum"].iloc[-1] / UNIT_MAP[unit]
            if not g.empty
            else np.nan
        )
        with cols[i % col_count]:
            st.metric(
                disp, f"{last_val:,.0f} {unit}" if not np.isnan(last_val) else "—"
            )
            fig_s = apply_elegant_theme(
                fig_s, theme=st.session_state.get("ui_theme", "dark")
            )
            fig_s.update_layout(height=225)
            render_plotly_with_spinner(fig_s, config=PLOTLY_CONFIG)

    # 5) SKU詳細
elif page == "SKU詳細":
    require_data()
    section_header("SKU 詳細", "個別SKUのトレンドとメモを一元管理。", icon="🗂️")
    end_m = sidebar_state.get("detail_end_month") or latest_month
    prods = (
        st.session_state.data_year[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    mode = st.radio("表示モード", ["単品", "複数比較"], horizontal=True)
    tb = toolbar_sku_detail(multi_mode=(mode == "複数比較"))
    df_year = st.session_state.data_year.copy()
    df_year["display_name"] = df_year["product_name"].fillna(df_year["product_code"])

    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    chart_rendered = False
    modal_codes: List[str] | None = None
    modal_is_multi = False

    if mode == "単品":
        prod_label = st.selectbox(
            "SKU選択", options=prods["product_code"] + " | " + prods["product_name"]
        )
        code = prod_label.split(" | ")[0]
        build_chart_card(
            df_year,
            selected_codes=[code],
            multi_mode=False,
            tb=tb,
            height=600,
        )
        chart_rendered = True
        modal_codes = [code]
        modal_is_multi = False

        g_y = df_year[df_year["product_code"] == code].sort_values("month")
        row = g_y[g_y["month"] == end_m]
        if not row.empty:
            rr = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "年計", f"{int(rr['year_sum']) if not pd.isna(rr['year_sum']) else '—'}"
            )
            c2.metric(
                "YoY", f"{rr['yoy']*100:.1f} %" if not pd.isna(rr["yoy"]) else "—"
            )
            c3.metric("Δ", f"{int(rr['delta'])}" if not pd.isna(rr["delta"]) else "—")

        with st.expander("AIサマリー", expanded=ai_on):
            if ai_on and not row.empty:
                st.info(
                    _ai_explain(
                        {
                            "年計": (
                                float(rr["year_sum"])
                                if not pd.isna(rr["year_sum"])
                                else 0.0
                            ),
                            "YoY": float(rr["yoy"]) if not pd.isna(rr["yoy"]) else 0.0,
                            "Δ": float(rr["delta"]) if not pd.isna(rr["delta"]) else 0.0,
                        }
                    )
                )

        st.subheader("メモ / タグ")
        note = st.text_area(
            "メモ（保存で保持）", value=st.session_state.notes.get(code, ""), height=100
        )
        tags_str = st.text_input(
            "タグ（カンマ区切り）", value=",".join(st.session_state.tags.get(code, []))
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("保存"):
            st.session_state.notes[code] = note
            st.session_state.tags[code] = [
                t.strip() for t in tags_str.split(",") if t.strip()
            ]
            st.success("保存しました")
        if c2.button("CSVでエクスポート"):
            meta = pd.DataFrame(
                [
                    {
                        "product_code": code,
                        "note": st.session_state.notes.get(code, ""),
                        "tags": ",".join(st.session_state.tags.get(code, [])),
                    }
                ]
            )
            st.download_button(
                "ダウンロード",
                data=meta.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"notes_{code}.csv",
                mime="text/csv",
            )
    else:
        opts = (prods["product_code"] + " | " + prods["product_name"]).tolist()
        sel = st.multiselect("SKU選択（最大60件）", options=opts, max_selections=60)
        codes = [s.split(" | ")[0] for s in sel]
        if codes or (tb.get("slope_conf") and tb["slope_conf"].get("quick") != "なし"):
            build_chart_card(
                df_year,
                selected_codes=codes,
                multi_mode=True,
                tb=tb,
                height=600,
            )
            chart_rendered = True
            modal_codes = codes
            modal_is_multi = True
            snap = latest_yearsum_snapshot(df_year, end_m)
            if codes:
                snap = snap[snap["product_code"].isin(codes)]
            with st.expander("AIサマリー", expanded=ai_on):
                if ai_on and not snap.empty:
                    st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]]))
            st.dataframe(
                snap[["product_code", "product_name", "year_sum", "yoy", "delta"]],
                use_container_width=True,
            )
            st.download_button(
                "CSVダウンロード",
                data=snap.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"sku_multi_{end_m}.csv",
                mime="text/csv",
            )
        else:
            st.info("SKUを選択してください。")

    if tb.get("expand_mode") and chart_rendered:
        with st.modal("グラフ拡大モード", key="sku_expand_modal"):
            st.caption("操作パネルは拡大表示中も利用できます。")
            tb_modal = toolbar_sku_detail(
                multi_mode=modal_is_multi,
                key_prefix="sku_modal",
                include_expand_toggle=False,
            )
            build_chart_card(
                df_year,
                selected_codes=modal_codes,
                multi_mode=modal_is_multi,
                tb=tb_modal,
                height=tb_modal.get("chart_height", 760),
            )
            if st.button("閉じる", key="close_expand_modal"):
                st.session_state.setdefault("ui", {})["expand_mode"] = False
                st.session_state["sku_expand_mode"] = False
                st.experimental_rerun()

# 5) 異常検知
elif page == "異常検知":
    require_data()
    section_header("異常検知", "回帰残差ベースで異常ポイントを抽出します。", icon="🚨")
    year_df = st.session_state.data_year.copy()
    unit = st.session_state.settings.get("currency_unit", "円")
    scale = UNIT_MAP.get(unit, 1)

    col_a, col_b = st.columns([1.1, 1.1])
    with col_a:
        window = st.slider("学習窓幅（月）", 6, 18, st.session_state.get("anomaly_window", 12), key="anomaly_window")
    with col_b:
        score_method = st.radio("スコア基準", ["zスコア", "MADスコア"], horizontal=True, key="anomaly_score_method")

    if score_method == "zスコア":
        thr_key = "anomaly_thr_z"
        threshold = st.slider(
            "異常判定しきい値",
            2.0,
            5.0,
            value=float(st.session_state.get(thr_key, 3.0)),
            step=0.1,
            key=thr_key,
        )
        robust = False
    else:
        thr_key = "anomaly_thr_mad"
        threshold = st.slider(
            "異常判定しきい値",
            2.5,
            6.0,
            value=float(st.session_state.get(thr_key, 3.5)),
            step=0.1,
            key=thr_key,
        )
        robust = True

    prod_opts = (
        year_df[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    prod_opts["label"] = (
        prod_opts["product_code"]
        + " | "
        + prod_opts["product_name"].fillna(prod_opts["product_code"])
    )
    selected_labels = st.multiselect(
        "対象SKU（未選択=全件）",
        options=prod_opts["label"].tolist(),
        key="anomaly_filter_codes",
    )
    selected_codes = [lab.split(" | ")[0] for lab in selected_labels]

    records: List[pd.DataFrame] = []
    for code, g in year_df.groupby("product_code"):
        if selected_codes and code not in selected_codes:
            continue
        s = g.sort_values("month").set_index("month")["year_sum"]
        res = detect_linear_anomalies(
            s,
            window=int(window),
            threshold=float(threshold),
            robust=robust,
        )
        if res.empty:
            continue
        res["product_code"] = code
        res["product_name"] = g["product_name"].iloc[0]
        res = res.merge(
            g[["month", "year_sum", "yoy", "delta"]],
            on="month",
            how="left",
        )
        res["score_abs"] = res["score"].abs()
        records.append(res)

    if not records:
        st.success("異常値は検出されませんでした。窓幅やしきい値を調整してください。")
    else:
        anomalies = pd.concat(records, ignore_index=True)
        anomalies = anomalies.sort_values("score_abs", ascending=False)
        anomalies["year_sum_disp"] = anomalies["year_sum"] / scale
        anomalies["delta_disp"] = anomalies["delta"] / scale
        total_count = len(anomalies)
        sku_count = anomalies["product_code"].nunique()
        pos_cnt = int((anomalies["score"] > 0).sum())
        neg_cnt = int((anomalies["score"] < 0).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("異常件数", f"{total_count:,}")
        m2.metric("対象SKU", f"{sku_count:,}")
        m3.metric("上振れ/下振れ", f"{pos_cnt:,} / {neg_cnt:,}")

        max_top = min(200, total_count)
        top_default = min(50, max_top)
        top_n = int(
            st.slider(
                "表示件数",
                min_value=1,
                max_value=max_top,
                value=top_default,
                key="anomaly_view_top",
            )
        )
        view = anomalies.head(top_n).copy()
        view_table = view[
            [
                "product_code",
                "product_name",
                "month",
                "year_sum_disp",
                "yoy",
                "delta_disp",
                "score",
            ]
        ].rename(
            columns={
                "product_code": "商品コード",
                "product_name": "商品名",
                "month": "月",
                "year_sum_disp": f"年計({unit})",
                "yoy": "YoY",
                "delta_disp": f"Δ({unit})",
                "score": "スコア",
            }
        )
        st.dataframe(view_table, use_container_width=True)
        st.caption("値は指定した単位換算、スコアはローカル回帰残差の標準化値です。")
        st.download_button(
            "CSVダウンロード",
            data=view_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"anomalies_{score_method}_{threshold:.1f}.csv",
            mime="text/csv",
        )

        anomaly_ai_on = st.toggle(
            "AI異常サマリー", value=False, key="anomaly_ai_toggle"
        )
        with st.expander("AI異常サマリー", expanded=anomaly_ai_on):
            if anomaly_ai_on and not view.empty:
                ai_df = view[
                    ["product_name", "month", "score", "year_sum", "yoy", "delta"]
                ].fillna(0)
                st.info(_ai_anomaly_report(ai_df))

        option_labels = [
            f"{row['product_code']}｜{row['product_name'] or row['product_code']}｜{row['month']}"
            for _, row in view.iterrows()
        ]
        if option_labels:
            sel_label = st.selectbox("詳細チャート", options=option_labels, key="anomaly_detail_select")
            code_sel, name_sel, month_sel = sel_label.split("｜")
            g = year_df[year_df["product_code"] == code_sel].sort_values("month").copy()
            g["year_sum_disp"] = g["year_sum"] / scale
            fig_anom = px.line(
                g,
                x="month",
                y="year_sum_disp",
                markers=True,
                title=f"{name_sel} 年計推移",
            )
            fig_anom.update_yaxes(title_text=f"年計（{unit}）", tickformat="~,d")
            fig_anom.update_traces(hovertemplate="月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<extra></extra>")

            code_anoms = anomalies[anomalies["product_code"] == code_sel]
            if not code_anoms.empty:
                fig_anom.add_scatter(
                    x=code_anoms["month"],
                    y=code_anoms["year_sum"] / scale,
                    mode="markers",
                    name="異常値",
                    marker=dict(color="#d94c53", size=10, symbol="triangle-up"),
                    hovertemplate="異常月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<br>スコア：%{customdata[0]:.2f}<extra></extra>",
                    customdata=np.stack([code_anoms["score"]], axis=-1),
                    showlegend=False,
                )
            target = code_anoms[code_anoms["month"] == month_sel]
            if not target.empty:
                tgt = target.iloc[0]
                fig_anom.add_annotation(
                    x=month_sel,
                    y=tgt["year_sum"] / scale,
                    text=f"スコア {tgt['score']:.2f}",
                    showarrow=True,
                    arrowcolor="#d94c53",
                    arrowhead=2,
                )
                yoy_txt = (
                    f"{tgt['yoy'] * 100:.1f}%" if tgt.get("yoy") is not None and not pd.isna(tgt.get("yoy")) else "—"
                )
                delta_txt = format_amount(tgt.get("delta"), unit)
                st.info(
                    f"{name_sel} {month_sel} の年計は {tgt['year_sum_disp']:.0f} {unit}、YoY {yoy_txt}、Δ {delta_txt}。"
                    f" 異常スコアは {tgt['score']:.2f} です。"
                )
            fig_anom = apply_elegant_theme(
                fig_anom, theme=st.session_state.get("ui_theme", "dark")
            )
            render_plotly_with_spinner(fig_anom, config=PLOTLY_CONFIG)

# 6) 相関分析
elif page == "相関分析":
    require_data()
    section_header("相関分析", "指標間の関係性からインサイトを発掘。", icon="🧭")
    end_m = sidebar_state.get("corr_end_month") or latest_month
    snapshot = latest_yearsum_snapshot(st.session_state.data_year, end_m)

    metric_opts = [
        "year_sum",
        "yoy",
        "delta",
        "slope_beta",
        "slope6m",
        "std6m",
        "hhi_share",
    ]
    analysis_mode = st.radio(
        "分析対象",
        ["指標間", "SKU間"],
        horizontal=True,
    )
    method = st.radio(
        "相関の種類",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda x: "Pearson" if x == "pearson" else "Spearman",
    )
    r_thr = st.slider("相関 r 閾値（|r|≥）", 0.0, 1.0, 0.0, 0.05)

    if analysis_mode == "指標間":
        metrics = st.multiselect(
            "指標",
            [m for m in metric_opts if m in snapshot.columns],
            default=[
                m
                for m in ["year_sum", "yoy", "delta", "slope_beta"]
                if m in snapshot.columns
            ],
        )
        winsor_pct = st.slider("外れ値丸め(%)", 0.0, 5.0, 1.0)
        log_enable = st.checkbox("ログ変換", value=False)
        ai_on = st.toggle(
            "AIサマリー",
            value=False,
            key="corr_ai_metric",
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
        )

        if metrics:
            df_plot = snapshot.copy()
            df_plot = winsorize_frame(df_plot, metrics, p=winsor_pct / 100)
            df_plot = maybe_log1p(df_plot, metrics, log_enable)
            tbl = corr_table(df_plot, metrics, method=method)
            tbl = tbl[abs(tbl["r"]) >= r_thr]

            st.subheader("相関の要点")
            for line in narrate_top_insights(tbl, NAME_MAP):
                st.write("・", line)
            sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
            weak_cnt = int((tbl["r"].abs() < 0.2).sum())
            st.write(f"統計的に有意な相関: {sig_cnt} 組")
            st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")

            with st.expander("AIサマリー", expanded=ai_on):
                if ai_on and not tbl.empty:
                    r_mean = float(tbl["r"].abs().mean())
                    st.info(
                        _ai_explain(
                            {
                                "有意本数": int((tbl["sig"] == "有意(95%)").sum()),
                                "平均|r|": r_mean,
                            }
                        )
                    )

            st.subheader("相関ヒートマップ")
            st.caption("右上=強い正、左下=強い負、白=関係薄")
            corr = df_plot[metrics].corr(method=method)
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True
            )
            fig_corr = apply_elegant_theme(
                fig_corr, theme=st.session_state.get("ui_theme", "dark")
            )
            render_plotly_with_spinner(fig_corr, config=PLOTLY_CONFIG)

            st.subheader("ペア・エクスプローラ")
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("指標X", metrics, index=0)
            with c2:
                y_col = st.selectbox(
                    "指標Y", metrics, index=1 if len(metrics) > 1 else 0
                )
            df_xy = df_plot[[x_col, y_col, "product_name", "product_code"]].dropna()
            if not df_xy.empty:
                m, b, r2 = fit_line(df_xy[x_col], df_xy[y_col])
                r = df_xy[x_col].corr(df_xy[y_col], method=method)
                lo, hi = fisher_ci(r, len(df_xy))
                fig_sc = px.scatter(
                    df_xy, x=x_col, y=y_col, hover_data=["product_code", "product_name"]
                )
                xs = np.linspace(df_xy[x_col].min(), df_xy[x_col].max(), 100)
                fig_sc.add_trace(
                    go.Scatter(x=xs, y=m * xs + b, mode="lines", name="回帰")
                )
                fig_sc.add_annotation(
                    x=0.99,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="bottom",
                    text=f"r={r:.2f} (95%CI [{lo:.2f},{hi:.2f}])<br>R²={r2:.2f}",
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.6)",
                )
                resid = np.abs(df_xy[y_col] - (m * df_xy[x_col] + b))
                outliers = df_xy.loc[resid.nlargest(3).index]
                for _, row in outliers.iterrows():
                    label = row["product_name"] or row["product_code"]
                    fig_sc.add_annotation(
                        x=row[x_col],
                        y=row[y_col],
                        text=label,
                        showarrow=True,
                        arrowhead=1,
                    )
                fig_sc = apply_elegant_theme(
                    fig_sc, theme=st.session_state.get("ui_theme", "dark")
                )
                render_plotly_with_spinner(fig_sc, config=PLOTLY_CONFIG)
                st.caption("rは -1〜+1。0は関連が薄い。CIに0を含まなければ有意。")
                st.caption("散布図の点が右上・左下に伸びれば正、右下・左上なら負。")
        else:
            st.info("指標を選択してください。")
    else:
        df_year = st.session_state.data_year.copy()
        series_metric_opts = [m for m in metric_opts if m in df_year.columns]
        if not series_metric_opts:
            st.info("SKU間相関に利用できる指標がありません。")
        else:
            sku_metric = st.selectbox(
                "対象指標",
                series_metric_opts,
                format_func=lambda x: NAME_MAP.get(x, x),
            )
            months_all = sorted(df_year["month"].unique())
            if not months_all:
                st.info("データが不足しています。")
            else:
                if end_m in months_all:
                    end_idx = months_all.index(end_m)
                else:
                    end_idx = len(months_all) - 1
                if end_idx < 0:
                    st.info("対象期間のデータがありません。")
                else:
                    max_period = end_idx + 1
                    if max_period < 2:
                        st.info("対象期間のデータが不足しています。")
                    else:
                        slider_min = 2
                        slider_max = max_period
                        default_period = max(slider_min, min(12, slider_max))
                        period = int(
                            st.slider(
                                "対象期間（月数）",
                                min_value=slider_min,
                                max_value=slider_max,
                                value=default_period,
                            )
                        )
                        start_idx = max(0, end_idx - period + 1)
                        months_window = months_all[start_idx : end_idx + 1]
                        df_window = df_year[df_year["month"].isin(months_window)]
                        pivot = (
                            df_window.pivot(
                                index="month", columns="product_code", values=sku_metric
                            ).sort_index()
                        )
                        pivot = pivot.dropna(how="all")
                        if pivot.empty:
                            st.info("選択した期間に利用できるデータがありません。")
                        else:
                            top_candidates = [
                                c for c in snapshot["product_code"] if c in pivot.columns
                            ]
                            if len(top_candidates) < 2:
                                st.info("対象SKUが不足しています。")
                            else:
                                top_max = min(60, len(top_candidates))
                                top_default = max(2, min(10, top_max))
                                top_n = int(
                                    st.slider(
                                        "対象SKU数（上位）",
                                        min_value=2,
                                        max_value=top_max,
                                        value=top_default,
                                    )
                                )
                                selected_codes = top_candidates[:top_n]
                                sku_pivot = pivot[selected_codes].dropna(
                                    axis=1, how="all"
                                )
                                available_codes = sku_pivot.columns.tolist()
                                min_periods = 3
                                valid_codes = [
                                    code
                                    for code in available_codes
                                    if sku_pivot[code].count() >= min_periods
                                ]
                                if len(valid_codes) < 2:
                                    st.info(
                                        "有効なSKUが2件未満です。期間やSKU数を調整してください。"
                                    )
                                else:
                                    sku_pivot = sku_pivot[valid_codes]
                                    months_used = sku_pivot.index.tolist()
                                    code_to_name = (
                                        df_year[["product_code", "product_name"]]
                                        .drop_duplicates()
                                        .set_index("product_code")["product_name"]
                                        .to_dict()
                                    )
                                    display_map = {
                                        code: f"{code}｜{code_to_name.get(code, code) or code}"
                                        for code in valid_codes
                                    }
                                    ai_on = st.toggle(
                                        "AIサマリー",
                                        value=False,
                                        key="corr_ai_sku",
                                        help="要約・コメント・自動説明を表示（オンデマンド計算）",
                                    )
                                    tbl_raw = corr_table(
                                        sku_pivot,
                                        valid_codes,
                                        method=method,
                                        pairwise=True,
                                        min_periods=min_periods,
                                    )
                                    tbl = tbl_raw.dropna(subset=["r"])
                                    tbl = tbl[abs(tbl["r"]) >= r_thr]

                                    st.subheader("相関の要点")
                                    if months_used:
                                        st.caption(
                                            f"対象期間: {months_used[0]}〜{months_used[-1]}（{len(months_used)}ヶ月）"
                                        )
                                    st.caption(
                                        "対象SKU: "
                                        + "、".join(display_map[code] for code in valid_codes)
                                    )
                                    for line in narrate_top_insights(tbl, display_map):
                                        st.write("・", line)
                                    sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
                                    weak_cnt = int((tbl["r"].abs() < 0.2).sum())
                                    st.write(f"統計的に有意な相関: {sig_cnt} 組")
                                    st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")
                                    if tbl.empty:
                                        st.info(
                                            "条件に合致するSKU間相関は見つかりませんでした。"
                                        )

                                    with st.expander("AIサマリー", expanded=ai_on):
                                        if ai_on and not tbl.empty:
                                            r_mean = float(tbl["r"].abs().mean())
                                            st.info(
                                                _ai_explain(
                                                    {
                                                        "有意本数": int(
                                                            (tbl["sig"] == "有意(95%)").sum()
                                                        ),
                                                        "平均|r|": r_mean,
                                                    }
                                                )
                                            )

                                    st.subheader("相関ヒートマップ")
                                    st.caption(
                                        "セルは対象期間におけるSKU同士の相関係数を示します。"
                                    )
                                    heatmap = sku_pivot.rename(columns=display_map)
                                    corr = heatmap.corr(
                                        method=method, min_periods=min_periods
                                    )
                                    fig_corr = px.imshow(
                                        corr,
                                        color_continuous_scale="RdBu_r",
                                        zmin=-1,
                                        zmax=1,
                                        text_auto=True,
                                    )
                                    fig_corr = apply_elegant_theme(
                                        fig_corr, theme=st.session_state.get("ui_theme", "dark")
                                    )
                                    render_plotly_with_spinner(
                                        fig_corr, config=PLOTLY_CONFIG
                                    )

                                    st.subheader("SKUペア・エクスプローラ")
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        x_code = st.selectbox(
                                            "SKU X",
                                            valid_codes,
                                            format_func=lambda c: display_map.get(c, c),
                                        )
                                    with c2:
                                        y_default = 1 if len(valid_codes) > 1 else 0
                                        y_code = st.selectbox(
                                            "SKU Y",
                                            valid_codes,
                                            index=y_default,
                                            format_func=lambda c: display_map.get(c, c),
                                        )
                                    df_xy = (
                                        sku_pivot[[x_code, y_code]]
                                        .dropna()
                                        .reset_index()
                                    )
                                    if len(df_xy) >= 2:
                                        x_label = display_map.get(x_code, x_code)
                                        y_label = display_map.get(y_code, y_code)
                                        df_xy = df_xy.rename(
                                            columns={
                                                "month": "月",
                                                x_code: x_label,
                                                y_code: y_label,
                                            }
                                        )
                                        m, b, r2 = fit_line(
                                            df_xy[x_label], df_xy[y_label]
                                        )
                                        r = df_xy[x_label].corr(
                                            df_xy[y_label], method=method
                                        )
                                        lo, hi = fisher_ci(r, len(df_xy))
                                        fig_sc = px.scatter(
                                            df_xy,
                                            x=x_label,
                                            y=y_label,
                                            hover_data=["月"],
                                        )
                                        xs = np.linspace(
                                            df_xy[x_label].min(), df_xy[x_label].max(), 100
                                        )
                                        fig_sc.add_trace(
                                            go.Scatter(
                                                x=xs, y=m * xs + b, mode="lines", name="回帰"
                                            )
                                        )
                                        fig_sc.add_annotation(
                                            x=0.99,
                                            y=0.01,
                                            xref="paper",
                                            yref="paper",
                                            xanchor="right",
                                            yanchor="bottom",
                                            text=f"r={r:.2f} (95%CI [{lo:.2f},{hi:.2f}])<br>R²={r2:.2f}｜n={len(df_xy)}",
                                            showarrow=False,
                                            align="right",
                                            bgcolor="rgba(255,255,255,0.6)",
                                        )
                                        resid = np.abs(
                                            df_xy[y_label] - (m * df_xy[x_label] + b)
                                        )
                                        outliers = df_xy.loc[
                                            resid.nlargest(min(3, len(resid))).index
                                        ]
                                        for _, row in outliers.iterrows():
                                            fig_sc.add_annotation(
                                                x=row[x_label],
                                                y=row[y_label],
                                                text=row["月"],
                                                showarrow=True,
                                                arrowhead=1,
                                            )
                                        fig_sc = apply_elegant_theme(
                                            fig_sc,
                                            theme=st.session_state.get("ui_theme", "dark"),
                                        )
                                        render_plotly_with_spinner(
                                            fig_sc, config=PLOTLY_CONFIG
                                        )
                                        st.caption(
                                            "各点は対象期間の月次値。右上（左下）に伸びれば同時に増加（減少）。"
                                        )
                                    else:
                                        st.info(
                                            "共通する月のデータが不足しています。期間やSKU数を調整してください。"
                                        )

    with st.expander("相関の読み方"):
        st.write("正の相関：片方が大きいほどもう片方も大きい")
        st.write("負の相関：片方が大きいほどもう片方は小さい")
        st.write(
            "|r|<0.2は弱い、0.2-0.5はややあり、0.5-0.8は中~強、>0.8は非常に強い（目安）"
        )

# 7) 併買カテゴリ
elif page == "併買カテゴリ":
    render_correlation_category_module(plot_config=PLOTLY_CONFIG)

# 8) アラート
elif page == "アラート":
    require_data()
    section_header("アラート", "閾値に該当したリスクSKUを自動抽出。", icon="⚠️")
    end_m = sidebar_state.get("alert_end_month") or latest_month
    s = st.session_state.settings
    alerts = build_alerts(
        st.session_state.data_year,
        end_month=end_m,
        yoy_threshold=s["yoy_threshold"],
        delta_threshold=s["delta_threshold"],
        slope_threshold=s["slope_threshold"],
    )
    if alerts.empty:
        st.success("閾値に該当するアラートはありません。")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button(
            "CSVダウンロード",
            data=alerts.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"alerts_{end_m}.csv",
            mime="text/csv",
        )

# 9) 設定
elif page == "設定":
    section_header("設定", "年計計算条件や閾値を調整します。", icon="⚙️")
    s = st.session_state.settings
    c1, c2, c3 = st.columns(3)
    with c1:
        s["window"] = st.number_input(
            "年計ウィンドウ（月）",
            min_value=3,
            max_value=24,
            value=int(s["window"]),
            step=1,
        )
        s["last_n"] = st.number_input(
            "傾き算出の対象点数",
            min_value=3,
            max_value=36,
            value=int(s["last_n"]),
            step=1,
        )
    with c2:
        s["yoy_threshold"] = st.number_input(
            "YoY 閾値（<=）", value=float(s["yoy_threshold"]), step=0.01, format="%.2f"
        )
        s["delta_threshold"] = int_input("Δ 閾値（<= 円）", int(s["delta_threshold"]))
    with c3:
        s["slope_threshold"] = st.number_input(
            "傾き 閾値（<=）",
            value=float(s["slope_threshold"]),
            step=0.1,
            format="%.2f",
        )
        s["currency_unit"] = st.selectbox(
            "通貨単位表記",
            options=["円", "千円", "百万円"],
            index=["円", "千円", "百万円"].index(s["currency_unit"]),
        )

    st.caption("※ 設定変更後は再計算が必要です。")
    if st.button("年計の再計算を実行", type="primary"):
        if st.session_state.data_monthly is None:
            st.warning("先にデータを取り込んでください。")
        else:
            long_df = st.session_state.data_monthly
            year_df = compute_year_rolling(
                long_df, window=s["window"], policy=s["missing_policy"]
            )
            year_df = compute_slopes(year_df, last_n=s["last_n"])
            st.session_state.data_year = year_df
            st.success("再計算が完了しました。")

# 10) 保存ビュー
elif page == "保存ビュー":
    section_header("保存ビュー", "設定や比較条件をブックマーク。", icon="🔖")
    s = st.session_state.settings
    cparams = st.session_state.compare_params
    st.write("現在の設定・選択（閾値、ウィンドウ、単位など）を名前を付けて保存します。")

    name = st.text_input("ビュー名")
    if st.button("保存"):
        if not name:
            st.warning("ビュー名を入力してください。")
        else:
            st.session_state.saved_views[name] = {
                "settings": dict(s),
                "compare": dict(cparams),
            }
            st.success(f"ビュー「{name}」を保存しました。")

    st.subheader("保存済みビュー")
    if not st.session_state.saved_views:
        st.info("保存済みビューはありません。")
    else:
        for k, v in st.session_state.saved_views.items():
            st.write(f"**{k}**: {json.dumps(v, ensure_ascii=False)}")
            if st.button(f"適用: {k}"):
                st.session_state.settings.update(v.get("settings", {}))
                st.session_state.compare_params = v.get("compare", {})
                st.session_state.compare_results = None
                st.success(f"ビュー「{k}」を適用しました。")

current_tour_step = get_current_tour_step()
apply_tour_highlight(current_tour_step)
