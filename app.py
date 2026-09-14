"""
Dataset Quality Auditor — Dual Theme (Dark & Light Mode) Professional Edition
An easy-to-use tabular data health scanner, auto-remediation tool, and ML benchmarker.
"""

import os
import io
import json
import html
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import importlib

from core.engine import AuditEngine
from core.cleaner import DatasetCleaner
from core.report_generator import ReportGenerator
from core.ml_benchmark import MLBenchmarkEngine
from core.drift_detector import DriftDetector
from core.rules_engine import RulesEngine, RuleDefinition
import core.data_doctor
importlib.reload(core.data_doctor)
from core.data_doctor import DataDoctor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dataset Quality Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "🌙 Dark Mode"
if "df" not in st.session_state:
    st.session_state.df = None
if "report" not in st.session_state:
    st.session_state.report = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaned_report" not in st.session_state:
    st.session_state.cleaned_report = None
if "clean_stats" not in st.session_state:
    st.session_state.clean_stats = None
if "benchmark_result" not in st.session_state:
    st.session_state.benchmark_result = None
if "drift_report" not in st.session_state:
    st.session_state.drift_report = None
if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = []
if "rules_result" not in st.session_state:
    st.session_state.rules_result = None

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------
# Sidebar Theme Controls & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Dataset Auditor")
    st.caption("Simple, Fast & Accurate Data Quality Scanner")
    st.divider()

    # Theme Switcher (Dark Mode / Light Mode)
    st.markdown("##### 🎨 Display Theme:")
    selected_theme = st.radio(
        "Select Theme Mode:",
        options=["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.theme_mode == "🌙 Dark Mode" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

    is_dark = (st.session_state.theme_mode == "🌙 Dark Mode")

    st.divider()
    st.markdown("##### 💡 Beginner Score Guide:")
    st.markdown("""
    - **90–100 (Grade A)**: 🟢 Production ready!
    - **75–89 (Grade B)**: 🔵 Good, minor fixes recommended.
    - **60–74 (Grade C)**: 🟡 Needs auto-cleaning.
    - **0–59 (Grade D/F)**: 🔴 Dirty data, cleaning required.
    """)

    st.divider()
    if st.session_state.report is not None:
        if st.button("🔄 Audit a New Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.report = None
            st.session_state.cleaned_df = None
            st.session_state.cleaned_report = None
            st.session_state.clean_stats = None
            st.session_state.benchmark_result = None
            st.rerun()

    st.caption("Pure Python • 100% Offline & Private")


# ---------------------------------------------------------
# Dynamic CSS Styling (Dark & Light Mode Aware)
# ---------------------------------------------------------
if is_dark:
    css_theme = """
    <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: radial-gradient(circle at top right, #0F172A 0%, #0B1120 50%, #030712 100%) !important;
            color: #F8FAFC !important;
        }

        code, pre, .font-mono, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .brand-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(20, 184, 166, 0.35);
            border-radius: 1rem;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }

        .brand-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #FFFFFF, #E2E8F0, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .tip-box {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 0.9rem 1.15rem;
            margin-bottom: 0.85rem;
            font-size: 0.875rem;
            color: #CBD5E1;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .tip-box:hover {
            border-color: #14B8A6;
            transform: translateY(-2px);
        }

        .kpi-card {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(51, 65, 85, 0.8);
            border-radius: 0.85rem;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .score-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .score-grade-a { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
        .score-grade-b { background: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid #38BDF8; }
        .score-grade-c { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
        .score-grade-d { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }

        .alert-card-warning {
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #F59E0B;
            border-radius: 0 0.5rem 0.5rem 0;
            padding: 0.65rem 1rem;
            margin-bottom: 0.45rem;
            color: #FEF3C7;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .alert-card-success {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
            border-radius: 0 0.5rem 0.5rem 0;
            padding: 0.65rem 1rem;
            margin-bottom: 0.45rem;
            color: #D1FAE5;
            font-size: 0.875rem;
            font-weight: 500;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
            color: #003731 !important;
            font-weight: 700 !important;
            border: 1px solid #2DD4BF !important;
            border-radius: 0.5rem !important;
            box-shadow: 0 0 16px rgba(20, 184, 166, 0.3) !important;
        }

        div.stButton > button:not([kind="primary"]) {
            background: rgba(15, 23, 42, 0.8) !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 0.5rem !important;
        }
        div.stButton > button:not([kind="primary"]):hover {
            border-color: #14B8A6 !important;
            color: #2DD4BF !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2DD4BF !important;
            border-bottom: 2px solid #14B8A6 !important;
        }
    </style>
    """
else:
    css_theme = """
    <style>
        html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%) !important;
            color: #0F172A !important;
        }

        code, pre, .font-mono, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        .brand-hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 100%);
            border: 1px solid rgba(20, 184, 166, 0.4);
            border-radius: 1rem;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        }

        .brand-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #0F172A;
            margin: 0;
        }

        .tip-box {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 0.75rem;
            padding: 0.9rem 1.15rem;
            margin-bottom: 0.85rem;
            font-size: 0.875rem;
            color: #334155;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .tip-box:hover {
            border-color: #0D9488;
            transform: translateY(-2px);
        }

        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 0.85rem;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .score-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .score-grade-a { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
        .score-grade-b { background: #E0F2FE; color: #0369A1; border: 1px solid #7DD3FC; }
        .score-grade-c { background: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; }
        .score-grade-d { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }

        .alert-card-warning {
            background: #FFFBEB;
            border-left: 4px solid #D97706;
            border: 1px solid #FDE68A;
            border-left-width: 4px;
            border-radius: 0 0.5rem 0.5rem 0;
            padding: 0.65rem 1rem;
            margin-bottom: 0.45rem;
            color: #92400E;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .alert-card-success {
            background: #F0FDF4;
            border-left: 4px solid #16A34A;
            border: 1px solid #BBF7D0;
            border-left-width: 4px;
            border-radius: 0 0.5rem 0.5rem 0;
            padding: 0.65rem 1rem;
            margin-bottom: 0.45rem;
            color: #166534;
            font-size: 0.875rem;
            font-weight: 500;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: 1px solid #0D9488 !important;
            border-radius: 0.5rem !important;
            box-shadow: 0 2px 10px rgba(13, 148, 136, 0.25) !important;
        }

        div.stButton > button:not([kind="primary"]) {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 0.5rem !important;
        }
        div.stButton > button:not([kind="primary"]):hover {
            border-color: #0D9488 !important;
            color: #0D9488 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0D9488 !important;
            border-bottom: 2px solid #0D9488 !important;
        }

        [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.75rem !important;
        }

        [data-testid="stFileUploader"] {
            background: #FFFFFF !important;
            border: 1px dashed #94A3B8 !important;
            border-radius: 0.75rem !important;
        }
    </style>
    """

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
{css_theme}
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dynamic Plotly Chart Helpers (Theme-Aware)
# ---------------------------------------------------------
def create_score_gauge(score: float, grade: str, delta: float = None, is_dark_mode: bool = True) -> go.Figure:
    color = "#10B981" if score >= 85 else ("#0284C7" if score >= 75 else ("#D97706" if score >= 60 else "#DC2626")) if not is_dark_mode else ("#34D399" if score >= 85 else ("#38BDF8" if score >= 75 else ("#FBBF24" if score >= 60 else "#F87171")))
    text_color = "#FFFFFF" if is_dark_mode else "#0F172A"
    tick_color = "#94A3B8" if is_dark_mode else "#64748B"
    bg_gauge = "rgba(30, 41, 59, 0.6)" if is_dark_mode else "rgba(241, 245, 249, 0.9)"
    border_color = "#334155" if is_dark_mode else "#CBD5E1"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta is not None else ""),
        value=score,
        delta={'reference': score - delta, 'increasing': {'color': color}} if delta else None,
        number={'suffix': " / 100", 'font': {'size': 36, 'color': text_color, 'family': "JetBrains Mono, monospace"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': tick_color},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': bg_gauge,
            'borderwidth': 1,
            'bordercolor': border_color,
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [60, 75], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [75, 90], 'color': "rgba(56, 189, 248, 0.15)"},
                {'range': [90, 100], 'color': "rgba(16, 185, 129, 0.15)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=20, b=5),
        height=200,
    )
    return fig


def create_radar_chart(dimension_scores: dict, is_dark_mode: bool = True) -> go.Figure:
    categories = list(dimension_scores.keys())
    scores = [ds.score for ds in dimension_scores.values()]
    categories.append(categories[0])
    scores.append(scores[0])
    
    fill_color = 'rgba(20, 184, 166, 0.25)' if is_dark_mode else 'rgba(13, 148, 136, 0.2)'
    line_color = '#14B8A6' if is_dark_mode else '#0D9488'
    marker_color = '#2DD4BF' if is_dark_mode else '#14B8A6'
    grid_color = '#334155' if is_dark_mode else '#CBD5E1'
    text_color = '#F8FAFC' if is_dark_mode else '#0F172A'
    bg_color = 'rgba(15, 23, 42, 0.6)' if is_dark_mode else 'rgba(255, 255, 255, 0.8)'
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        marker=dict(size=6, color=marker_color),
        hoverinfo='r+theta'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color=grid_color, family="JetBrains Mono"), gridcolor=grid_color),
            angularaxis=dict(tickfont=dict(size=11, color=text_color, family="Inter"), gridcolor=grid_color),
            bgcolor=bg_color
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        height=230
    )
    return fig


def create_comparison_bar_chart(initial_report, cleaned_report, is_dark_mode: bool = True) -> go.Figure:
    dims = list(initial_report.dimension_scores.keys())
    init_scores = [initial_report.dimension_scores[d].score for d in dims]
    clean_scores = [cleaned_report.dimension_scores[d].score for d in dims]
    
    title_color = '#F8FAFC' if is_dark_mode else '#0F172A'
    text_color = '#94A3B8' if is_dark_mode else '#475569'
    grid_color = '#334155' if is_dark_mode else '#E2E8F0'
    bg_color = 'rgba(15, 23, 42, 0.6)' if is_dark_mode else 'rgba(255, 255, 255, 0.8)'
    
    fig = go.Figure(data=[
        go.Bar(name='Before Cleaning', x=dims, y=init_scores, marker_color='#F59E0B', opacity=0.85),
        go.Bar(name='After Auto-Clean', x=dims, y=clean_scores, marker_color='#10B981')
    ])
    fig.update_layout(
        barmode='group',
        title=dict(text="Score Improvement by Category (Before vs. After)", font=dict(color=title_color, size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family="Inter"),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color, range=[0, 105]),
        legend=dict(font=dict(color=title_color)),
        height=280,
        margin=dict(l=15, r=15, t=35, b=15)
    )
    return fig


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_uploaded_dataset(file, max_mb: int = 100) -> pd.DataFrame:
    """Safely parses uploaded dataset with size limits and encoding fallbacks."""
    if hasattr(file, "size") and file.size > max_mb * 1024 * 1024:
        raise ValueError(f"File size exceeds maximum allowable limit of {max_mb} MB.")

    fname = file.name.lower() if hasattr(file, "name") else "dataset.csv"
    
    if fname.endswith(".csv") or fname.endswith(".txt"):
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
    elif fname.endswith(".xlsx") or fname.endswith(".xls"):
        df = pd.read_excel(file)
    elif fname.endswith(".parquet"):
        df = pd.read_parquet(file)
    elif fname.endswith(".json"):
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)

    if df is None or df.empty:
        raise ValueError("The uploaded dataset contains zero records (empty dataset).")
    return df


# ---------------------------------------------------------
# Main Page View
# ---------------------------------------------------------
report = st.session_state.report
df = st.session_state.df

if report is None:
    # ---------------------------------------------------------
    # MAIN PAGE: HERO UPLOADER (Clean & Simple)
    # ---------------------------------------------------------
    st.markdown(f"""
    <div class="brand-hero">
        <div>
            <h1 class="brand-title">🛡️ Dataset Quality Auditor</h1>
            <p style="color: {'#94A3B8' if is_dark else '#64748B'}; font-size: 0.95rem; margin-top: 0.25rem; margin-bottom: 0;">
                Scan data health, auto-remediate issues in 1 click, and test machine learning accuracy uplift.
            </p>
        </div>
        <div style="background: rgba(20, 184, 166, 0.15); border: 1px solid #14B8A6; color: {'#2DD4BF' if is_dark else '#0D9488'}; font-weight: 800; font-size: 0.75rem; padding: 0.35rem 0.85rem; border-radius: 9999px; font-family: 'JetBrains Mono', monospace;">
            v2.0 DUAL-THEME
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.markdown("<div class='tip-box'><b>1. Upload File 📁</b><br>Drop any CSV, Excel, or Parquet file.</div>", unsafe_allow_html=True)
    with c_s2:
        st.markdown("<div class='tip-box'><b>2. View Score 🎯</b><br>See your 0–100 Data Health Score across 5 pillars.</div>", unsafe_allow_html=True)
    with c_s3:
        st.markdown("<div class='tip-box'><b>3. 1-Click Clean ✨</b><br>Auto-fix errors & download clean data instantly.</div>", unsafe_allow_html=True)

    st.markdown("### 📤 Upload Your Dataset")
    
    with st.container():
        main_uploaded_file = st.file_uploader(
            "Select or drop your dataset file (.csv, .xlsx, .parquet, .json):",
            type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
            key="main_page_uploader"
        )

        detected_cols = []
        preview_df = None
        if main_uploaded_file is not None:
            try:
                preview_df = load_uploaded_dataset(main_uploaded_file)
                detected_cols = list(preview_df.columns)
                st.success(f"📁 **Loaded:** `{main_uploaded_file.name}` ({len(preview_df):,} rows × {preview_df.shape[1]} columns)")
                with st.expander("👀 Preview Top 5 Rows & Schema", expanded=False):
                    st.dataframe(preview_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {e}")

        u_col1, u_col2 = st.columns([3, 1])
        with u_col1:
            if detected_cols:
                target_choice = st.selectbox(
                    "Target / Prediction Column (Optional):",
                    options=["-- None (Exploratory / Unsupervised) --"] + detected_cols,
                    key="main_target_select",
                    help="Choose the column you want AI to predict (e.g. loan_status, churn, price)."
                )
                selected_target = None if target_choice.startswith("--") else target_choice
            else:
                main_target_input = st.text_input(
                    "Target Column (Optional):",
                    value="",
                    placeholder="e.g. loan_status, churn (Leave blank if exploratory)",
                    key="main_target_input"
                )
                selected_target = main_target_input.strip() if main_target_input.strip() else None

        with u_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            main_audit_btn = st.button("🚀 Audit Dataset", type="primary", use_container_width=True, disabled=main_uploaded_file is None)

    if main_audit_btn and main_uploaded_file is not None:
        try:
            with st.spinner("🔍 Checking dataset for quality issues..."):
                raw_df = preview_df if preview_df is not None else load_uploaded_dataset(main_uploaded_file)
                engine = AuditEngine()
                st.session_state.report = engine.audit(raw_df, dataset_name=main_uploaded_file.name, target_col=selected_target)
                st.session_state.df = raw_df
                st.session_state.dataset_name = main_uploaded_file.name
                st.session_state.target_col = selected_target
                st.session_state.cleaned_df = None
                st.session_state.cleaned_report = None
                st.session_state.clean_stats = None
                st.session_state.benchmark_result = None
                st.toast(f"✅ Audited {len(raw_df):,} rows successfully!", icon="🛡️")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 1-Click Sample Datasets (Engaging Interactive Cards)
    st.markdown("#### 🧪 Or Test Instantly with 1-Click Datasets:")
    st.caption("No dataset on hand? Click any sample below to explore complete audit findings:")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💳 FinTech Loan Default\n(12.4k rows • Leaky & Outliers)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="loan_approval.csv", target_col="loan_status")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "loan_approval.csv"
            st.session_state.target_col = "loan_status"
            st.session_state.benchmark_result = None
            st.rerun()
    with b2:
        if st.button("📡 Telecom Customer Churn\n(7.2k rows • Class Imbalance)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "customer_churn_leaky.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="customer_churn.csv", target_col="churned")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "customer_churn.csv"
            st.session_state.target_col = "churned"
            st.session_state.benchmark_result = None
            st.rerun()
    with b3:
        if st.button("🏥 Healthcare Patient Registry\n(10.0k rows • Anomalies & Missing)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "medical_patient_anomalous.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="patient_registry.csv", target_col="readmitted")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "patient_registry.csv"
            st.session_state.target_col = "readmitted"
            st.session_state.benchmark_result = None
            st.rerun()

else:
    # ---------------------------------------------------------
    # ACTIVE AUDIT DASHBOARD (Crisp, Simple & Action-Oriented)
    # ---------------------------------------------------------
    is_cleaned = st.session_state.cleaned_report is not None
    active_report = st.session_state.cleaned_report if is_cleaned else report
    score = active_report.overall_score
    grade = active_report.grade
    delta_pts = (active_report.overall_score - report.overall_score) if is_cleaned else None

    # Top Brand Bar
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        escaped_title = html.escape(report.dataset_name)
        escaped_target = html.escape(str(report.target_column or 'None (Exploratory)'))
        st.markdown(f"""
        <div class="brand-hero" style="margin-bottom: 0;">
            <div>
                <h1 class="brand-title">📋 {escaped_title} {'<span style="color: #10B981; font-size: 0.95rem;">(Cleaned ✨)</span>' if is_cleaned else ''}</h1>
                <p style="color: {'#94A3B8' if is_dark else '#64748B'}; font-size: 0.85rem; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace;">
                    Target: <span style="color: {'#2DD4BF' if is_dark else '#0D9488'}; font-weight: 600;">{escaped_target}</span> &nbsp;•&nbsp; {active_report.total_rows:,} rows &nbsp;•&nbsp; {active_report.total_columns} columns
                </p>
            </div>
            <div>
                <div class="score-badge {'score-grade-a' if score >= 90 else ('score-grade-b' if score >= 75 else ('score-grade-c' if score >= 60 else 'score-grade-d'))}">
                    Grade {grade} ({score:.1f}/100)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("📤 Upload Another File", use_container_width=True):
            st.session_state.df = None
            st.session_state.report = None
            st.session_state.cleaned_df = None
            st.session_state.cleaned_report = None
            st.session_state.clean_stats = None
            st.session_state.benchmark_result = None
            st.rerun()

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Score Gauge & Radar (Dynamic Theme Aware)
    g_col1, g_col2 = st.columns([1, 1])
    with g_col1:
        st.plotly_chart(create_score_gauge(score, grade, delta_pts, is_dark_mode=is_dark), use_container_width=True)
    with g_col2:
        st.plotly_chart(create_radar_chart(active_report.dimension_scores, is_dark_mode=is_dark), use_container_width=True)

    # Key Summary Stats (4 Crisp Native Cards)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Rows", f"{active_report.total_rows:,}", "100% Parsed")
    with k2:
        num_c = len([p for p in active_report.column_profiles.values() if 'numeric' in p.inferred_type.lower()])
        cat_c = len([p for p in active_report.column_profiles.values() if 'categorical' in p.inferred_type.lower()])
        st.metric("Total Columns", f"{active_report.total_columns}", f"{num_c} Num • {cat_c} Cat")
    with k3:
        st.metric("Issues Found", f"{active_report.total_issues_count}", f"{active_report.severity_counts.get('LOW', 0)} Low • {active_report.severity_counts.get('MEDIUM', 0)} Med")
    with k4:
        crit_high = active_report.severity_counts.get('CRITICAL', 0) + active_report.severity_counts.get('HIGH', 0)
        st.metric(
            "Critical Fixes",
            f"{crit_high}",
            "Needs attention" if crit_high > 0 else "Clean",
            delta_color="inverse" if crit_high > 0 else "normal"
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Top Alerts (Crisp 1-liners)
    st.markdown("##### ⚠️ What Needs Attention:")
    for alert in active_report.summary_alerts:
        if alert.startswith("✓"):
            st.markdown(f"<div class='alert-card-success'>✓ {alert.replace('✓', '').strip()}</div>", unsafe_allow_html=True)
        else:
            clean_text = alert.replace("⚠", "").replace("[!]", "").replace("🚨", "").strip()
            st.markdown(f"<div class='alert-card-warning'>⚠️ {clean_text}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 6 Crisp & Simple Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 1. Quality Checks",
        "📊 2. Explore Columns",
        "✨ 3. 1-Click Clean",
        "🚀 4. ML Accuracy Test",
        "🌊 5. Compare Datasets",
        "🩺 6. AI Data Doctor & Export"
    ])

    # ---------------------------------------------------------
    # Tab 1: 9 Quality Checks
    # ---------------------------------------------------------
    with tab1:
        st.markdown("#### 🔍 The 9 Quality Checks")
        st.caption("Click any check below to expand its findings and recommended fix:")
        
        friendly_names = {
            "missing_values": ("Missing Data", "Empty cells or blank values"),
            "duplicate_rows": ("Duplicate Rows", "Repeated copycat rows"),
            "outliers": ("Outliers", "Unusual extreme high or low numbers"),
            "class_imbalance": ("Class Imbalance", "Unequal prediction labels (e.g. 95% No vs 5% Yes)"),
            "data_leakage": ("Data Leakage", "Features that cheat by giving away the answer"),
            "high_correlation": ("Redundant Features", "Two columns saying the exact same thing"),
            "data_types": ("Data Type Fixes", "Numbers or dates saved as text (e.g. '$100')"),
            "constant_columns": ("Useless Columns", "Columns where every single row is identical"),
            "suspicious_values": ("Corrupted Values", "Database error tokens like -999 or NULL")
        }

        for cid, cr in active_report.checker_results.items():
            fname, fdesc = friendly_names.get(cid, (cr.checker_name, cr.summary))
            icon = "✅" if cr.passed else ("🔴" if cr.score < 60 else "⚠️")
            
            with st.expander(f"{icon} **{fname}** — Score: {cr.score:.0f}% ({fdesc})", expanded=not cr.passed):
                st.write(f"**Findings:** {cr.summary}")
                if cr.issues:
                    for issue in cr.issues:
                        st.markdown(f"- **{issue.title}**: {issue.description}")
                        st.info(f"💡 **Suggested Fix:** {issue.remediation_suggestion}")
                else:
                    st.success("Clean! Zero issues found in this check.")

    # ---------------------------------------------------------
    # Tab 2: Column Profiler & Explorer
    # ---------------------------------------------------------
    with tab2:
        st.markdown("#### 📊 Column Explorer & Schema Profiler")
        st.caption("View data types, missing rates, uniqueness cardinality, and distributions:")
        
        prof_rows = [{
            "Column": p.name,
            "Type": p.inferred_type.capitalize(),
            "Missing": f"{p.missing_count} ({p.missing_percentage:.1f}%)",
            "Unique Values": f"{p.unique_count:,}",
            "Issues": f"⚠️ {p.issues_count}" if p.issues_count > 0 else "✅ Clean"
        } for p in active_report.column_profiles.values()]
        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True)

        st.markdown("##### 🔬 View Feature Chart:")
        sel_col = st.selectbox("Select a column to view its distribution chart:", list(active_report.column_profiles.keys()))
        if sel_col:
            active_df = st.session_state.cleaned_df if is_cleaned else df
            p_type = active_report.column_profiles[sel_col].inferred_type
            
            if "numeric" in p_type.lower() or pd.api.types.is_numeric_dtype(active_df[sel_col]):
                fig = px.histogram(active_df, x=sel_col, marginal="box", nbins=30, color_discrete_sequence=['#14B8A6'])
            else:
                top_c = active_df[sel_col].astype(str).value_counts().head(10).reset_index()
                top_c.columns = [sel_col, 'Count']
                fig = px.bar(top_c, x=sel_col, y='Count', color='Count', color_continuous_scale='Teal')
            
            p_bg = "rgba(15, 23, 42, 0.6)" if is_dark else "rgba(255, 255, 255, 0.8)"
            p_text = "#94A3B8" if is_dark else "#475569"
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=p_bg, font=dict(color=p_text), height=280)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 3: 1-Click Auto-Clean
    # ---------------------------------------------------------
    with tab3:
        st.markdown("#### ✨ 1-Click Automated Data Cleaning")
        st.caption("Auto-fixes missing values, removes outliers, cleans currency symbols, and removes leaky columns.")

        fixable_actions = []
        seen_fix = set()
        for r in report.recommended_actions:
            if r.get("auto_fixable") and r.get("fix_action"):
                fa = r.get("fix_action")
                if fa not in seen_fix:
                    seen_fix.add(fa)
                    fixable_actions.append(r)

        if fixable_actions:
            st.write(f"**Found {len(fixable_actions)} automatic fixes ready to apply:**")
            sel_all = st.checkbox("Select All Fixes", value=True, key="clean_sel_all")

            chosen_actions = []
            for idx, rec in enumerate(fixable_actions):
                chk = st.checkbox(
                    f"**{rec['title']}** — {rec['action']}",
                    value=sel_all,
                    key=f"chk_fix_{idx}_{rec['fix_action']}"
                )
                if chk:
                    chosen_actions.append(rec['fix_action'])

            if st.button("⚡ Apply Selected Fixes & Boost Score", type="primary", disabled=len(chosen_actions) == 0):
                with st.spinner("Cleaning dataset and recalculating health score..."):
                    cleaner = DatasetCleaner()
                    cleaned_df, stats = cleaner.clean(df=df, selected_actions=chosen_actions, target_col=report.target_column, report=report)
                    engine = AuditEngine()
                    cleaned_report = engine.audit(cleaned_df, dataset_name=f"{report.dataset_name} (Cleaned)", target_col=report.target_column)
                    
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.cleaned_report = cleaned_report
                    st.session_state.clean_stats = stats
                    st.session_state.benchmark_result = None
                    st.toast("Cleaned dataset successfully!", icon="✨")
                    st.rerun()

            if st.session_state.cleaned_df is not None:
                c_rep = st.session_state.cleaned_report
                st.success(f"🎉 Cleaning Done! Health score improved: **{report.overall_score:.1f}** (Grade {report.grade}) ➡️ **{c_rep.overall_score:.1f}** (Grade {c_rep.grade})!")
                st.plotly_chart(create_comparison_bar_chart(report, c_rep, is_dark_mode=is_dark), use_container_width=True)

                st.markdown("##### 📥 Download Your Cleaned Data:")
                d1, d2 = st.columns(2)
                with d1:
                    csv_buf = io.StringIO()
                    st.session_state.cleaned_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="📄 Download Cleaned CSV",
                        data=csv_buf.getvalue(),
                        file_name=f"{report.dataset_name.lower().replace('.csv', '')}_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d2:
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                        st.session_state.cleaned_df.to_excel(writer, index=False)
                    st.download_button(
                        label="📊 Download Cleaned Excel (.xlsx)",
                        data=excel_buf.getvalue(),
                        file_name=f"{report.dataset_name.lower().replace('.csv', '')}_cleaned.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        else:
            st.success("Your dataset is already in optimal condition! Zero automated cleaning needed.")

    # ---------------------------------------------------------
    # Tab 4: Machine Learning Test (Accuracy Boost)
    # ---------------------------------------------------------
    with tab4:
        st.markdown("#### 🚀 Machine Learning Accuracy Test")
        st.caption("Does cleaning your data make AI models smarter and more accurate? Let's test!")

        if not report.target_column:
            st.info("💡 Please choose a **Target Column** to run this test:")
            q_target = st.selectbox("Select Target Feature:", list(df.columns), key="ml_pick_target")
            if st.button("Set Target & Test", type="primary"):
                st.session_state.target_col = q_target
                engine = AuditEngine()
                st.session_state.report = engine.audit(df, dataset_name=report.dataset_name, target_col=q_target)
                st.rerun()
        else:
            if st.button("⚡ Run ML Benchmark (Raw vs. Clean Data)", type="primary"):
                with st.spinner("Training baseline models on Dirty vs. Clean data..."):
                    if st.session_state.cleaned_df is None:
                        cleaner = DatasetCleaner()
                        c_df, _ = cleaner.clean(df, target_col=report.target_column, report=report)
                    else:
                        c_df = st.session_state.cleaned_df

                    benchmarker = MLBenchmarkEngine()
                    b_res = benchmarker.run_benchmark(raw_df=df, clean_df=c_df, target_col=report.target_column)
                    st.session_state.benchmark_result = b_res

            if st.session_state.benchmark_result:
                b_res = st.session_state.benchmark_result
                st.success(f"🎯 **Result**: {b_res.summary_text}")

                if b_res.task_type == "classification":
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Model Accuracy", f"{b_res.clean_model.metrics.get('accuracy', 0)*100:.1f}%", f"{b_res.uplift.get('accuracy', 0)*100:+.1f}% jump")
                    with m2:
                        st.metric("F1 Quality Score", f"{b_res.clean_model.metrics.get('f1', 0):.3f}", f"{b_res.uplift.get('f1', 0):+.3f}")
                    with m3:
                        st.metric("Precision", f"{b_res.clean_model.metrics.get('precision', 0):.3f}", f"{b_res.uplift.get('precision', 0):+.3f}")
                else:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("R² Fit Score", f"{b_res.clean_model.metrics.get('r2', 0):.3f}", f"{b_res.uplift.get('r2', 0):+.3f}")
                    with m2:
                        st.metric("Error (MAE)", f"{b_res.clean_model.metrics.get('mae', 0):.2f}", f"{b_res.uplift.get('mae', 0):+.2f}")

    # ---------------------------------------------------------
    # Tab 5: Data Drift (Compare 2 Datasets)
    # ---------------------------------------------------------
    with tab5:
        st.markdown("#### 🌊 Compare 2 Datasets (Data Drift)")
        st.caption("Check if your new test data or next month's data has drifted from your baseline training data.")

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("##### 1. Baseline Data (Train)")
            st.write(f"Using current: `{report.dataset_name}` ({active_report.total_rows:,} rows)")
        with dc2:
            st.markdown("##### 2. Upload New Data (Test)")
            drift_file = st.file_uploader("Upload test dataset (.csv, .xlsx)", type=["csv", "xlsx"], key="drift_quick_file")

        if st.button("🌊 Compare Distributions", type="primary", disabled=drift_file is None):
            try:
                test_df = load_uploaded_dataset(drift_file)
                detector = DriftDetector()
                d_rep = detector.compare_datasets(df, test_df, ref_name=report.dataset_name, curr_name=drift_file.name)
                
                status_icon = "🟢" if d_rep.overall_status == "STABLE" else ("🟡" if d_rep.overall_status == "MODERATE_DRIFT" else "🔴")
                st.markdown(f"### {status_icon} Drift Status: `{d_rep.overall_status}` (Drift Index: {d_rep.overall_drift_score:.1f}%)")
                for a in d_rep.summary_alerts:
                    st.markdown(f"<div class='alert-card-warning'>{a}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # Tab 6: AI Data Doctor & Export Reports (100% Offline & Smart Engine)
    # ---------------------------------------------------------
    with tab6:
        st.markdown("#### 🩺 AI Data Doctor & Executive Diagnosis")
        st.caption("100% Offline, Zero-API-Key Diagnostic Engine. Instant root-cause analysis, ML risk evaluations, and exportable reports.")
        
        doctor = DataDoctor()
        diag = doctor.diagnose(report)

        def query_doctor(q_text: str) -> str:
            if hasattr(doctor, "answer_query"):
                return doctor.answer_query(q_text, report)
            return (
                f"💡 **AI Executive Overview for '{report.dataset_name}'**:\n\n"
                f"- **Overall Health**: {report.overall_score:.1f}/100 (Grade {report.grade})\n"
                f"- **Risk Level**: {diag.risk_level} ({diag.overall_health_badge})\n"
                f"- **Summary**: {diag.executive_summary}\n\n"
                f"💡 **Remediation**: Use the **✨ 1-Click Clean** tab to automatically resolve detected issues."
            )

        # 1. Executive Summary & Health Badge
        st.markdown(f"""
        <div class="brand-hero" style="margin-top: 0.5rem; margin-bottom: 1rem; border-color: rgba(99, 102, 241, 0.4);">
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {'#FFFFFF' if is_dark else '#0F172A'}; margin-bottom: 0.3rem;">
                    {diag.overall_health_badge} &nbsp;·&nbsp; <span style="font-size: 0.9rem; color: #64748B;">Risk Index: {diag.risk_score}/100 ({diag.risk_level})</span>
                </div>
                <div style="font-size: 0.9rem; color: {'#CBD5E1' if is_dark else '#334155'}; line-height: 1.45;">
                    {diag.executive_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Interactive AI Data Doctor Consultation
        st.markdown("##### 💬 Ask the AI Data Doctor (1-Click Presets):")

        # Quick preset buttons in columns
        p1, p2, p3, p4 = st.columns(4)
        active_query = None
        with p1:
            if st.button("❓ Why is score low?", use_container_width=True, key="preset_why"):
                active_query = "Why is my quality score low?"
        with p2:
            if st.button("🤖 ML Model Risks?", use_container_width=True, key="preset_ml"):
                active_query = "How will this affect ML models and algorithms?"
        with p3:
            if st.button("🎯 Top Priority Fix?", use_container_width=True, key="preset_prio"):
                active_query = "What is my top priority fix to improve quality?"
        with p4:
            if st.button("⚖️ Business Risks?", use_container_width=True, key="preset_risk"):
                active_query = "What are my business and compliance risks?"

        # Feature Diagnostic Dropdown + Ask
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        col_select_col, col_btn_col = st.columns([3, 1])
        with col_select_col:
            selected_col_query = st.selectbox(
                "🔍 Quick Feature Inspection (select column to analyze):",
                options=list(report.column_profiles.keys()),
                key="doctor_col_select"
            )
        with col_btn_col:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔬 Diagnose Column", use_container_width=True, key="preset_col_btn"):
                active_query = f"Explain feature {selected_col_query}"

        # Custom text inquiry input
        custom_q = st.text_input(
            "✍️ Or type a custom question:",
            placeholder="e.g. 'How to fix missing values?', 'Explain class imbalance', or 'Is this ready for XGBoost?'",
            key="custom_doctor_query"
        )
        if custom_q:
            active_query = custom_q

        # If a query is active, store in session_state and answer
        if active_query:
            st.session_state["active_doctor_query"] = active_query
            st.session_state["active_doctor_answer"] = query_doctor(active_query)
        elif "active_doctor_answer" not in st.session_state:
            # Default initial insight
            st.session_state["active_doctor_query"] = "Executive Data Quality Overview"
            st.session_state["active_doctor_answer"] = query_doctor("executive overview")

        # Render Data Doctor's Response Card
        if "active_doctor_answer" in st.session_state:
            doc_bg = "rgba(30, 41, 59, 0.75)" if is_dark else "#FFFFFF"
            doc_border = "rgba(56, 189, 248, 0.4)" if is_dark else "#BAE6FD"
            doc_title = "#38BDF8" if is_dark else "#0284C7"
            doc_sub = "#E2E8F0" if is_dark else "#1E293B"
            
            st.markdown(f"""
            <div style="background: {doc_bg}; border: 1px solid {doc_border}; border-radius: 0.85rem; padding: 1.25rem 1.5rem; margin-top: 1rem; margin-bottom: 1.25rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: {doc_title}; letter-spacing: 0.05em; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;">
                    💡 Data Doctor Consultation &nbsp;›&nbsp; <span style="color: {doc_sub}; text-transform: none;">{st.session_state.get('active_doctor_query', '')}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state["active_doctor_answer"])
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Step-by-Step Remediation Roadmap
        st.markdown("##### 🗺️ Recommended Action Plan:")
        for step in diag.remediation_roadmap:
            step_color = "#CBD5E1" if is_dark else "#334155"
            st.markdown(f"<div style='margin-bottom: 0.35rem; color: {step_color};'>• {step}</div>", unsafe_allow_html=True)

        # 4. Downloads
        st.markdown("---")
        st.markdown("##### 📥 Export Interactive Reports:")
        exp1, exp2 = st.columns(2)
        with exp1:
            html_rep = ReportGenerator.generate_html_report(report, st.session_state.cleaned_report)
            st.download_button(
                label="🌐 Download Interactive HTML Report",
                data=html_rep,
                file_name=f"{report.dataset_name.lower().replace('.csv', '')}_quality_report.html",
                mime="text/html",
                use_container_width=True
            )
        with exp2:
            py_pipe = ReportGenerator.generate_python_script(report)
            st.download_button(
                label="🐍 Download Python Cleaning Script (.py)",
                data=py_pipe,
                file_name=f"clean_{report.dataset_name.lower().replace('.csv', '')}_pipeline.py",
                mime="text/x-python",
                use_container_width=True
            )
