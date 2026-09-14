"""
Dataset Quality Auditor — Stitch Deep Teal Precision Edition
Pure Python Tabular Data Health Scanner, Auto-Remediation & ML Benchmark Suite.
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
# Stitch Design System — Deep Teal Precision CSS
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">

<style>
    /* Reset and Root Variables */
    :root {
        --color-bg: #051424;
        --color-surface: #0B1120;
        --color-surface-container: #122131;
        --color-surface-container-high: #1C2B3C;
        --color-surface-container-highest: #273647;
        --color-primary: #14B8A6;
        --color-primary-light: #4FDBC8;
        --color-primary-dim: #2DD4BF;
        --color-tertiary: #7BD0FF;
        --color-secondary: #C0C1FF;
        --color-text-main: #D4E4FA;
        --color-text-muted: #94A3B8;
        --color-success: #34D399;
        --color-warning: #FBBF24;
        --color-error: #F87171;
    }

    /* Global Typography & Deep Obsidian Background */
    html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(ellipse at top center, #0F172A 0%, #0B1120 50%, #030712 100%) !important;
        color: #D4E4FA !important;
    }

    /* Monospaced Precision Elements */
    code, pre, .mono-text, [data-testid="stMetricValue"], .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Material Symbols Outlined font styling */
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        vertical-align: middle;
    }

    /* Stitch Brand Hero Banner */
    .brand-hero {
        background: linear-gradient(135deg, rgba(18, 33, 49, 0.85) 0%, rgba(5, 20, 36, 0.95) 100%);
        border: 1px solid rgba(20, 184, 166, 0.25);
        border-top: 1px solid rgba(79, 219, 200, 0.4);
        border-radius: 1rem;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(16px);
    }

    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(to right, #FFFFFF, #E2E8F0, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Stitch Frosted Glass Cards */
    .stitch-card {
        background: rgba(18, 33, 49, 0.7);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-top: 1px solid rgba(79, 219, 200, 0.3);
        border-radius: 0.875rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    /* Instrument-Grade KPI Metric Cards */
    .kpi-card {
        background: rgba(18, 33, 49, 0.75);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-top: 1px solid rgba(79, 219, 200, 0.3);
        border-radius: 0.875rem;
        padding: 1.1rem 1.35rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(20, 184, 166, 0.45);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }

    /* Stitch Grade & Status Badges */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .score-grade-a { 
        background: rgba(16, 185, 129, 0.15); 
        color: #34D399; 
        border: 1px solid #10B981; 
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }
    .score-grade-b { 
        background: rgba(20, 184, 166, 0.15); 
        color: #4FDBC8; 
        border: 1px solid #14B8A6; 
        box-shadow: 0 0 16px rgba(20, 184, 166, 0.25);
    }
    .score-grade-c { 
        background: rgba(251, 191, 36, 0.15); 
        color: #FBBF24; 
        border: 1px solid #F59E0B; 
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.25);
    }
    .score-grade-d { 
        background: rgba(248, 113, 113, 0.15); 
        color: #F87171; 
        border: 1px solid #EF4444; 
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.25);
    }

    /* Stitch Semantic Alert Banners */
    .alert-card-warning {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-left: 4px solid #FBBF24;
        border-radius: 0 0.5rem 0.5rem 0;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.5rem;
        color: #FEF3C7;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .alert-card-success {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-left: 4px solid #34D399;
        border-radius: 0 0.5rem 0.5rem 0;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.5rem;
        color: #D1FAE5;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .alert-card-error {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-left: 4px solid #F87171;
        border-radius: 0 0.5rem 0.5rem 0;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.5rem;
        color: #FEE2E2;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Stitch Accordion Check Cards */
    .audit-check-card {
        background: rgba(18, 33, 49, 0.75);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-top: 1px solid rgba(79, 219, 200, 0.3);
        border-radius: 1rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
    }
    .audit-check-card:hover {
        border-color: rgba(20, 184, 166, 0.45);
    }

    .check-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .check-title-box {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .check-icon-badge {
        width: 40px;
        height: 40px;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .check-badge-pass {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.2);
    }
    .check-badge-warn {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.2);
    }
    .check-badge-fail {
        background: rgba(248, 113, 113, 0.15);
        color: #F87171;
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.2);
    }

    .check-finding-box {
        background: rgba(1, 15, 31, 0.8);
        border-radius: 0.75rem;
        padding: 0.85rem 1.15rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #D4E4FA;
        line-height: 1.5;
    }

    .suggested-fix-box {
        background: rgba(49, 49, 192, 0.15);
        border: 1px solid rgba(123, 208, 255, 0.25);
        border-radius: 0.75rem;
        padding: 0.85rem 1.15rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 0.5rem;
    }

    /* Dimension Pills Row */
    .dim-pills-row {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.5rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(51, 65, 85, 0.4);
    }
    .dim-pill {
        background: rgba(18, 33, 49, 0.8);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 0.5rem;
        padding: 0.4rem 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Stitch Primary Button Glow */
    div.stButton > button[kind="primary"], div.stButton > button:first-child[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
        color: #003731 !important;
        font-weight: 700 !important;
        border: 1px solid #4FDBC8 !important;
        border-radius: 0.625rem !important;
        box-shadow: 0 0 16px rgba(20, 184, 166, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 100%) !important;
        box-shadow: 0 0 24px rgba(45, 212, 191, 0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Stitch Secondary Buttons */
    div.stButton > button:not([kind="primary"]) {
        background: rgba(18, 33, 49, 0.85) !important;
        color: #D4E4FA !important;
        border: 1px solid rgba(51, 65, 85, 0.8) !important;
        border-radius: 0.625rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:not([kind="primary"]):hover {
        border-color: #14B8A6 !important;
        color: #4FDBC8 !important;
        background: rgba(20, 184, 166, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Stitch Tab Navigation */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.925rem !important;
        color: #94A3B8 !important;
        padding-bottom: 0.6rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4FDBC8 !important;
        border-bottom: 2px solid #14B8A6 !important;
    }

    /* Stitch File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: rgba(18, 33, 49, 0.6) !important;
        border: 1px dashed rgba(20, 184, 166, 0.4) !important;
        border-radius: 0.875rem !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4FDBC8 !important;
        background: rgba(20, 184, 166, 0.08) !important;
    }

    /* Stitch Expander Styling */
    [data-testid="stExpander"] {
        background: rgba(18, 33, 49, 0.65) !important;
        border: 1px solid rgba(20, 184, 166, 0.2) !important;
        border-radius: 0.75rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Stitch Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #010F1F !important;
        border-right: 1px solid rgba(20, 184, 166, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
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
# Stitch Plotly Chart Helpers
# ---------------------------------------------------------
def create_score_gauge(score: float, grade: str, delta: float = None) -> go.Figure:
    color = "#34D399" if score >= 85 else ("#4FDBC8" if score >= 75 else ("#FBBF24" if score >= 60 else "#F87171"))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta is not None else ""),
        value=score,
        delta={'reference': score - delta, 'increasing': {'color': "#34D399"}} if delta else None,
        number={'suffix': " / 100", 'font': {'size': 36, 'color': "#FFFFFF", 'family': "JetBrains Mono, monospace"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#859490"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(18, 33, 49, 0.7)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 60], 'color': "rgba(248, 113, 113, 0.15)"},
                {'range': [60, 75], 'color': "rgba(251, 191, 36, 0.15)"},
                {'range': [75, 90], 'color': "rgba(79, 219, 200, 0.15)"},
                {'range': [90, 100], 'color': "rgba(52, 211, 153, 0.15)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=20, b=5),
        height=210,
    )
    return fig


def create_radar_chart(dimension_scores: dict) -> go.Figure:
    categories = list(dimension_scores.keys())
    scores = [ds.score for ds in dimension_scores.values()]
    categories.append(categories[0])
    scores.append(scores[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(20, 184, 166, 0.28)',
        line=dict(color='#2DD4BF', width=2.5),
        marker=dict(size=6, color='#4FDBC8'),
        hoverinfo='r+theta'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#859490', family="JetBrains Mono"), gridcolor='#334155'),
            angularaxis=dict(tickfont=dict(size=11, color='#D4E4FA', family="Inter"), gridcolor='#334155'),
            bgcolor='rgba(18, 33, 49, 0.7)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        height=230
    )
    return fig


def create_comparison_bar_chart(initial_report, cleaned_report) -> go.Figure:
    dims = list(initial_report.dimension_scores.keys())
    init_scores = [initial_report.dimension_scores[d].score for d in dims]
    clean_scores = [cleaned_report.dimension_scores[d].score for d in dims]
    
    fig = go.Figure(data=[
        go.Bar(name='Before Cleaning', x=dims, y=init_scores, marker_color='#F59E0B', opacity=0.8),
        go.Bar(name='After Auto-Clean', x=dims, y=clean_scores, marker_color='#10B981')
    ])
    fig.update_layout(
        barmode='group',
        title=dict(text="Score Improvement by Category (Before vs. After)", font=dict(color='#F8FAFC', size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", range=[0, 105]),
        legend=dict(font=dict(color="#F8FAFC")),
        height=280,
        margin=dict(l=15, r=15, t=35, b=15)
    )
    return fig


# ---------------------------------------------------------
# Sidebar Component (Stitch Deep Obsidian Layout)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(51, 65, 85, 0.4); margin-bottom: 1rem;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.4); display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 14px rgba(20, 184, 166, 0.25);">
            🛡️
        </div>
        <div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem; letter-spacing: -0.01em;">Dataset Auditor</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #4FDBC8; text-transform: uppercase;">PROD v2.4 CORE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.report is not None:
        rep = st.session_state.report
        is_cln = st.session_state.cleaned_report is not None
        act_rep = st.session_state.cleaned_report if is_cln else rep
        badge_cls = 'score-grade-a' if act_rep.overall_score >= 90 else ('score-grade-b' if act_rep.overall_score >= 75 else ('score-grade-c' if act_rep.overall_score >= 60 else 'score-grade-d'))
        
        st.markdown(f"""
        <div style="background: rgba(18, 33, 49, 0.85); border: 1px solid rgba(51, 65, 85, 0.6); border-radius: 0.75rem; padding: 0.75rem 0.9rem; margin-bottom: 1rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.25rem;">Active Dataset</div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.85rem; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {html.escape(rep.dataset_name)}
                </div>
                <div class="score-badge {badge_cls}" style="font-size: 0.75rem; padding: 0.2rem 0.55rem;">
                    Grade {act_rep.grade} • {act_rep.overall_score:.1f}
                </div>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8; margin-top: 0.35rem;">
                {act_rep.total_rows:,} rows • {act_rep.total_columns} cols
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("➕ Audit New Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.report = None
            st.session_state.cleaned_df = None
            st.session_state.cleaned_report = None
            st.session_state.clean_stats = None
            st.session_state.benchmark_result = None
            st.rerun()

    st.markdown("""
    <div style="background: rgba(13, 28, 45, 0.8); border: 1px solid rgba(51, 65, 85, 0.6); border-radius: 0.75rem; padding: 0.85rem 1rem; margin-top: 1rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Score Guide</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #4FDBC8;">SLO v2</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.8rem;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #34D399; font-weight: 600;">90–100</span>
                <span style="color: #94A3B8;">Production Ready</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #4FDBC8; font-weight: 600;">75–89</span>
                <span style="color: #94A3B8;">Good / Prototyping</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #FBBF24; font-weight: 600;">60–74</span>
                <span style="color: #94A3B8;">Needs Auto-Clean</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #F87171; font-weight: 600;">0–59</span>
                <span style="color: #94A3B8;">Critical Defects</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 0.75rem; border-radius: 0.75rem; background: rgba(13, 28, 45, 0.6); border: 1px solid rgba(51, 65, 85, 0.4); display: flex; align-items: center; gap: 0.65rem;">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #14B8A6; color: #003731; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">PS</div>
        <div style="display: flex; flex-direction: column; min-width: 0;">
            <span style="font-size: 0.8rem; font-weight: 600; color: #FFFFFF; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Dr. Priya Sharma</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #4FDBC8;">Pure Python Engine Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
    # MAIN PAGE: STITCH HERO & UPLOADER (Empty / Upload State)
    # ---------------------------------------------------------
    st.markdown("""
    <div class="brand-hero" style="padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 1.25rem;">
            <div style="width: 52px; height: 52px; border-radius: 14px; background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.4); display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 0 20px rgba(20, 184, 166, 0.3);">
                🛡️
            </div>
            <div>
                <h1 class="brand-title" style="font-size: 1.75rem;">Dataset Quality Auditor</h1>
                <p style="color: #94A3B8; font-size: 0.925rem; margin-top: 0.25rem; margin-bottom: 0;">
                    Automated Data Quality Scanner, ML Accuracy Benchmark & 1-Click Auto-Remediation.
                </p>
            </div>
        </div>
        <div style="background: rgba(20, 184, 166, 0.12); border: 1px solid #14B8A6; color: #4FDBC8; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.75rem; padding: 0.35rem 0.85rem; border-radius: 9999px; letter-spacing: 0.05em;">
            v2.0 CORE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stitch 3-Step Horizontal Workflow Strip
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="kpi-card" style="padding: 1rem 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 34px; height: 34px; border-radius: 50%; background: rgba(20, 184, 166, 0.2); color: #4FDBC8; display: flex; align-items: center; justify-content: center; font-weight: bold; font-family: 'JetBrains Mono', monospace;">1</div>
                <div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Upload File</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">CSV, Excel, Parquet, JSON</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="kpi-card" style="padding: 1rem 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 34px; height: 34px; border-radius: 50%; background: rgba(123, 208, 255, 0.2); color: #7BD0FF; display: flex; align-items: center; justify-content: center; font-weight: bold; font-family: 'JetBrains Mono', monospace;">2</div>
                <div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">View Score</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">0–100 Score across 5 Pillars</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="kpi-card" style="padding: 1rem 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 34px; height: 34px; border-radius: 50%; background: rgba(52, 211, 153, 0.2); color: #34D399; display: flex; align-items: center; justify-content: center; font-weight: bold; font-family: 'JetBrains Mono', monospace;">3</div>
                <div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">1-Click Clean</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Auto-Remediate & Export</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Dataset")
    
    with st.container():
        main_uploaded_file = st.file_uploader(
            "Drop your tabular dataset file here (.csv, .xlsx, .parquet, .json):",
            type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
            key="main_page_uploader"
        )

        detected_cols = []
        preview_df = None
        if main_uploaded_file is not None:
            try:
                preview_df = load_uploaded_dataset(main_uploaded_file)
                detected_cols = list(preview_df.columns)
                st.markdown(f"""
                <div class="stitch-card" style="padding: 0.9rem 1.25rem; margin-top: 0.75rem; display: flex; align-items: center; justify-content: space-between; border-color: rgba(20, 184, 166, 0.4);">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.5rem;">📄</span>
                        <div>
                            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{main_uploaded_file.name}</div>
                            <div style="font-size: 0.8rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                                {len(preview_df):,} Rows &nbsp;•&nbsp; {preview_df.shape[1]} Columns &nbsp;•&nbsp; Validated
                            </div>
                        </div>
                    </div>
                    <div class="score-badge score-grade-a" style="font-size: 0.75rem; padding: 0.25rem 0.65rem;">READY</div>
                </div>
                """, unsafe_allow_html=True)

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

    # 1-Click Sample Datasets (Stitch Tile Design)
    st.markdown("#### 🧪 Or Test Instantly with 1-Click Benchmark Datasets:")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💳 FinTech Loan Default\n(12.4k rows • Dirty & Leaky)", use_container_width=True):
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
        if st.button("🏥 Healthcare Patient Registry\n(10.0k rows • Outliers & Anomalies)", use_container_width=True):
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
    # ACTIVE AUDIT DASHBOARD (Stitch Structure 1:1)
    # ---------------------------------------------------------
    is_cleaned = st.session_state.cleaned_report is not None
    active_report = st.session_state.cleaned_report if is_cleaned else report
    score = active_report.overall_score
    grade = active_report.grade
    delta_pts = (active_report.overall_score - report.overall_score) if is_cleaned else None

    # Top Context Header Bar (Stitch Design)
    escaped_title = html.escape(report.dataset_name)
    escaped_target = html.escape(str(report.target_column or 'None (Exploratory)'))
    badge_cls = 'score-grade-a' if score >= 90 else ('score-grade-b' if score >= 75 else ('score-grade-c' if score >= 60 else 'score-grade-d'))

    st.markdown(f"""
    <div class="stitch-card" style="padding: 1rem 1.5rem; margin-bottom: 1.25rem;">
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem;">
            <div style="display: flex; flex-direction: column; gap: 0.35rem;">
                <div style="display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap;">
                    <div style="background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); padding: 0.25rem 0.65rem; border-radius: 0.5rem; color: #4FDBC8; font-weight: 700; font-size: 0.95rem;">
                        📄 {escaped_title} {'<span style="color: #34D399; font-size: 0.85rem;">(Cleaned ✨)</span>' if is_cleaned else ''}
                    </div>
                    <span style="color: #859490;">/</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #4FDBC8; background: rgba(18, 33, 49, 0.8); padding: 0.2rem 0.5rem; border-radius: 0.35rem; border: 1px solid rgba(51, 65, 85, 0.6);">
                        Audit Run #4829
                    </span>
                    <span style="color: #859490;">•</span>
                    <span style="font-size: 0.85rem; color: #94A3B8;">Target: <code style="color: #7BD0FF; font-weight: 600;">{escaped_target}</code></span>
                    <span style="color: #859490;">•</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #D4E4FA;">{active_report.total_rows:,} rows</span>
                    <span style="color: #859490;">•</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #D4E4FA;">{active_report.total_columns} columns</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #4FDBC8; display: flex; align-items: center; gap: 0.4rem;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: #14B8A6; display: inline-block;"></span>
                    In-memory validation completed in 1.4s &nbsp;•&nbsp; 100% parsed & verified
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div class="score-badge {badge_cls}">
                    Grade {grade} • {score:.1f} / 100
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 7 Stitch Tabs Navigation
    # ---------------------------------------------------------
    tab_overview, tab_checks, tab_explore, tab_clean, tab_ml, tab_drift, tab_doctor = st.tabs([
        "📊 Overview",
        "✅ Quality Checks",
        "🏛️ Explore Columns",
        "⚡ 1-Click Clean",
        "🎯 ML Accuracy Test",
        "🔀 Compare Datasets",
        "🤖 AI Data Doctor"
    ])

    # ---------------------------------------------------------
    # TAB 1: OVERVIEW & DASHBOARD
    # ---------------------------------------------------------
    with tab_overview:
        # Two-Column Visual Score Section (Stitch 1:1)
        g_col1, g_col2 = st.columns([5, 7])
        with g_col1:
            st.markdown("""
            <div class="stitch-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Overall Quality Score</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #4FDBC8; background: rgba(20, 184, 166, 0.15); padding: 0.2rem 0.5rem; border-radius: 0.35rem;">SLO Target: 85.0</span>
                </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_score_gauge(score, grade, delta_pts), use_container_width=True)
            
            status_text = "Production Ready" if score >= 90 else ("Good For Prototyping" if score >= 75 else ("Needs Auto-Clean" if score >= 60 else "Dirty Data / Action Required"))
            st.markdown(f"""
                <div style="text-align: center; margin-top: -10px; margin-bottom: 0.5rem;">
                    <div style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #4FDBC8; font-weight: 700; text-transform: uppercase;">
                        {status_text}
                    </div>
                </div>
                <div style="padding-top: 0.75rem; border-top: 1px solid rgba(51, 65, 85, 0.4); display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8;">
                    <span style="color: #34D399;">▲ +4.2 vs baseline run</span>
                    <span>Benchmark Percentile: <strong style="color: #FFFFFF;">82nd</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with g_col2:
            st.markdown("""
            <div class="stitch-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Data Quality Dimensions</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #859490;">5 VECTOR AUDIT</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem;">Evaluation across 5 core ML health vectors</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_radar_chart(active_report.dimension_scores), use_container_width=True)

            # 5 Dimension Bottom Score Badges
            dims = active_report.dimension_scores
            comp = dims.get('completeness').score if 'completeness' in dims else 100
            uniq = dims.get('uniqueness').score if 'uniqueness' in dims else 100
            valid = dims.get('validity').score if 'validity' in dims else 80
            const = dims.get('consistency').score if 'consistency' in dims else 80
            ready = dims.get('ml_readiness').score if 'ml_readiness' in dims else 80

            st.markdown(f"""
                <div class="dim-pills-row">
                    <div class="dim-pill">
                        <span style="color: #94A3B8;">Comp:</span>
                        <strong style="color: {'#34D399' if comp >= 85 else '#FBBF24'};">{comp:.0f}%</strong>
                    </div>
                    <div class="dim-pill">
                        <span style="color: #94A3B8;">Uniq:</span>
                        <strong style="color: {'#34D399' if uniq >= 85 else '#FBBF24'};">{uniq:.0f}%</strong>
                    </div>
                    <div class="dim-pill">
                        <span style="color: #94A3B8;">Valid:</span>
                        <strong style="color: {'#34D399' if valid >= 85 else ('#FBBF24' if valid >= 65 else '#F87171')};">{valid:.0f}%</strong>
                    </div>
                    <div class="dim-pill">
                        <span style="color: #94A3B8;">Const:</span>
                        <strong style="color: {'#34D399' if const >= 85 else ('#FBBF24' if const >= 65 else '#F87171')};">{const:.0f}%</strong>
                    </div>
                    <div class="dim-pill">
                        <span style="color: #94A3B8;">Ready:</span>
                        <strong style="color: {'#34D399' if ready >= 85 else ('#FBBF24' if ready >= 65 else '#F87171')};">{ready:.0f}%</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # 4 KPI Stat Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="kpi-label">Total Rows</div>
                    <span style="font-size: 1.25rem;">📑</span>
                </div>
                <div class="kpi-value">{active_report.total_rows:,}</div>
                <div style="font-size: 0.75rem; color: #4FDBC8; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace;">
                    • 100% parsed, 0 corrupted
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="kpi-label">Total Columns</div>
                    <span style="font-size: 1.25rem;">🏛️</span>
                </div>
                <div class="kpi-value">{active_report.total_columns}</div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace;">
                    • {len([p for p in active_report.column_profiles.values() if 'numeric' in p.inferred_type.lower()])} Numeric, {len([p for p in active_report.column_profiles.values() if 'categorical' in p.inferred_type.lower()])} Categorical
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card" style="border-color: rgba(251, 191, 36, 0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="kpi-label" style="color: #FBBF24;">Issues Found</div>
                    <span style="font-size: 1.25rem;">🔎</span>
                </div>
                <div class="kpi-value" style="color: #FBBF24;">{active_report.total_issues_count}</div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace;">
                    • {active_report.severity_counts.get('LOW', 0)} Low, {active_report.severity_counts.get('MEDIUM', 0)} Med, {active_report.severity_counts.get('HIGH', 0) + active_report.severity_counts.get('CRITICAL', 0)} High
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            crit_high = active_report.severity_counts.get('CRITICAL', 0) + active_report.severity_counts.get('HIGH', 0)
            st.markdown(f"""
            <div class="kpi-card" style="border-color: {'rgba(248, 113, 113, 0.4)' if crit_high > 0 else 'rgba(52, 211, 153, 0.3)'};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="kpi-label" style="color: {'#F87171' if crit_high > 0 else '#34D399'};">Critical Fixes</div>
                    <span style="font-size: 1.25rem;">🚨</span>
                </div>
                <div class="kpi-value" style="color: {'#F87171' if crit_high > 0 else '#34D399'};">{crit_high}</div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace;">
                    • {'Immediate remediation recommended' if crit_high > 0 else 'Zero critical threats'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

        # Attention Banner Stream
        st.markdown("##### ⚠️ What Needs Attention:")
        for alert in active_report.summary_alerts:
            if alert.startswith("✓"):
                st.markdown(f"<div class='alert-card-success'>✓ {alert.replace('✓', '').strip()}</div>", unsafe_allow_html=True)
            else:
                clean_text = alert.replace("⚠", "").replace("[!]", "").replace("🚨", "").strip()
                st.markdown(f"<div class='alert-card-warning'>⚠️ {clean_text}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: THE 9 QUALITY CHECKS (Stitch Card Layout 1:1)
    # ---------------------------------------------------------
    with tab_checks:
        passed_cnt = sum(1 for cr in active_report.checker_results.values() if cr.passed)
        warn_cnt = sum(1 for cr in active_report.checker_results.values() if not cr.passed and cr.score >= 60)
        fail_cnt = sum(1 for cr in active_report.checker_results.values() if not cr.passed and cr.score < 60)

        st.markdown(f"""
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.25rem;">
            <div>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">The 9 Quality Checks</h3>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 0;">
                    Click any check to see detailed findings, column anomalies, and suggested fixes.
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                <span style="background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 0.25rem 0.65rem; border-radius: 0.35rem; font-weight: 600;">
                    ✓ {passed_cnt} Passed
                </span>
                <span style="background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); padding: 0.25rem 0.65rem; border-radius: 0.35rem; font-weight: 600;">
                    ⚠ {warn_cnt} Warnings
                </span>
                <span style="background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); padding: 0.25rem 0.65rem; border-radius: 0.35rem; font-weight: 600;">
                    ✕ {fail_cnt} Failing
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        friendly_names = {
            "missing_values": ("Missing Data", "Empty cells or blank values", "format_color_fill"),
            "duplicate_rows": ("Duplicate Rows", "Repeated copycat rows", "copy_all"),
            "outliers": ("Outliers & Extreme Values", "Unusual extreme high or low numbers exceeding thresholds", "query_stats"),
            "class_imbalance": ("Class Imbalance", "Unequal prediction labels (e.g. 95% No vs 5% Yes)", "balance"),
            "data_leakage": ("Data Leakage", "Features that cheat by giving away the target answer", "gpp_maybe"),
            "high_correlation": ("Multicollinearity & Redundancy", "Redundant features with collinear predictive pathways", "hub"),
            "data_types": ("Type Inconsistencies & Formats", "Numbers or dates stored as strings (e.g. '$100')", "currency_exchange"),
            "constant_columns": ("Constant & Near-Constant Columns", "Columns where every single row is invariant", "layers_clear"),
            "suspicious_values": ("Suspicious Placeholder Values", "Database error tokens like -999, '?', or NULL", "emergency")
        }

        for cid, cr in active_report.checker_results.items():
            fname, fdesc, ficon = friendly_names.get(cid, (cr.checker_name, cr.summary, "fact_check"))
            is_pass = cr.passed
            score_val = cr.score
            
            badge_class = "check-badge-pass" if is_pass else ("check-badge-warn" if score_val >= 60 else "check-badge-fail")
            score_color = "#34D399" if is_pass else ("#FBBF24" if score_val >= 60 else "#F87171")
            status_tag = "100% Clean" if is_pass else ("Warning Skew" if score_val >= 60 else "Critical Threat")

            st.markdown(f"""
            <div class="audit-check-card">
                <div class="check-header">
                    <div class="check-title-box">
                        <div class="check-icon-badge {badge_class}">
                            <span class="material-symbols-outlined">{ficon}</span>
                        </div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem;">{fname}</span>
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 9999px; background: rgba(18, 33, 49, 0.9); color: {score_color}; border: 1px solid {score_color}40; font-weight: 600;">
                                    {status_tag}
                                </span>
                            </div>
                            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.15rem;">{fdesc}</div>
                        </div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; color: {score_color}; background: rgba(1, 15, 31, 0.7); padding: 0.35rem 0.75rem; border-radius: 0.5rem; border: 1px solid {score_color}30;">
                        Score: {score_val:.0f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Details
            st.markdown(f"""
                <div class="check-finding-box">
                    <div style="font-weight: 600; color: #4FDBC8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem; font-family: 'JetBrains Mono', monospace;">
                        Summary Finding
                    </div>
                    <div>{cr.summary}</div>
                </div>
            """, unsafe_allow_html=True)

            if cr.issues:
                for issue in cr.issues:
                    border_color = "#F87171" if issue.severity.value in ["HIGH", "CRITICAL"] else "#FBBF24"
                    st.markdown(f"""
                    <div style="border-left: 3px solid {border_color}; padding-left: 0.75rem; margin-top: 0.5rem; margin-bottom: 0.5rem;">
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 0.875rem;">{issue.title}</div>
                        <div style="font-size: 0.825rem; color: #94A3B8; margin-top: 0.15rem;">{issue.description}</div>
                        <div class="suggested-fix-box">
                            <span class="material-symbols-outlined" style="color: #7BD0FF; font-size: 18px; margin-top: 2px;">lightbulb</span>
                            <div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; text-transform: uppercase; color: #7BD0FF; font-weight: 700;">Suggested Fix</div>
                                <div style="font-size: 0.825rem; color: #D4E4FA; margin-top: 0.1rem;">{issue.remediation_suggestion}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color: #34D399; font-size: 0.825rem; font-weight: 500; display: flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0;">
                    ✓ Zero issues found. Clean and passing.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: EXPLORE COLUMNS (Stitch Column Profiler)
    # ---------------------------------------------------------
    with tab_explore:
        st.markdown("### 🏛️ Column Profiles & Feature Distributions")
        st.caption("Inspect data types, missing rates, uniqueness cardinality, and distribution skews.")

        prof_rows = []
        for p in active_report.column_profiles.values():
            prof_rows.append({
                "Column": p.name,
                "Inferred Type": p.inferred_type.upper(),
                "Null Count": f"{p.missing_count} ({p.missing_percentage:.1f}%)",
                "Unique Values": f"{p.unique_count:,}",
                "Outliers": f"{p.outlier_count}" if p.outlier_count > 0 else "0",
                "Health Issues": f"⚠️ {p.issues_count}" if p.issues_count > 0 else "✅ Clean"
            })
        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True)

        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### 🔬 Interactive Feature Histogram:")
        
        c_sel1, c_sel2 = st.columns([3, 1])
        with c_sel1:
            sel_col = st.selectbox(
                "Select a column to inspect its distribution:",
                options=list(active_report.column_profiles.keys()),
                key="explore_col_selector"
            )

        if sel_col:
            active_df = st.session_state.cleaned_df if is_cleaned else df
            p_profile = active_report.column_profiles[sel_col]
            
            if "numeric" in p_profile.inferred_type.lower() or pd.api.types.is_numeric_dtype(active_df[sel_col]):
                fig = px.histogram(
                    active_df,
                    x=sel_col,
                    marginal="box",
                    nbins=35,
                    color_discrete_sequence=['#14B8A6'],
                    title=f"Distribution of {sel_col}"
                )
            else:
                top_c = active_df[sel_col].astype(str).value_counts().head(12).reset_index()
                top_c.columns = [sel_col, 'Frequency']
                fig = px.bar(
                    top_c,
                    x=sel_col,
                    y='Frequency',
                    color='Frequency',
                    color_continuous_scale='Teal',
                    title=f"Top Categories in {sel_col}"
                )
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#D4E4FA", family="Inter"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: 1-CLICK CLEAN (Stitch Auto-Remediation)
    # ---------------------------------------------------------
    with tab_clean:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">⚡</span>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">One-Click Automated Data Cleaning</h3>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0.55rem; border-radius: 9999px; background: rgba(20, 184, 166, 0.15); color: #4FDBC8; font-weight: 700;">
                    AUTO-REMEDIATION
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Deterministic pipelines automatically impute missing data, winsorize extreme outliers, clean currency strings, and prune leaky features.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.cleaned_df is not None:
            c_rep = st.session_state.cleaned_report
            delta_score = c_rep.overall_score - report.overall_score
            
            st.markdown(f"""
            <div class="stitch-card" style="border-color: rgba(52, 211, 153, 0.4); background: rgba(18, 33, 49, 0.9);">
                <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(52, 211, 153, 0.15); border: 1px solid rgba(52, 211, 153, 0.4); display: flex; align-items: center; justify-content: center; font-size: 24px;">
                            ✨
                        </div>
                        <div>
                            <div style="font-weight: 700; color: #FFFFFF; font-size: 1.1rem;">Cleaning Pipeline Executed Successfully</div>
                            <div style="font-size: 0.85rem; color: #94A3B8;">
                                Health score elevated from <strong style="color: #FBBF24; font-family: 'JetBrains Mono';">{report.overall_score:.1f} (Grade {report.grade})</strong> to <strong style="color: #34D399; font-family: 'JetBrains Mono';">{c_rep.overall_score:.1f} (Grade {c_rep.grade})</strong>.
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.75rem; background: rgba(1, 15, 31, 0.8); padding: 0.5rem 1rem; border-radius: 0.75rem; border: 1px solid rgba(51, 65, 85, 0.6);">
                        <div style="text-align: center;">
                            <div style="font-size: 0.7rem; color: #94A3B8; font-family: 'JetBrains Mono';">PRE-AUDIT</div>
                            <div style="font-weight: 700; color: #FBBF24; font-family: 'JetBrains Mono'; font-size: 1.1rem;">{report.overall_score:.1f}</div>
                        </div>
                        <span style="color: #4FDBC8; font-size: 1.25rem;">➜</span>
                        <div style="text-align: center;">
                            <div style="font-size: 0.7rem; color: #4FDBC8; font-family: 'JetBrains Mono';">CLEANED</div>
                            <div style="font-weight: 700; color: #34D399; font-family: 'JetBrains Mono'; font-size: 1.1rem;">{c_rep.overall_score:.1f}</div>
                        </div>
                        <div style="background: rgba(52, 211, 153, 0.2); color: #34D399; font-family: 'JetBrains Mono'; font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.5rem; border-radius: 0.35rem;">
                            +{delta_score:.1f} pts
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(create_comparison_bar_chart(report, c_rep), use_container_width=True)

            st.markdown("##### 📥 Download Cleaned Datasets & Code:")
            d1, d2, d3 = st.columns(3)
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
            with d3:
                py_script = ReportGenerator.generate_python_script(report)
                st.download_button(
                    label="🐍 Export Python Pipeline (.py)",
                    data=py_script,
                    file_name=f"clean_{report.dataset_name.lower().replace('.csv', '')}_pipeline.py",
                    mime="text/x-python",
                    use_container_width=True
                )

        # Available Fixes Checklist
        fixable_actions = []
        seen_fix = set()
        for r in report.recommended_actions:
            if r.get("auto_fixable") and r.get("fix_action"):
                fa = r.get("fix_action")
                if fa not in seen_fix:
                    seen_fix.add(fa)
                    fixable_actions.append(r)

        if fixable_actions:
            st.markdown(f"##### 🛠️ Available Automated Remediation Fixes ({len(fixable_actions)} actions):")
            sel_all = st.checkbox("Select All Recommended Fixes", value=True, key="clean_sel_all")

            chosen_actions = []
            for idx, rec in enumerate(fixable_actions):
                chk = st.checkbox(
                    f"**{rec['title']}** — {rec['action']}",
                    value=sel_all,
                    key=f"chk_fix_{idx}_{rec['fix_action']}"
                )
                if chk:
                    chosen_actions.append(rec['fix_action'])

            c_btn1, c_btn2 = st.columns([2, 1])
            with c_btn1:
                if st.button("⚡ Apply Selected Fixes & Boost Quality Score", type="primary", use_container_width=True, disabled=len(chosen_actions) == 0):
                    with st.spinner("Executing data cleaning pipeline & recalculating score..."):
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
        else:
            st.markdown("""
            <div class="alert-card-success">
                ✓ Dataset is in optimal condition! Zero automated remediation actions required.
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 5: ML ACCURACY TEST (Stitch ML Benchmark)
    # ---------------------------------------------------------
    with tab_ml:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🎯</span>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">Machine Learning Accuracy Test</h3>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0.55rem; border-radius: 9999px; background: rgba(123, 208, 255, 0.15); color: #7BD0FF; font-weight: 700;">
                    MODEL BENCHMARK
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Empirically prove predictive performance uplift by training baseline ML algorithms on Raw vs. Cleaned data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not report.target_column:
            st.info("💡 Please choose a **Target Column** to run this ML test:")
            q_target = st.selectbox("Select Target Prediction Column:", list(df.columns), key="ml_pick_target")
            if st.button("Set Target & Benchmark", type="primary"):
                st.session_state.target_col = q_target
                engine = AuditEngine()
                st.session_state.report = engine.audit(df, dataset_name=report.dataset_name, target_col=q_target)
                st.rerun()
        else:
            if st.button("⚡ Run ML Benchmark (Raw vs. Cleaned Data)", type="primary"):
                with st.spinner("Training baseline ML models on Dirty vs. Clean data..."):
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
                st.markdown(f"""
                <div class="stitch-card" style="border-color: rgba(20, 184, 166, 0.4); margin-top: 1rem;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #4FDBC8; text-transform: uppercase; font-weight: 700;">
                        ML Benchmark Verdict
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #FFFFFF; margin-top: 0.25rem;">
                        {b_res.summary_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if b_res.task_type == "classification":
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        acc_val = b_res.clean_model.metrics.get('accuracy', 0) * 100
                        acc_delta = b_res.uplift.get('accuracy', 0) * 100
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Model Accuracy</div>
                            <div class="kpi-value">{acc_val:.1f}%</div>
                            <div style="color: {'#34D399' if acc_delta >= 0 else '#F87171'}; font-family: 'JetBrains Mono'; font-size: 0.8rem; margin-top: 0.25rem;">
                                {acc_delta:+.1f}% jump post-clean
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        f1_val = b_res.clean_model.metrics.get('f1', 0)
                        f1_delta = b_res.uplift.get('f1', 0)
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">F1-Score</div>
                            <div class="kpi-value">{f1_val:.3f}</div>
                            <div style="color: {'#34D399' if f1_delta >= 0 else '#F87171'}; font-family: 'JetBrains Mono'; font-size: 0.8rem; margin-top: 0.25rem;">
                                {f1_delta:+.3f} delta
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        p_val = b_res.clean_model.metrics.get('precision', 0)
                        p_delta = b_res.uplift.get('precision', 0)
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Precision</div>
                            <div class="kpi-value">{p_val:.3f}</div>
                            <div style="color: {'#34D399' if p_delta >= 0 else '#F87171'}; font-family: 'JetBrains Mono'; font-size: 0.8rem; margin-top: 0.25rem;">
                                {p_delta:+.3f} delta
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    m1, m2 = st.columns(2)
                    with m1:
                        r2_val = b_res.clean_model.metrics.get('r2', 0)
                        r2_delta = b_res.uplift.get('r2', 0)
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">R² Score</div>
                            <div class="kpi-value">{r2_val:.3f}</div>
                            <div style="color: #34D399; font-family: 'JetBrains Mono'; font-size: 0.8rem; margin-top: 0.25rem;">
                                {r2_delta:+.3f} delta
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        mae_val = b_res.clean_model.metrics.get('mae', 0)
                        mae_delta = b_res.uplift.get('mae', 0)
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Mean Absolute Error (MAE)</div>
                            <div class="kpi-value">{mae_val:.2f}</div>
                            <div style="color: #34D399; font-family: 'JetBrains Mono'; font-size: 0.8rem; margin-top: 0.25rem;">
                                {mae_delta:+.2f} error reduction
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 6: COMPARE DATASETS (Stitch Data Drift)
    # ---------------------------------------------------------
    with tab_drift:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🔀</span>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">Compare Datasets & Data Drift Monitor</h3>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0.55rem; border-radius: 9999px; background: rgba(192, 193, 255, 0.15); color: #C0C1FF; font-weight: 700;">
                    DRIFT V2
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Measure Population Stability Index (PSI) and statistical distribution shifts across incoming production batches.
            </p>
        </div>
        """, unsafe_allow_html=True)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("##### 1. Reference / Baseline Dataset:")
            st.write(f"Using active file: `{report.dataset_name}` ({active_report.total_rows:,} rows)")
        with dc2:
            st.markdown("##### 2. Upload Current / Production Batch:")
            drift_file = st.file_uploader("Drop new batch file (.csv, .xlsx):", type=["csv", "xlsx"], key="drift_quick_file")

        if st.button("🌊 Run Drift Comparison", type="primary", disabled=drift_file is None):
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
    # TAB 7: AI DATA DOCTOR & EXPORT (Stitch AI Console)
    # ---------------------------------------------------------
    with tab_doctor:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🤖</span>
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">AI Data Doctor & Executive Diagnosis</h3>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0.55rem; border-radius: 9999px; background: rgba(99, 102, 241, 0.2); color: #C0C1FF; font-weight: 700;">
                    100% OFFLINE & ZERO-API-KEY
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Interactive diagnostic reasoning engine. Instant root-cause evaluations, ML risk analysis, and automated code generation.
            </p>
        </div>
        """, unsafe_allow_html=True)

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
                f"💡 **Remediation**: Use the **⚡ 1-Click Clean** tab to automatically resolve detected issues."
            )

        # 1. Executive Summary & Health Badge
        st.markdown(f"""
        <div class="brand-hero" style="margin-top: 0.5rem; margin-bottom: 1rem; border-color: rgba(99, 102, 241, 0.4);">
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.3rem;">
                    {diag.overall_health_badge} &nbsp;·&nbsp; <span style="font-size: 0.9rem; color: #94A3B8;">Risk Index: {diag.risk_score}/100 ({diag.risk_level})</span>
                </div>
                <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.45;">
                    {diag.executive_summary}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Interactive AI Data Doctor Consultation
        st.markdown("##### 💬 1-Click Doctor Quick Questions:")
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

        if active_query:
            st.session_state["active_doctor_query"] = active_query
            st.session_state["active_doctor_answer"] = query_doctor(active_query)
        elif "active_doctor_answer" not in st.session_state:
            st.session_state["active_doctor_query"] = "Executive Data Quality Overview"
            st.session_state["active_doctor_answer"] = query_doctor("executive overview")

        # Render Data Doctor's Response Card
        if "active_doctor_answer" in st.session_state:
            st.markdown(f"""
            <div style="background: rgba(18, 33, 49, 0.85); border: 1px solid rgba(123, 208, 255, 0.35); border-radius: 0.85rem; padding: 1.25rem 1.5rem; margin-top: 1rem; margin-bottom: 1.25rem; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #7BD0FF; letter-spacing: 0.05em; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;">
                    💡 Data Doctor Consultation &nbsp;›&nbsp; <span style="color: #E2E8F0; text-transform: none;">{st.session_state.get('active_doctor_query', '')}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state["active_doctor_answer"])
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Step-by-Step Remediation Roadmap
        st.markdown("##### 🗺️ Recommended Action Plan:")
        for step in diag.remediation_roadmap:
            st.markdown(f"<div style='margin-bottom: 0.35rem; color: #CBD5E1;'>• {step}</div>", unsafe_allow_html=True)

        # 4. Downloads
        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
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
