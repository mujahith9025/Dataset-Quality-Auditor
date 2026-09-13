"""
Dataset Quality Auditor — Streamlit Application (Level 2.0 Next-Level Edition)
An enterprise-grade, interactive tabular data health scanner, ML baseline benchmarker,
distribution drift scanner, custom business rule engine, and AI Data Doctor.
"""

import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from core.engine import AuditEngine
from core.cleaner import DatasetCleaner
from core.report_generator import ReportGenerator
from core.ml_benchmark import MLBenchmarkEngine
from core.drift_detector import DriftDetector
from core.rules_engine import RulesEngine, RuleDefinition
from core.data_doctor import DataDoctor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dataset Quality Auditor 2.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Aesthetics & Glassmorphic CSS Styling
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0F172A, #090D16, #030712);
        color: #F8FAFC;
    }

    /* Top Brand Hero Header */
    .brand-hero {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(20, 184, 166, 0.25);
        border-radius: 1.25rem;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 20px rgba(20, 184, 166, 0.08);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-size: 1.65rem;
        font-weight: 900;
        letter-spacing: -0.025em;
        background: linear-gradient(to right, #FFFFFF, #E2E8F0, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Glassmorphic Stat KPI Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 1rem;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(20, 184, 166, 0.4);
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }

    /* Grade Badges */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .score-grade-a { 
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.3)); 
        color: #34D399; 
        border: 1px solid rgba(16, 185, 129, 0.5); 
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .score-grade-b { 
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(14, 165, 233, 0.3)); 
        color: #38BDF8; 
        border: 1px solid rgba(56, 189, 248, 0.5); 
    }
    .score-grade-c { 
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3)); 
        color: #FBBF24; 
        border: 1px solid rgba(245, 158, 11, 0.5); 
    }
    .score-grade-d { 
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3)); 
        color: #F87171; 
        border: 1px solid rgba(239, 68, 68, 0.5); 
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }

    /* Severity Badges */
    .sev-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sev-critical { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .sev-high { background: rgba(249, 115, 22, 0.2); color: #FDBA74; border: 1px solid rgba(249, 115, 22, 0.4); }
    .sev-medium { background: rgba(245, 158, 11, 0.2); color: #FDE68A; border: 1px solid rgba(245, 158, 11, 0.4); }
    .sev-low { background: rgba(14, 165, 233, 0.2); color: #BAE6FD; border: 1px solid rgba(14, 165, 233, 0.4); }

    /* Alert Banner Cards */
    .alert-card-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-left: 4px solid #F59E0B;
        border-radius: 0 0.75rem 0.75rem 0;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.6rem;
        color: #FEF3C7;
        font-weight: 600;
        font-size: 0.925rem;
    }

    .alert-card-success {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 0.75rem 0.75rem 0;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.6rem;
        color: #D1FAE5;
        font-weight: 600;
        font-size: 0.925rem;
    }

    .pipe-step {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.85rem;
        border-radius: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        font-size: 0.8rem;
        font-weight: 600;
        color: #E2E8F0;
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
# Plotly Visualization Helpers
# ---------------------------------------------------------
def create_score_gauge(score: float, grade: str, delta: float = None) -> go.Figure:
    color = "#10B981" if score >= 85 else ("#38BDF8" if score >= 75 else ("#F59E0B" if score >= 60 else "#EF4444"))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta is not None else ""),
        value=score,
        delta={'reference': score - delta, 'increasing': {'color': "#10B981"}} if delta else None,
        number={'suffix': " / 100", 'font': {'size': 36, 'color': "#FFFFFF", 'family': "Inter, sans-serif"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.6)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [60, 75], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [75, 90], 'color': "rgba(56, 189, 248, 0.15)"},
                {'range': [90, 100], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=25, b=5),
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
        fillcolor='rgba(20, 184, 166, 0.25)',
        line=dict(color='#14B8A6', width=2.5),
        marker=dict(size=7, color='#2DD4BF'),
        hoverinfo='r+theta',
        name='Quality Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color='#94A3B8'),
                gridcolor='#334155'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#F8FAFC', family='Inter'),
                gridcolor='#334155'
            ),
            bgcolor='rgba(15, 23, 42, 0.6)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=35, r=35, t=25, b=25),
        height=260
    )
    return fig


def create_column_distribution_chart(df: pd.DataFrame, col_name: str, inferred_type: str) -> go.Figure:
    series = df[col_name].dropna()
    
    if "numeric" in inferred_type.lower() or pd.api.types.is_numeric_dtype(series):
        num_series = pd.to_numeric(series, errors='coerce').dropna()
        fig = px.histogram(
            num_series,
            x=col_name,
            marginal="box",
            nbins=35,
            color_discrete_sequence=['#14B8A6'],
            opacity=0.85
        )
        fig.update_layout(
            title=dict(text=f"Distribution & Outlier Bounds: '{col_name}'", font=dict(color='#F8FAFC', size=14, family='Inter')),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    else:
        top_counts = series.astype(str).value_counts().head(15).reset_index()
        top_counts.columns = [col_name, 'Count']
        
        fig = px.bar(
            top_counts,
            x=col_name,
            y='Count',
            color='Count',
            color_continuous_scale='Teal',
            text_auto=True
        )
        fig.update_layout(
            title=dict(text=f"Top Category Frequencies: '{col_name}'", font=dict(color='#F8FAFC', size=14, family='Inter')),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        return fig


def create_correlation_heatmap(corr_data: dict) -> go.Figure:
    cols = corr_data["columns"]
    vals = corr_data["values"]
    
    fig = go.Figure(data=go.Heatmap(
        z=vals,
        x=cols,
        y=cols,
        colorscale='RdBu_r',
        zmin=-1.0,
        zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in vals],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "#FFFFFF"},
        colorbar=dict(title="Correlation (r)", tickfont=dict(color="#94A3B8"))
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#F8FAFC"),
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
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
        title=dict(text="Dimension Score Elevation (Before vs. After Remediation)", font=dict(color='#F8FAFC', size=14, family='Inter')),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", range=[0, 105]),
        legend=dict(font=dict(color="#F8FAFC")),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ---------------------------------------------------------
# Sidebar: Ingestion & Navigation Actions
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Dataset Auditor 2.0")
    st.caption("Next-Level Tabular Quality, ML Benchmarking & Drift Engine")
    st.divider()

    sidebar_uploaded = st.file_uploader(
        "Upload Single Dataset (.CSV / .XLSX)",
        type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
        key="sidebar_uploader",
        help="Upload tabular data in CSV, Excel, or Parquet format."
    )

    sidebar_target = st.text_input(
        "Target Feature (Optional)",
        value="",
        placeholder="e.g. loan_status, churned",
        key="sidebar_target_input"
    )

    if st.button("🚀 Audit from Sidebar", type="primary", use_container_width=True, disabled=sidebar_uploaded is None):
        if sidebar_uploaded:
            try:
                raw_df = pd.read_csv(sidebar_uploaded) if sidebar_uploaded.name.endswith(".csv") else pd.read_excel(sidebar_uploaded)
                engine = AuditEngine()
                st.session_state.report = engine.audit(raw_df, dataset_name=sidebar_uploaded.name, target_col=sidebar_target.strip() or None)
                st.session_state.df = raw_df
                st.session_state.dataset_name = sidebar_uploaded.name
                st.session_state.target_col = sidebar_target.strip() or None
                st.session_state.cleaned_df = None
                st.session_state.cleaned_report = None
                st.session_state.clean_stats = None
                st.session_state.benchmark_result = None
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.report is not None:
        if st.button("🔄 Reset / Audit New Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.report = None
            st.session_state.cleaned_df = None
            st.session_state.cleaned_report = None
            st.session_state.clean_stats = None
            st.session_state.benchmark_result = None
            st.rerun()

    st.divider()
    st.caption("💡 **Enterprise Suite Capabilities**:\n- 9 Quality Checkers & Score (0–100)\n- ML Baseline Uplift Benchmark\n- Train vs. Test Drift (PSI & KS)\n- Custom Business Assertions\n- AI Data Doctor Executive Report\n- CI/CD Quality Gate & CLI")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_uploaded_dataset(file) -> pd.DataFrame:
    fname = file.name.lower()
    if fname.endswith(".csv") or fname.endswith(".txt"):
        return pd.read_csv(file)
    elif fname.endswith(".xlsx") or fname.endswith(".xls"):
        return pd.read_excel(file)
    elif fname.endswith(".parquet"):
        return pd.read_parquet(file)
    elif fname.endswith(".json"):
        return pd.read_json(file)
    return pd.read_csv(file)


# ---------------------------------------------------------
# Main Page Display
# ---------------------------------------------------------
report = st.session_state.report
df = st.session_state.df

if report is None:
    # ---------------------------------------------------------
    # MAIN PAGE: HERO HEADER & PROMINENT UPLOAD INTERFACE
    # ---------------------------------------------------------
    st.markdown("""
    <div class="brand-hero">
        <div>
            <h1 class="brand-title">🛡️ Dataset Quality Auditor 2.0</h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.25rem;">
                Enterprise Tabular Health, ML Benchmarking, Drift Scanner & AI Data Doctor
            </p>
        </div>
        <div style="background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); color: #2DD4BF; font-weight: 800; font-size: 0.75rem; padding: 0.35rem 0.85rem; border-radius: 9999px; text-transform: uppercase;">
            Pure Python • Enterprise v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Main Page Prominent Upload Zone
    st.markdown("### 📤 Upload Your Dataset (.CSV, .XLSX, .Parquet, .JSON)")
    
    with st.container():
        main_uploaded_file = st.file_uploader(
            "Drag & drop your CSV or Excel dataset file directly here:",
            type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
            key="main_page_uploader",
            help="Upload your dataset file to automatically run all 9 quality checks."
        )

        detected_cols = []
        preview_df = None
        if main_uploaded_file is not None:
            try:
                preview_df = load_uploaded_dataset(main_uploaded_file)
                detected_cols = list(preview_df.columns)
                st.success(f"📁 **File Loaded:** `{main_uploaded_file.name}` — **{len(preview_df):,}** rows × **{preview_df.shape[1]}** columns ({main_uploaded_file.size / 1024:.1f} KB)")
                with st.expander("👀 Sneak Peek (First 5 Rows)", expanded=False):
                    st.dataframe(preview_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse file preview: {e}")

        u_col1, u_col2 = st.columns([3, 1])
        with u_col1:
            if detected_cols:
                target_choice = st.selectbox(
                    "Target / Label Column (Optional)",
                    options=["-- None (Unsupervised / Exploratory) --"] + detected_cols,
                    key="main_target_select",
                    help="Select target column to run class imbalance & target data leakage checks."
                )
                selected_target = None if target_choice.startswith("--") else target_choice
            else:
                main_target_input = st.text_input(
                    "Target / Label Column (Optional)",
                    value="",
                    placeholder="e.g. loan_status, churn, default (Leave blank if unsupervised)",
                    key="main_target_input",
                    help="Specify classification/regression target column to run class imbalance & leakage audits."
                )
                selected_target = main_target_input.strip() if main_target_input.strip() else None

        with u_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            main_audit_btn = st.button("🚀 Run Quality Audit", type="primary", use_container_width=True, disabled=main_uploaded_file is None)

    if main_audit_btn and main_uploaded_file is not None:
        try:
            with st.spinner("🔍 Reading dataset and executing 9 automated audit checks..."):
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
                st.toast(f"✅ Audited {len(raw_df):,} rows x {raw_df.shape[1]} columns successfully!", icon="🛡️")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. What the system checks (4 Cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
            <h4 style="color: #F8FAFC; margin-bottom: 0.25rem;">1. Audit & Profiler</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">9 automated checks, 0–100 score, sentinels, outliers, and leakage.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
            <h4 style="color: #F8FAFC; margin-bottom: 0.25rem;">2. ML Benchmark</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">Measures Accuracy/F1 uplift on dirty vs. auto-remediated data.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🌊</div>
            <h4 style="color: #F8FAFC; margin-bottom: 0.25rem;">3. Drift Scanner</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">Calculates PSI, KS tests, and catches novel unseen categories.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📐</div>
            <h4 style="color: #F8FAFC; margin-bottom: 0.25rem;">4. Custom Rules</h4>
            <p style="color: #94A3B8; font-size: 0.85rem;">Visual assertion builder for domain constraints & CI/CD gates.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. 1-Click Test Benchmarks on Main Page
    st.markdown("#### 🧪 Or Test Instantly with 1-Click Benchmark Datasets:")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📊 Loan Default Benchmark (Dirty & Leaky)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="Loan Approval Benchmark", target_col="loan_status")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "Loan Approval Benchmark"
            st.session_state.target_col = "loan_status"
            st.session_state.benchmark_result = None
            st.rerun()
    with b2:
        if st.button("📉 Telecom Churn Benchmark (Severe Skew)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "customer_churn_leaky.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="Telecom Churn Benchmark", target_col="churned")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "Telecom Churn Benchmark"
            st.session_state.target_col = "churned"
            st.session_state.benchmark_result = None
            st.rerun()
    with b3:
        if st.button("🩺 Patient Health Benchmark (Bio-Anomalies)", use_container_width=True):
            raw_df = pd.read_csv(os.path.join(SAMPLES_DIR, "medical_patient_anomalous.csv"))
            engine = AuditEngine()
            st.session_state.report = engine.audit(raw_df, dataset_name="Medical Patient Health", target_col="readmitted")
            st.session_state.df = raw_df
            st.session_state.dataset_name = "Medical Patient Health"
            st.session_state.target_col = "readmitted"
            st.session_state.benchmark_result = None
            st.rerun()

else:
    # ---------------------------------------------------------
    # ACTIVE ENTERPRISE AUDIT DASHBOARD
    # ---------------------------------------------------------
    is_cleaned = st.session_state.cleaned_report is not None
    active_report = st.session_state.cleaned_report if is_cleaned else report
    score = active_report.overall_score
    grade = active_report.grade
    delta_pts = (active_report.overall_score - report.overall_score) if is_cleaned else None

    # 1. Top Brand Banner & Action Bar
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.markdown(f"""
        <div class="brand-hero" style="margin-bottom: 0;">
            <div>
                <h1 class="brand-title">📋 {report.dataset_name} {'<span style="color: #34D399; font-size: 1rem; font-weight: bold;">(Cleaned)</span>' if is_cleaned else ''}</h1>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.25rem;">
                    Audited: {report.created_at} • Target Column: <code style="color: #2DD4BF;">{report.target_column or 'None'}</code> • Memory: {report.memory_usage_mb} MB
                </p>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <div class="score-badge {'score-grade-a' if score >= 90 else ('score-grade-b' if score >= 75 else ('score-grade-c' if score >= 60 else 'score-grade-d'))}">
                    Grade {grade} ({score:.1f}/100)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_col2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("📤 Audit Another CSV", use_container_width=True):
            st.session_state.df = None
            st.session_state.report = None
            st.session_state.cleaned_df = None
            st.session_state.cleaned_report = None
            st.session_state.clean_stats = None
            st.session_state.benchmark_result = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Hero Visual Graphics Section: Gauge + Radar
    g_col1, g_col2 = st.columns([1, 1])
    with g_col1:
        st.plotly_chart(create_score_gauge(score, grade, delta_pts), use_container_width=True)
    with g_col2:
        st.plotly_chart(create_radar_chart(active_report.dimension_scores), use_container_width=True)

    # 3. Modern KPI Stats Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Observations</div>
            <div class="kpi-value">{active_report.total_rows:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Features</div>
            <div class="kpi-value">{active_report.total_columns}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Issues</div>
            <div class="kpi-value">{active_report.total_issues_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        crit_high = active_report.severity_counts.get('CRITICAL', 0) + active_report.severity_counts.get('HIGH', 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Critical / High Alerts</div>
            <div class="kpi-value" style="color: {'#F87171' if crit_high > 0 else '#34D399'};">{crit_high}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Executive Summary Alerts Banner
    alert_header_col1, alert_header_col2 = st.columns([3, 1])
    with alert_header_col1:
        st.markdown("### ⚠️ Executive Summary Alerts")
    with alert_header_col2:
        summary_text = f"DATA QUALITY SCORE: {score:.1f}/100 (Grade {grade})\nDataset: {report.dataset_name}\n" + "\n".join(active_report.summary_alerts)
        if st.button("📋 Copy Executive Summary", use_container_width=True):
            st.toast("Summary formatted for sharing!", icon="📋")
            st.code(summary_text, language="text")

    for alert in active_report.summary_alerts:
        if alert.startswith("✓"):
            st.markdown(f"<div class='alert-card-success'>✓ {alert.replace('✓', '').strip()}</div>", unsafe_allow_html=True)
        else:
            clean_text = alert.replace("⚠", "").replace("[!]", "").replace("🚨", "").strip()
            st.markdown(f"<div class='alert-card-warning'>⚠️ {clean_text}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Enterprise Navigation Tabs
    tab_audit, tab_clean, tab_ml, tab_drift, tab_rules, tab_doctor, tab_export = st.tabs([
        "🔍 9 Quality Checkers & Profiler",
        "✨ 1-Click Auto-Clean",
        "🚀 ML Baseline Benchmark",
        "🌊 Data Drift Scanner",
        "📐 Custom Business Rules",
        "🩺 AI Data Doctor",
        "🛠️ CI/CD & Developer Hub"
    ])

    # ---------------------------------------------------------
    # Tab 1: 9 Quality Checkers & Profiler
    # ---------------------------------------------------------
    with tab_audit:
        st.markdown("#### Inspection Breakdown Across All 9 Quality Modules")
        for cid, cr in active_report.checker_results.items():
            status_emoji = "✅" if cr.passed else ("🔴" if cr.score < 60 else "⚠️")
            with st.expander(f"{status_emoji} **{cr.checker_name}** — Score: {cr.score:.1f}% ({len(cr.issues)} issues)", expanded=not cr.passed):
                st.write(f"**Summary:** {cr.summary}")
                
                if cr.metrics:
                    m_cols = st.columns(len(cr.metrics))
                    for i, (k, v) in enumerate(cr.metrics.items()):
                        if not isinstance(v, (dict, list)):
                            formatted_k = k.replace("_", " ").title()
                            with m_cols[i % len(m_cols)]:
                                st.metric(label=formatted_k, value=str(v))

                if cr.issues:
                    st.markdown("**Issues & Recommended Remediation:**")
                    for issue in cr.issues:
                        sev_class = f"sev-{issue.severity.value.lower()}"
                        st.markdown(f"<span class='sev-badge {sev_class}'>{issue.severity.value}</span> **{issue.title}**", unsafe_allow_html=True)
                        st.caption(issue.description)
                        st.info(f"💡 **Action:** {issue.remediation_suggestion}")
                        if issue.suggested_code_snippet:
                            st.code(issue.suggested_code_snippet, language="python")
                else:
                    st.success("All checks in this category passed with zero anomalies!")

        st.markdown("---")
        st.markdown("#### 📊 Statistical Column Profiler & Distribution")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            col_search = st.text_input("🔍 Search column names...", value="", placeholder="Type feature name...")
        with f_col2:
            type_filter = st.selectbox("Filter Type", ["All Types", "Numeric", "Categorical", "Datetime"])
        with f_col3:
            issues_only = st.checkbox("Show Only Columns with Issues", value=False)

        filtered_profs = []
        for cname, prof in active_report.column_profiles.items():
            matches_search = col_search.lower() in cname.lower()
            matches_type = type_filter == "All Types" or type_filter.lower() in prof.inferred_type.lower()
            matches_issues = not issues_only or prof.issues_count > 0
            if matches_search and matches_type and matches_issues:
                filtered_profs.append(prof)

        prof_data = [{
            "Column": prof.name,
            "Inferred Type": prof.inferred_type,
            "Physical Dtype": prof.physical_dtype,
            "Missing (%)": f"{prof.missing_count} ({prof.missing_percentage}%)",
            "Unique Ratio": f"{prof.unique_count} ({prof.unique_percentage}%)",
            "Issues Flagged": prof.issues_count,
            "Is Target": "YES" if prof.is_target else "No"
        } for prof in filtered_profs]
        st.dataframe(pd.DataFrame(prof_data), use_container_width=True)

        st.markdown("##### 🔬 Interactive Feature Visualizer (Histogram & Boxplots)")
        available_cols = [p.name for p in filtered_profs] if filtered_profs else list(active_report.column_profiles.keys())
        selected_col_name = st.selectbox("Select Feature:", available_cols)
        
        if selected_col_name:
            col_prof = active_report.column_profiles[selected_col_name]
            active_df = st.session_state.cleaned_df if is_cleaned else df
            st.plotly_chart(create_column_distribution_chart(active_df, selected_col_name, col_prof.inferred_type), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔥 Feature Correlation Matrix")
        corr_checker = active_report.checker_results.get("high_correlation")
        if corr_checker and corr_checker.visual_data and "correlation_matrix" in corr_checker.visual_data:
            st.plotly_chart(create_correlation_heatmap(corr_checker.visual_data["correlation_matrix"]), use_container_width=True)

    # ---------------------------------------------------------
    # Tab 2: 1-Click Auto-Remediation
    # ---------------------------------------------------------
    with tab_clean:
        st.markdown("#### ✨ 1-Click Automated Data Cleaning Wizard")
        st.caption("Apply smart imputation, outlier clipping, sentinel sanitization, and leakage removal.")

        fixable_actions = []
        seen_fix_actions = set()
        for r in report.recommended_actions:
            if r.get("auto_fixable") and r.get("fix_action"):
                fa = r.get("fix_action")
                if fa not in seen_fix_actions:
                    seen_fix_actions.add(fa)
                    fixable_actions.append(r)

        if fixable_actions:
            t_col1, t_col2 = st.columns([3, 1])
            with t_col1:
                st.write(f"**Available Remediations ({len(fixable_actions)} actions):**")
            with t_col2:
                select_all = st.checkbox("Select / Deselect All", value=True, key="clean_select_all_toggle")

            selected_actions = []
            for idx, rec in enumerate(fixable_actions):
                checked = st.checkbox(
                    f"**{rec['title']}** — {rec['action']}",
                    value=select_all,
                    key=f"fix_action_checkbox_{idx}_{rec['fix_action']}"
                )
                if checked:
                    selected_actions.append(rec['fix_action'])

            if st.button("⚡ Execute Selected Cleaning Actions", type="primary", disabled=len(selected_actions) == 0, key="btn_exec_clean"):
                with st.spinner("Applying automated data cleaning transformations..."):
                    cleaner = DatasetCleaner()
                    cleaned_df, stats = cleaner.clean(
                        df=df,
                        selected_actions=selected_actions,
                        target_col=report.target_column,
                        report=report
                    )
                    engine = AuditEngine()
                    cleaned_report = engine.audit(
                        cleaned_df,
                        dataset_name=f"{report.dataset_name} (Cleaned)",
                        target_col=report.target_column
                    )
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.cleaned_report = cleaned_report
                    st.session_state.clean_stats = stats
                    st.session_state.benchmark_result = None
                    st.toast("🎉 Remediated dataset and improved score!", icon="✨")
                    st.rerun()

            if st.session_state.cleaned_df is not None:
                c_stats = st.session_state.clean_stats
                c_rep = st.session_state.cleaned_report
                st.success(f"🎉 Cleaning Complete! Score elevated from **{report.overall_score:.1f}** (Grade {report.grade}) ➡️ **{c_rep.overall_score:.1f}** (Grade {c_rep.grade})!")
                st.plotly_chart(create_comparison_bar_chart(report, c_rep), use_container_width=True)

                st.markdown("##### 📥 Download Remediated Dataset:")
                d1, d2 = st.columns(2)
                with d1:
                    csv_buffer = io.StringIO()
                    st.session_state.cleaned_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📄 Download Cleaned CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{report.dataset_name.replace(' ', '_').lower()}_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        st.session_state.cleaned_df.to_excel(writer, index=False)
                    st.download_button(
                        label="📊 Download Cleaned Excel (.xlsx)",
                        data=excel_buffer.getvalue(),
                        file_name=f"{report.dataset_name.replace(' ', '_').lower()}_cleaned.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        else:
            st.success("No critical auto-remediation actions needed! Dataset is already clean.")

    # ---------------------------------------------------------
    # Tab 3: ML Baseline Benchmark (Highest Impact)
    # ---------------------------------------------------------
    with tab_ml:
        st.markdown("#### 🚀 Automated ML Baseline Benchmark (Dirty vs. Clean Uplift)")
        st.caption("Quantifies the exact performance ROI of data quality remediation by training baseline models.")

        if not report.target_column:
            st.warning("⚠️ Please specify a **Target Column** in the sidebar or upload screen to run the ML Baseline Benchmark.")
            available_cols = list(df.columns)
            quick_target = st.selectbox("Or choose a Target Column now:", available_cols, key="quick_bench_target")
            if st.button("Set Target and Run Benchmark"):
                st.session_state.target_col = quick_target
                engine = AuditEngine()
                st.session_state.report = engine.audit(df, dataset_name=report.dataset_name, target_col=quick_target)
                st.rerun()
        else:
            m_col1, m_col2 = st.columns([3, 1])
            with m_col1:
                model_choice = st.selectbox(
                    "Select Baseline Estimator Architecture:",
                    ["Random Forest (Ensemble)", "Logistic / Ridge Linear Baseline", "Decision Tree"],
                    key="bench_model_choice"
                )
            with m_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                run_bench_btn = st.button("⚡ Run ML Benchmark", type="primary", use_container_width=True)

            model_map = {
                "Random Forest (Ensemble)": "random_forest",
                "Logistic / Ridge Linear Baseline": "logistic_regression",
                "Decision Tree": "decision_tree"
            }

            if run_bench_btn or st.session_state.benchmark_result is not None:
                if run_bench_btn:
                    with st.spinner("Training baseline models on Raw vs. Remediated data..."):
                        # Ensure cleaned DF exists
                        if st.session_state.cleaned_df is None:
                            cleaner = DatasetCleaner()
                            clean_df, _ = cleaner.clean(df, target_col=report.target_column, report=report)
                        else:
                            clean_df = st.session_state.cleaned_df

                        benchmarker = MLBenchmarkEngine()
                        b_res = benchmarker.run_benchmark(
                            raw_df=df,
                            clean_df=clean_df,
                            target_col=report.target_column,
                            model_type=model_map[model_choice]
                        )
                        st.session_state.benchmark_result = b_res

                b_res = st.session_state.benchmark_result
                if b_res:
                    st.success(f"🎯 **Benchmark Result ({b_res.task_type.upper()})**: {b_res.summary_text}")
                    
                    # Metric cards
                    if b_res.task_type == "classification":
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        with mc1:
                            st.metric("Accuracy (Cleaned)", f"{b_res.clean_model.metrics.get('accuracy', 0)*100:.1f}%", f"{b_res.uplift.get('accuracy', 0)*100:+.1f}%")
                        with mc2:
                            st.metric("F1-Score (Cleaned)", f"{b_res.clean_model.metrics.get('f1', 0):.3f}", f"{b_res.uplift.get('f1', 0):+.3f}")
                        with mc3:
                            st.metric("Precision", f"{b_res.clean_model.metrics.get('precision', 0):.3f}", f"{b_res.uplift.get('precision', 0):+.3f}")
                        with mc4:
                            st.metric("ROC-AUC", f"{b_res.clean_model.metrics.get('roc_auc', 0):.3f}", f"{b_res.uplift.get('roc_auc', 0):+.3f}")
                    else:
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("R² Fit Score", f"{b_res.clean_model.metrics.get('r2', 0):.3f}", f"{b_res.uplift.get('r2', 0):+.3f}")
                        with mc2:
                            st.metric("MAE Error", f"{b_res.clean_model.metrics.get('mae', 0):.2f}", f"{b_res.uplift.get('mae', 0):+.2f}")
                        with mc3:
                            st.metric("RMSE Error", f"{b_res.clean_model.metrics.get('rmse', 0):.2f}", f"{b_res.uplift.get('rmse', 0):+.2f}")

                    # Top Feature Importances
                    if b_res.top_features:
                        st.markdown("##### 🌲 Top Predictive Features (Cleaned Model):")
                        feat_df = pd.DataFrame(b_res.top_features, columns=["Feature", "Importance"])
                        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Teal")
                        fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", font=dict(color="#F8FAFC"), height=300)
                        st.plotly_chart(fig_feat, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 4: Train vs. Test & Temporal Data Drift Scanner
    # ---------------------------------------------------------
    with tab_drift:
        st.markdown("#### 🌊 Train vs. Test & Temporal Data Drift Scanner")
        st.caption("Upload a Reference (Train/Baseline) dataset and a Current (Test/Target) dataset to evaluate distribution shifts.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("##### 1️⃣ Baseline / Reference Dataset (Train)")
            ref_file = st.file_uploader("Upload Reference File (.csv)", type=["csv", "xlsx"], key="drift_ref_file")
            ref_choice = st.selectbox("Or use current active dataset as Reference:", ["Active Uploaded Dataset", "Loan Default Benchmark", "Telecom Churn Benchmark"], key="drift_ref_preset")
        with d_col2:
            st.markdown("##### 2️⃣ Target / Current Dataset (Test)")
            curr_file = st.file_uploader("Upload Current / Test File (.csv)", type=["csv", "xlsx"], key="drift_curr_file")
            curr_choice = st.selectbox("Or use preset dirty variant as Target:", ["Active Cleaned Dataset (if available)", "Telecom Churn Benchmark", "Loan Default Benchmark"], key="drift_curr_preset")

        if st.button("🌊 Run Statistical Drift Analysis", type="primary", use_container_width=True):
            try:
                # Load Reference DF
                if ref_file is not None:
                    ref_df = load_uploaded_dataset(ref_file)
                    r_name = ref_file.name
                elif ref_choice == "Active Uploaded Dataset":
                    ref_df = df
                    r_name = report.dataset_name
                else:
                    ref_df = pd.read_csv(os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv"))
                    r_name = "Loan Baseline"

                # Load Current DF
                if curr_file is not None:
                    curr_df = load_uploaded_dataset(curr_file)
                    c_name = curr_file.name
                elif curr_choice == "Active Cleaned Dataset (if available)" and st.session_state.cleaned_df is not None:
                    curr_df = st.session_state.cleaned_df
                    c_name = f"{report.dataset_name} (Cleaned)"
                else:
                    curr_df = pd.read_csv(os.path.join(SAMPLES_DIR, "customer_churn_leaky.csv"))
                    c_name = "Churn Test Data"

                detector = DriftDetector()
                d_report = detector.compare_datasets(ref_df, curr_df, ref_name=r_name, curr_name=c_name)
                st.session_state.drift_report = (d_report, ref_df, curr_df)
                st.toast("Drift analysis complete!", icon="🌊")
            except Exception as e:
                st.error(f"Error computing dataset drift: {e}")

        if st.session_state.drift_report:
            d_report, ref_df_cached, curr_df_cached = st.session_state.drift_report
            st.markdown(f"### 🛡️ Drift Status: `{d_report.overall_status}` (Drift Index: {d_report.overall_drift_score:.1f}%)")
            
            for a in d_report.summary_alerts:
                st.markdown(f"<div class='alert-card-warning'>{a}</div>", unsafe_allow_html=True)

            # Column drift breakdown table
            d_rows = [{
                "Column": col,
                "Type": res.column_type,
                "Status": res.drift_status,
                "PSI Score": f"{res.psi_score:.3f}" if res.psi_score is not None else "N/A",
                "KS Stat (p-val)": f"{res.ks_statistic:.3f} (p={res.ks_p_value:.2f})" if res.ks_statistic is not None else "N/A",
                "Novel Categories": ", ".join(res.novel_categories) if res.novel_categories else "None",
                "Details": res.details
            } for col, res in d_report.column_reports.items()]
            st.dataframe(pd.DataFrame(d_rows), use_container_width=True)

    # ---------------------------------------------------------
    # Tab 5: Custom Business Rules (Assertions Engine)
    # ---------------------------------------------------------
    with tab_rules:
        st.markdown("#### 📐 Visual Custom Business Rule Builder (Data Assertions)")
        st.caption("Define strict business logic, regex patterns, allowed values, and range constraints.")

        # Preset Templates
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("💳 Load FinTech & Lending Rules"):
                st.session_state.custom_rules = RulesEngine.get_template_rules("fintech", df)
                st.toast("Loaded FinTech template rules!", icon="💳")
        with p_col2:
            if st.button("🛒 Load E-Commerce & Retail Rules"):
                st.session_state.custom_rules = RulesEngine.get_template_rules("ecommerce", df)
                st.toast("Loaded E-Commerce template rules!", icon="🛒")
        with p_col3:
            if st.button("🧼 Load Data Hygiene Rules"):
                st.session_state.custom_rules = RulesEngine.get_template_rules("hygiene", df)
                st.toast("Loaded Data Hygiene rules!", icon="🧼")

        st.markdown("##### ➕ Add Custom Assertion:")
        with st.expander("Create New Rule Definition", expanded=False):
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                new_col = st.selectbox("Target Feature:", list(df.columns), key="rule_new_col")
                new_type = st.selectbox("Rule Type:", ["range", "not_null", "allowed_values", "regex", "unique", "comparison"], key="rule_new_type")
            with r_col2:
                new_desc = st.text_input("Rule Description:", f"Constraint on {new_col}", key="rule_new_desc")
                new_sev = st.selectbox("Severity:", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], key="rule_new_sev")
            with r_col3:
                min_v = st.number_value = st.number_input("Min Value (Range):", value=0.0, key="rule_min") if new_type == "range" else None
                max_v = st.number_input("Max Value (Range):", value=100.0, key="rule_max") if new_type == "range" else None
                regex_pat = st.text_input("Regex Pattern:", r"^[\w\.-]+@[\w\.-]+\.\w+$", key="rule_regex") if new_type == "regex" else None

            if st.button("Add Assertion to Suite", type="secondary"):
                new_r = RuleDefinition(
                    rule_id=f"rule_{len(st.session_state.custom_rules)+1}",
                    rule_type=new_type,
                    column=new_col,
                    description=new_desc,
                    severity=new_sev,
                    min_value=min_v,
                    max_value=max_v,
                    regex_pattern=regex_pat
                )
                st.session_state.custom_rules.append(new_r)
                st.toast(f"Added rule: '{new_desc}'", icon="➕")

        if st.session_state.custom_rules:
            st.markdown(f"##### 📋 Active Rule Suite ({len(st.session_state.custom_rules)} rules):")
            for idx, r in enumerate(st.session_state.custom_rules):
                st.write(f"- `{r.severity}` **[{r.rule_type.upper()}]** `{r.column}`: {r.description}")

            if st.button("▶ Run All Custom Assertions", type="primary"):
                r_engine = RulesEngine()
                st.session_state.rules_result = r_engine.run_suite(df, st.session_state.custom_rules)

            if st.session_state.rules_result:
                s_res = st.session_state.rules_result
                st.markdown(f"### Overall Pass Rate: `{s_res.overall_pass_rate}%` ({s_res.passed_rules}/{s_res.total_rules} Passed)")
                for r_res in s_res.results:
                    status_badge = "✅" if r_res.status == "PASSED" else "❌"
                    with st.expander(f"{status_badge} **{r_res.rule.description}** — {r_res.pass_rate_pct}% Pass Rate", expanded=(r_res.status == "FAILED")):
                        st.write(r_res.message)
                        if r_res.sample_violations:
                            st.markdown("**Sample Violating Rows:**")
                            st.dataframe(pd.DataFrame(r_res.sample_violations), use_container_width=True)

    # ---------------------------------------------------------
    # Tab 6: AI Data Doctor & Executive Briefing
    # ---------------------------------------------------------
    with tab_doctor:
        st.markdown("#### 🩺 AI Data Doctor — Executive Briefing & Diagnostic Engine")
        st.caption("Synthesizes multi-dimension audit findings into plain-English business & engineering diagnostics.")

        doctor = DataDoctor()
        diag = doctor.diagnose(report)

        st.markdown(f"""
        <div class="kpi-card" style="border-left: 6px solid #14B8A6; margin-bottom: 1.5rem;">
            <div class="kpi-label">Health Status & Risk Meter</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="kpi-value" style="font-size: 1.5rem;">{diag.overall_health_badge}</div>
                <div style="font-weight: 800; font-size: 1.1rem; color: #38BDF8;">Risk Index: {diag.risk_score}/100</div>
            </div>
            <p style="color: #E2E8F0; margin-top: 0.75rem; font-size: 0.95rem;">{diag.executive_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        doc_c1, doc_c2 = st.columns(2)
        with doc_c1:
            st.markdown("##### 💼 Business & Downstream ML Impact")
            st.write(diag.business_impact)
            st.markdown("##### ⚖️ Data Governance & Compliance Assessment")
            for c in diag.compliance_risks:
                st.write(f"- {c}")

        with doc_c2:
            st.markdown("##### 🛠️ Top Root Cause Failure Vectors")
            for rc in diag.root_cause_vectors:
                st.write(f"- {rc}")
            st.markdown("##### 🗺️ Prioritized Engineering Roadmap")
            for step in diag.remediation_roadmap:
                st.write(f"- {step}")

        st.markdown("---")
        st.markdown("##### 🤖 Ask the AI Data Doctor (Gemini Generative Assistant)")
        gemini_key = st.text_input("Enter Google Gemini API Key (Optional):", type="password", key="gemini_key_input")
        user_doc_q = st.text_input("Ask a question about this dataset:", "How will the identified missing values and outliers impact an XGBoost model?", key="gemini_q_input")

        if st.button("💬 Consult Data Doctor AI", type="secondary"):
            if gemini_key:
                with st.spinner("Consulting Gemini AI with dataset audit telemetry..."):
                    ai_reply = doctor.ask_gemini(user_doc_q, report, gemini_key)
                    st.info(ai_reply)
            else:
                st.info(f"💡 **Data Doctor Diagnostic Reasoning:** Based on audit of '{report.dataset_name}', the primary bottlenecks are missing values and sentinel outliers. Gradient boosting models like XGBoost can handle some missing values via default split directions, but dirty sentinel tokens (e.g. -999) severely distort split boundaries. Run 1-Click Auto-Remediation in Tab 2 before model fitting.")

    # ---------------------------------------------------------
    # Tab 7: CI/CD & Developer Hub
    # ---------------------------------------------------------
    with tab_export:
        st.markdown("#### 🛠️ CI/CD Quality Gate, CLI Commands & Report Center")
        st.caption("Automate data quality gates in GitHub Actions, GitLab CI, or Airflow pipelines.")

        st.markdown("##### 💻 Developer CLI Commands:")
        st.code(f"""
# 1. Run automated quality gate audit (fails build if score < 80.0)
python cli.py audit {report.dataset_name}.csv --min-score 80.0 --fail-on-leakage --export-html report.html

# 2. Check statistical data drift between Train and Test datasets
python cli.py drift --reference train.csv --current test.csv --max-psi 0.25

# 3. Enforce custom business rules
python cli.py rules dataset.csv --rules-file rules.json --min-pass-rate 95.0

# 4. 1-Click Auto-Remediate in terminal
python cli.py clean dirty_data.csv --output cleaned_data.csv
""", language="bash")

        st.markdown("##### 🐙 Copyable GitHub Actions CI/CD Workflow (`.github/workflows/data_quality_gate.yml`):")
        gh_workflow = """name: "Dataset Quality Gate"
on: [push, pull_request]
jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python cli.py audit data.csv --min-score 80.0 --export-html audit_report.html
      - uses: actions/upload-artifact@v4
        with:
          name: data-quality-report
          path: audit_report.html
"""
        st.code(gh_workflow, language="yaml")

        st.markdown("##### 📑 Export Audit Artifacts:")
        e1, e2, e3 = st.columns(3)
        with e1:
            html_rep = ReportGenerator.generate_html_report(report, st.session_state.cleaned_report)
            st.download_button(
                label="🌐 Interactive HTML Report",
                data=html_rep,
                file_name=f"{report.dataset_name.replace(' ', '_').lower()}_report.html",
                mime="text/html",
                use_container_width=True
            )
        with e2:
            py_pipe = ReportGenerator.generate_python_script(report)
            st.download_button(
                label="🐍 Python Cleaning Pipeline",
                data=py_pipe,
                file_name=f"clean_{report.dataset_name.replace(' ', '_').lower()}_pipeline.py",
                mime="text/x-python",
                use_container_width=True
            )
        with e3:
            json_rep = ReportGenerator.generate_json_report(report)
            st.download_button(
                label="📦 JSON Audit Schema",
                data=json_rep,
                file_name=f"{report.dataset_name.replace(' ', '_').lower()}_audit.json",
                mime="application/json",
                use_container_width=True
            )
