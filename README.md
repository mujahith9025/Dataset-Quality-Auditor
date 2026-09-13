# 🛡️ Dataset Quality Auditor 2.0

> **Enterprise Tabular Health Scanner, ML Baseline Benchmarking, Distribution Drift Detector, Custom Rule Assertions & AI Data Doctor**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dataset-quality-auditor-htzg4xsmh5qmnb433ypd3e.streamlit.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B.svg)](https://dataset-quality-auditor-htzg4xsmh5qmnb433ypd3e.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](tests/)

---

### 🌐 Live Cloud Application
🚀 **Try the live web app in your browser:**  
👉 **[https://dataset-quality-auditor-htzg4xsmh5qmnb433ypd3e.streamlit.app/](https://dataset-quality-auditor-htzg4xsmh5qmnb433ypd3e.streamlit.app/)**

---

## 🌟 Overview

**Dataset Quality Auditor 2.0** is an end-to-end data validation and auto-remediation platform built for Data Scientists, Machine Learning Engineers, and Data Analysts. It automatically audits tabular datasets across **5 quality dimensions**, executes **9 automated quality checks**, measures downstream **ML performance ROI (Accuracy/F1 uplift)**, catches **Train-vs-Test distribution drift (PSI & KS-tests)**, enforces **custom business rules**, and generates executive briefings with an **AI Data Doctor**.

```
                      ┌──────────────────────────────────────────────────────────┐
                      │               DATASET QUALITY AUDITOR 2.0                │
                      └────────────────────────────┬─────────────────────────────┘
                                                   │
    ┌──────────────────────┬───────────────────────┼──────────────────────┬──────────────────────┐
    ▼                      ▼                       ▼                      ▼                      ▼
┌──────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ 1. 9 Quality     │ │ 2. Automated ML   │ │ 3. Train vs Test  │ │ 4. Custom Rules   │ │ 5. CI/CD Quality  │
│    Checkers &    │ │    Baseline       │ │    Data Drift     │ │    Assertions     │ │    Gate & CLI     │
│    Auto-Clean    │ │    Benchmark      │ │    (PSI & KS)     │ │    Engine         │ │    Workflow       │
└──────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘ └───────────────────┘
```

---

## 🚀 Key Features

### 1. 🔍 Comprehensive Quality Audit (0–100 Score & 5 Dimensions)
- **Completeness (25%)**: Missing values, sparse columns, systemic missingness.
- **Uniqueness (15%)**: Duplicate records, repeated candidate keys.
- **Validity (20%)**: Outliers (IQR + Z-score bounds), corrupted database sentinels (`-999`, `9999`, `N/A`, `NULL`).
- **Consistency (15%)**: Currency/numeric strings stored as text (`"$120,000"`, `"15.5%"`), constant columns.
- **ML Readiness (25%)**: Severe class skew (imbalance ratio), target data leakage (`|r| >= 0.85`), and multicollinearity.

### 2. ✨ 1-Click Automated Data Remediation
- Imputes missing values using median/mode strategies.
- Winsorizes/clips extreme distribution outliers.
- Sanitizes sentinel tokens to proper nulls.
- Strips currency/percentage symbols and parses true physical data types.
- Prunes leaky target features and constant columns.
- **Exports Cleaned Data**: Instant 1-click download as `.csv` or `.xlsx`.

### 3. 🚀 Automated ML Baseline Benchmark (Uplift Quantification)
- Trains baseline models (`RandomForest`, `LogisticRegression`, `Ridge`, `DecisionTree`) on **Raw/Dirty Data** vs. **Cleaned Data**.
- Quantifies exact performance uplift: **Accuracy**, **F1-Score**, **Precision**, **Recall**, **ROC-AUC**, and **R²**.
- Displays feature importance bar charts from the trained models.

### 4. 🌊 Train vs. Test & Temporal Data Drift Scanner
- Compares **Reference (Train)** and **Current (Test/Production)** datasets.
- Computes **Population Stability Index (PSI)** with standard risk thresholds (`<0.1` Stable, `0.1-0.25` Moderate, `>0.25` Critical Drift).
- Evaluates **Kolmogorov-Smirnov (KS) Two-Sample Tests** with asymptotic p-values.
- Flags **Novel / Unseen Categories** in test data that would cause production inference crashes.

### 5. 📐 Custom Business Rules (Declarative Assertions)
- Visual assertion builder supporting `range`, `not_null`, `allowed_values`, `regex`, `unique`, and `comparison` constraints.
- Pre-built templates for **FinTech / Lending**, **E-Commerce**, and **Data Hygiene**.
- Displays pass rates, violation counts, and sample failing rows for immediate drilldown.

### 6. 🩺 AI "Data Doctor" Executive Diagnostic Engine
- Offline heuristic briefing summarizing business risk, compliance liability, and root causes.
- Prioritized engineering remediation roadmap.
- Optional **Google Gemini API** integration for interactive natural language queries.

### 7. 🛠️ CI/CD Quality Gate & Developer CLI
- Complete command-line interface with exit codes (`0` on pass, `2` on gate failure).
- Ready-to-use GitHub Actions workflow file (`.github/workflows/data_quality_gate.yml`).

---

## 📦 Installation & Quickstart

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/dataset-quality-auditor.git
cd dataset-quality-auditor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
*The app will automatically open at `http://localhost:8501`.*

---

## 💻 CLI Usage

```bash
# 1. Audit dataset quality and enforce a minimum score of 80
python cli.py audit samples/loan_approval_dirty.csv --target loan_status --fail-under 80.0 --export-html report.html

# 2. Check for data drift between Train and Test datasets
python cli.py drift --reference train.csv --current test.csv --max-psi 0.25

# 3. Enforce custom business rules from a JSON schema
python cli.py rules dataset.csv --rules-file rules.json --min-pass-rate 95.0

# 4. Auto-remediate dataset defects and save cleaned output
python cli.py clean dirty_data.csv --output cleaned_data.csv --target loan_status
```

---

## 🧪 Running Tests

Execute the comprehensive test suite covering all 9 checkers, ML benchmark, drift detector, and rules engine:

```bash
pytest tests/
```

---

## 📁 Project Structure

```
dataset-quality-auditor/
├── .github/
│   └── workflows/
│       └── data_quality_gate.yml  # GitHub Actions CI/CD Quality Gate
├── .streamlit/
│   └── config.toml                # Streamlit dark emerald theme config
├── core/
│   ├── checkers/                  # 9 specialized audit modules
│   ├── engine.py                  # Audit orchestrator
│   ├── scoring.py                 # 5-dimension scoring math (0-100)
│   ├── cleaner.py                 # 17 automated cleaning transforms
│   ├── ml_benchmark.py            # Automated ML training & uplift scoring
│   ├── drift_detector.py          # KS-tests, PSI, and novel category drift
│   ├── rules_engine.py            # Declarative business assertions runner
│   ├── data_doctor.py             # Executive diagnostic synthesizer & Gemini AI
│   └── report_generator.py        # Interactive HTML, Python & JSON exporters
├── models/
│   └── schemas.py                 # Pydantic data schemas
├── samples/                       # Realistic benchmark test datasets
│   ├── loan_approval_dirty.csv
│   ├── customer_churn_leaky.csv
│   └── medical_patient_anomalous.csv
├── tests/
│   └── test_auditor.py            # Automated test suite
├── app.py                         # Main Streamlit Web Application
├── cli.py                         # Developer CLI tool
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
