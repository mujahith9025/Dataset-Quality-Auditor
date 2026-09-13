"""
Comprehensive Test Suite for Dataset Quality Auditor 2.0
Tests the 9 Audit Checkers, Weighted Scoring, Auto-Remediation, ML Benchmark Uplift,
Data Drift Detector, Custom Business Rules Engine, AI Data Doctor, and Security Edge Cases.
"""

import os
import pandas as pd
import numpy as np

from core.engine import AuditEngine
from core.cleaner import DatasetCleaner
from core.report_generator import ReportGenerator
from core.ml_benchmark import MLBenchmarkEngine
from core.drift_detector import DriftDetector, calculate_psi, ks_2samp_test
from core.rules_engine import RulesEngine, RuleDefinition
from core.data_doctor import DataDoctor

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_core_audit_and_clean():
    loan_path = os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv")
    assert os.path.exists(loan_path)

    df = pd.read_csv(loan_path)
    engine = AuditEngine()
    report = engine.audit(df, dataset_name="Loan Test", target_col="loan_status")

    assert 60.0 <= report.overall_score <= 80.0
    assert len(report.checker_results) == 9
    assert len(report.summary_alerts) > 0

    # Clean
    cleaner = DatasetCleaner()
    cleaned_df, stats = cleaner.clean(df, report=report, target_col="loan_status")
    cleaned_report = engine.audit(cleaned_df, target_col="loan_status")

    assert cleaned_report.overall_score > report.overall_score
    assert cleaned_report.overall_score >= 88.0

    # HTML / Python Exporters
    html = ReportGenerator.generate_html_report(report)
    assert len(html) > 1000
    py_script = ReportGenerator.generate_python_script(report)
    assert "def clean_dataset" in py_script


def test_ml_baseline_benchmark():
    loan_path = os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv")
    df = pd.read_csv(loan_path)

    engine = AuditEngine()
    report = engine.audit(df, target_col="loan_status")
    cleaner = DatasetCleaner()
    cleaned_df, _ = cleaner.clean(df, report=report, target_col="loan_status")

    benchmarker = MLBenchmarkEngine()
    res = benchmarker.run_benchmark(df, cleaned_df, target_col="loan_status", model_type="random_forest")

    assert res.task_type == "classification"
    assert "accuracy" in res.raw_model.metrics
    assert "f1" in res.clean_model.metrics
    assert len(res.summary_text) > 20
    assert len(res.top_features) > 0


def test_drift_detector():
    np.random.seed(42)
    # Baseline
    train_data = pd.DataFrame({
        "age": np.random.normal(35, 5, 200),
        "income": np.random.exponential(50000, 200),
        "category": np.random.choice(["A", "B", "C"], 200)
    })
    # Target with heavy drift in age and novel category "D"
    test_data = pd.DataFrame({
        "age": np.random.normal(55, 10, 200),
        "income": np.random.exponential(51000, 200),
        "category": np.random.choice(["A", "B", "D"], 200)
    })

    detector = DriftDetector()
    report = detector.compare_datasets(train_data, test_data)

    assert report.shared_columns_count == 3
    assert report.column_reports["age"].drift_status in ["MODERATE_DRIFT", "CRITICAL_DRIFT"]
    assert "D" in report.column_reports["category"].novel_categories
    assert len(report.summary_alerts) > 0


def test_rules_engine():
    df = pd.DataFrame({
        "age": [25, 45, 12, 105, 30],
        "email": ["alice@work.com", "bob@gmail.com", "invalid-email", "carol@yahoo.com", "dave@corp.org"],
        "status": ["APPROVED", "REJECTED", "PENDING", "UNKNOWN_STATUS", "APPROVED"]
    })

    rules = [
        RuleDefinition(
            rule_id="r1",
            rule_type="range",
            column="age",
            description="Age must be 18-100",
            min_value=18,
            max_value=100
        ),
        RuleDefinition(
            rule_id="r2",
            rule_type="regex",
            column="email",
            description="Valid email regex",
            regex_pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"
        ),
        RuleDefinition(
            rule_id="r3",
            rule_type="allowed_values",
            column="status",
            description="Valid status domain",
            allowed_values=["APPROVED", "REJECTED", "PENDING"]
        )
    ]

    engine = RulesEngine()
    res = engine.run_suite(df, rules)

    assert res.total_rules == 3
    assert res.failed_rules == 3  # Each has 1-2 invalid rows
    assert res.results[0].failed_count == 2  # age 12 and 105


def test_data_doctor():
    loan_path = os.path.join(SAMPLES_DIR, "loan_approval_dirty.csv")
    df = pd.read_csv(loan_path)
    engine = AuditEngine()
    report = engine.audit(df, target_col="loan_status")

    doctor = DataDoctor()
    diag = doctor.diagnose(report)

    assert diag.risk_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(diag.root_cause_vectors) > 0
    assert len(diag.remediation_roadmap) > 0


def test_security_xss_and_html_escaping():
    # Inject XSS payload into dataset name and column name
    malicious_df = pd.DataFrame({
        "<script>alert(1)</script>": [1, 2, 3],
        "safe_col": ["a", "b", "<img src=x onerror=alert('xss')>"]
    })
    engine = AuditEngine()
    report = engine.audit(malicious_df, dataset_name="<script>alert('xss')</script>")
    
    html_out = ReportGenerator.generate_html_report(report)
    assert "<script>alert('xss')</script>" not in html_out
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_out or "&lt;script&gt;" in html_out


def test_rules_engine_invalid_regex_safety():
    df = pd.DataFrame({"email": ["test@example.com", "bad"]})
    # Malformed regex
    bad_rule = RuleDefinition(
        rule_id="bad_reg",
        rule_type="regex",
        column="email",
        description="Bad regex",
        regex_pattern=r"([a-z+"  # unclosed parenthesis
    )
    engine = RulesEngine()
    res = engine.evaluate_rule(df, bad_rule)
    assert res.status == "FAILED"
    assert "Invalid regular expression" in res.message


def test_cleaner_comma_and_currency_parsing():
    df = pd.DataFrame({
        "revenue": ["$1,250,500.00", "€45,000", "12,000.50", "$99.99"],
        "target": [1, 0, 1, 0]
    })
    cleaner = DatasetCleaner()
    cleaned_df, stats = cleaner.clean(df, selected_actions=["clean_numeric_string:revenue"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["revenue"])
    assert cleaned_df["revenue"].iloc[0] == 1250500.0
    assert cleaned_df["revenue"].iloc[1] == 45000.0


def test_degenerate_and_empty_edge_cases():
    empty_df = pd.DataFrame()
    engine = AuditEngine()
    report = engine.audit(empty_df, dataset_name="Empty Test")
    assert report.total_rows == 0
    assert report.overall_score >= 70.0
