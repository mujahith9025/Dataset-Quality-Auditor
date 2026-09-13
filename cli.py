"""
Dataset Quality Auditor — Developer CLI & CI/CD Tool
Command-line interface for automating dataset quality checks, distribution drift detection,
custom business assertions, and automated cleaning in terminal workflows and CI/CD pipelines.
"""

import sys
import os
import argparse
import json
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import AuditEngine
from core.cleaner import DatasetCleaner
from core.report_generator import ReportGenerator
from core.ml_benchmark import MLBenchmarkEngine
from core.drift_detector import DriftDetector
from core.rules_engine import RulesEngine, RuleDefinition
from core.data_doctor import DataDoctor


def print_banner():
    banner = """
========================================================================
       DATASET QUALITY AUDITOR — ENTERPRISE QUALITY & DRIFT CLI           
========================================================================
"""
    print(banner)


def load_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    if file_path.endswith(".csv") or file_path.endswith(".txt"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    return pd.read_csv(file_path)


def audit_command(args):
    print_banner()
    df = load_file(args.file)
    dataset_name = args.name or os.path.basename(args.file)
    print(f"[*] Auditing {len(df):,} rows x {df.shape[1]} columns (Target: {args.target or 'None'})...\n")

    engine = AuditEngine()
    report = engine.audit(df, dataset_name=dataset_name, target_col=args.target)

    # Print Quality Score Card
    print("+" + "-"*70 + "+")
    print(f"|  DATA QUALITY SCORE: {report.overall_score:5.1f} / 100.0   (Grade: {report.grade:2})                       |")
    print("+" + "-"*70 + "+")

    # Print Dimensions
    print("\n--- 5-DIMENSION QUALITY PROFILE ---")
    for dim_name, ds in report.dimension_scores.items():
        bar_len = int(ds.score / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {dim_name:15} [{bar}] {ds.score:5.1f}%  (Weight: {int(ds.weight*100)}%, Grade: {ds.grade})")

    # Print Summary Alerts
    print("\n--- EXECUTIVE SUMMARY ALERTS ---")
    for alert in report.summary_alerts:
        safe_alert = alert.replace("⚠", "[!]").replace("✓", "[OK]").replace("🚨", "[CRITICAL]")
        print(f"  {safe_alert}")

    # Top Actions
    print(f"\n--- TOP RECOMMENDED ACTIONS ({len(report.recommended_actions)} total) ---")
    for rec in report.recommended_actions[:5]:
        print(f"  [{rec['severity']:8}] Step {rec['step']}: {rec['title']}")
        print(f"             Action: {rec['action']}")

    # Export Reports
    if args.export_html:
        html = ReportGenerator.generate_html_report(report)
        with open(args.export_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n[OK] Exported interactive HTML report to: {args.export_html}")

    if args.export_script:
        script = ReportGenerator.generate_python_script(report)
        with open(args.export_script, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"[OK] Exported Python cleaning pipeline to: {args.export_script}")

    if args.export_json:
        json_data = ReportGenerator.generate_json_report(report)
        with open(args.export_json, "w", encoding="utf-8") as f:
            f.write(json_data)
        print(f"[OK] Exported JSON audit log to: {args.export_json}")

    print("\n" + "="*72)
    # Check CI/CD Quality Gate Failures
    if args.fail_under and report.overall_score < args.fail_under:
        print(f"[CI GATE FAILED] Score {report.overall_score:.1f} is below required threshold of {args.fail_under}.")
        sys.exit(2)

    if args.fail_on_leakage and any("leakage" in a.lower() for a in report.summary_alerts):
        print(f"[CI GATE FAILED] Target data leakage detected in dataset.")
        sys.exit(3)


def drift_command(args):
    print_banner()
    print(f"[*] Ingesting Reference: {args.reference}")
    ref_df = load_file(args.reference)
    print(f"[*] Ingesting Current:   {args.current}")
    curr_df = load_file(args.current)

    detector = DriftDetector()
    drift_report = detector.compare_datasets(ref_df, curr_df, ref_name=args.reference, curr_name=args.current)

    print("+" + "-"*70 + "+")
    print(f"|  DATASET DRIFT INDEX: {drift_report.overall_drift_score:5.1f}%   (Status: {drift_report.overall_status:16})  |")
    print("+" + "-"*70 + "+")
    print(f"  Shared Columns: {drift_report.shared_columns_count} | Critical: {drift_report.critical_columns_count} | Moderate: {drift_report.moderate_columns_count} | Stable: {drift_report.stable_columns_count}")

    print("\n--- COLUMN DRIFT BREAKDOWN ---")
    for col, res in drift_report.column_reports.items():
        status_tag = "[CRITICAL]" if res.drift_status == "CRITICAL_DRIFT" else ("[WARNING]" if res.drift_status == "MODERATE_DRIFT" else "[OK]")
        print(f"  {status_tag:10} {col:20} -> {res.details}")

    if args.max_psi:
        high_psi = [c for c, r in drift_report.column_reports.items() if r.psi_score and r.psi_score > args.max_psi]
        if high_psi:
            print(f"\n[CI GATE FAILED] Columns exceeding max PSI threshold ({args.max_psi}): {', '.join(high_psi)}")
            sys.exit(2)


def rules_command(args):
    print_banner()
    df = load_file(args.file)
    print(f"[*] Loaded dataset: {args.file} ({len(df):,} rows)")

    if not os.path.exists(args.rules_file):
        print(f"[ERROR] Rules file not found: {args.rules_file}")
        sys.exit(1)

    with open(args.rules_file, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    rules = [RuleDefinition(**r) for r in rules_data]
    print(f"[*] Executing {len(rules)} custom business assertions...\n")

    engine = RulesEngine()
    suite_res = engine.run_suite(df, rules)

    print("+" + "-"*70 + "+")
    print(f"|  ASSERTION PASS RATE: {suite_res.overall_pass_rate:5.1f}%   ({suite_res.passed_rules}/{suite_res.total_rules} Passed - Status: {suite_res.status})        |")
    print("+" + "-"*70 + "+")

    for r_res in suite_res.results:
        status_tag = "[PASS]" if r_res.status == "PASSED" else "[FAIL]"
        print(f"  {status_tag} {r_res.rule.column:15} | {r_res.rule.description[:40]:40} | {r_res.message}")

    if args.min_pass_rate and suite_res.overall_pass_rate < args.min_pass_rate:
        print(f"\n[CI GATE FAILED] Pass rate {suite_res.overall_pass_rate:.1f}% is below required {args.min_pass_rate}%.")
        sys.exit(2)


def clean_command(args):
    print_banner()
    df = load_file(args.file)
    print(f"[*] Ingesting dataset: {args.file} ({len(df):,} rows x {df.shape[1]} cols)")

    engine = AuditEngine()
    initial_report = engine.audit(df, target_col=args.target)
    print(f"[*] Pre-Cleaning Quality Score: {initial_report.overall_score:.1f}/100 (Grade: {initial_report.grade})")

    cleaner = DatasetCleaner()
    cleaned_df, stats = cleaner.clean(df, report=initial_report, target_col=args.target)

    post_report = engine.audit(cleaned_df, target_col=args.target)
    print(f"[*] Post-Cleaning Quality Score: {post_report.overall_score:.1f}/100 (Grade: {post_report.grade})")
    print(f"[*] Score Improvement: +{post_report.overall_score - initial_report.overall_score:.1f} points")

    out_path = args.output
    if out_path.endswith(".parquet"):
        cleaned_df.to_parquet(out_path, index=False)
    elif out_path.endswith(".xlsx"):
        cleaned_df.to_excel(out_path, index=False)
    else:
        cleaned_df.to_csv(out_path, index=False)

    print(f"[OK] Cleaned dataset saved to: {out_path} ({len(cleaned_df):,} rows x {cleaned_df.shape[1]} cols)")


def main():
    parser = argparse.ArgumentParser(description="Dataset Quality Auditor & Drift CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Audit subparser
    audit_parser = subparsers.add_parser("audit", help="Audit dataset quality and produce score card")
    audit_parser.add_argument("file", help="Path to input dataset (.csv, .xlsx, .parquet, .json)")
    audit_parser.add_argument("--target", "-t", help="Target column for classification/leakage checks", default=None)
    audit_parser.add_argument("--name", "-n", help="Dataset name", default=None)
    audit_parser.add_argument("--fail-under", "-f", type=float, help="Exit with code 2 if score is below this threshold", default=None)
    audit_parser.add_argument("--fail-on-leakage", action="store_true", help="Exit with code 3 if target leakage is detected")
    audit_parser.add_argument("--export-html", help="Save interactive HTML report to file path")
    audit_parser.add_argument("--export-script", help="Save Python cleaning script to file path")
    audit_parser.add_argument("--export-json", help="Save JSON audit result to file path")

    # Drift subparser
    drift_parser = subparsers.add_parser("drift", help="Compare two datasets for statistical & categorical drift")
    drift_parser.add_argument("--reference", "-r", required=True, help="Path to reference baseline dataset (e.g. train.csv)")
    drift_parser.add_argument("--current", "-c", required=True, help="Path to current target dataset (e.g. test.csv)")
    drift_parser.add_argument("--max-psi", type=float, help="Fail if any column PSI exceeds threshold (e.g. 0.25)")

    # Rules subparser
    rules_parser = subparsers.add_parser("rules", help="Run declarative custom business assertions")
    rules_parser.add_argument("file", help="Path to input dataset")
    rules_parser.add_argument("--rules-file", "-r", required=True, help="Path to JSON file with rule definitions")
    rules_parser.add_argument("--min-pass-rate", type=float, default=100.0, help="Minimum percentage pass rate required")

    # Clean subparser
    clean_parser = subparsers.add_parser("clean", help="Auto-remediate dataset defects and export clean data")
    clean_parser.add_argument("file", help="Path to raw dataset")
    clean_parser.add_argument("--output", "-o", required=True, help="Path to save cleaned dataset")
    clean_parser.add_argument("--target", "-t", help="Target column to preserve", default=None)

    args = parser.parse_args()
    if args.command == "audit":
        audit_command(args)
    elif args.command == "drift":
        drift_command(args)
    elif args.command == "rules":
        rules_command(args)
    elif args.command == "clean":
        clean_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
