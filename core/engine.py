"""
Audit Engine Orchestrator
Coordinates dataset profiling, executes all 9 checkers, and packages raw audit findings.
"""

from typing import Any, Dict, List, Optional, Union
import uuid
import datetime
import pandas as pd
import numpy as np

from models.schemas import (
    AuditReport,
    CheckerResult,
    ColumnProfile,
    DimensionEnum,
    DimensionScore,
    Issue,
    SeverityEnum
)
from core.checkers import ALL_CHECKERS


class AuditEngine:
    """Core Auditor engine for running quality checks and compiling comprehensive reports."""

    def __init__(self, checkers: Optional[List[Any]] = None):
        self.checkers = [cls() for cls in (checkers or ALL_CHECKERS)]

    def profile_columns(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, ColumnProfile]:
        """Profiles each column in the dataframe for statistical and categorical metrics."""
        profiles = {}
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            unique_pct = round((unique_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

            # Inferred type
            if pd.api.types.is_numeric_dtype(series):
                if unique_count <= 2:
                    inferred = "boolean / binary"
                elif unique_count < 10:
                    inferred = "discrete numeric"
                else:
                    inferred = "continuous numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred = "datetime"
            else:
                if unique_count == n_rows and n_rows > 10:
                    inferred = "identifier / key"
                elif unique_pct < 5.0 or unique_count <= 20:
                    inferred = "categorical"
                else:
                    inferred = "text / object"

            # Sample values (first 5 distinct non-null)
            clean_samples = series.dropna().unique()[:5].tolist()
            sample_values = [str(x) if isinstance(x, (pd.Timestamp, np.datetime64)) else x for x in clean_samples]

            # Numerical stats
            stats: Dict[str, Any] = {}
            if pd.api.types.is_numeric_dtype(series):
                num_s = series.dropna()
                if not num_s.empty:
                    stats = {
                        "min": round(float(num_s.min()), 2),
                        "max": round(float(num_s.max()), 2),
                        "mean": round(float(num_s.mean()), 2),
                        "median": round(float(num_s.median()), 2),
                        "std": round(float(num_s.std()), 2) if len(num_s) > 1 else 0.0,
                        "q25": round(float(num_s.quantile(0.25)), 2),
                        "q75": round(float(num_s.quantile(0.75)), 2),
                    }
            else:
                # Top value frequency
                top_counts = series.value_counts(dropna=True).head(3)
                stats = {
                    "top_values": {str(k): int(v) for k, v in top_counts.items()}
                }

            profiles[col] = ColumnProfile(
                name=col,
                physical_dtype=str(series.dtype),
                inferred_type=inferred,
                total_count=n_rows,
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=unique_count,
                unique_percentage=unique_pct,
                sample_values=sample_values,
                stats=stats,
                is_target=(col == target_col),
                issues_count=0,
                issues=[]
            )

        return profiles

    def run_audit(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset",
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs all checkers and returns aggregated results."""
        checker_results: Dict[str, CheckerResult] = {}
        all_issues: List[Issue] = []

        # Run all modular checkers
        for checker in self.checkers:
            try:
                res = checker.check(df, target_col=target_col)
                checker_results[checker.checker_id] = res
                all_issues.extend(res.issues)
            except Exception as e:
                # Fallback on unexpected error in a checker
                checker_results[checker.checker_id] = CheckerResult(
                    checker_id=checker.checker_id,
                    checker_name=checker.checker_name,
                    dimension=checker.dimension,
                    passed=False,
                    score=50.0,
                    summary=f"Audit check encountered an error: {str(e)}",
                    issues=[Issue(
                        id=f"{checker.checker_id}_error",
                        dimension=checker.dimension,
                        checker_id=checker.checker_id,
                        severity=SeverityEnum.MEDIUM,
                        title=f"Checker Error in {checker.checker_name}",
                        description=str(e),
                        remediation_suggestion="Inspect dataset format."
                    )]
                )

        # Profile columns
        column_profiles = self.profile_columns(df, target_col=target_col)

        # Map issues to affected columns
        for issue in all_issues:
            for col in issue.affected_columns:
                if col in column_profiles:
                    column_profiles[col].issues_count += 1
                    column_profiles[col].issues.append(f"[{issue.severity.value}] {issue.title}")

        return {
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "target_column": target_col,
            "checker_results": checker_results,
            "column_profiles": column_profiles,
            "all_issues": all_issues
        }

    def audit(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset",
        target_col: Optional[str] = None
    ) -> AuditReport:
        """Runs audit checks and builds the complete standardized AuditReport."""
        from core.scoring import ScoringEngine
        raw_result = self.run_audit(df, dataset_name=dataset_name, target_col=target_col)
        scorer = ScoringEngine()
        return scorer.build_audit_report(raw_result)
