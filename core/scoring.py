"""
Scoring & Actionable Recommendation Engine
Calculates 5-dimension quality scores, weighted composite score (0-100), letter grades, priority alert banners, and remediation recipes.
"""

from typing import Any, Dict, List, Optional, Tuple
import datetime
import uuid
import pandas as pd
import numpy as np

from models.schemas import (
    AuditReport,
    CheckerResult,
    DimensionEnum,
    DimensionScore,
    Issue,
    SeverityEnum
)


DIMENSION_WEIGHTS = {
    DimensionEnum.COMPLETENESS: 0.25,
    DimensionEnum.UNIQUENESS: 0.15,
    DimensionEnum.VALIDITY: 0.20,
    DimensionEnum.CONSISTENCY: 0.15,
    DimensionEnum.ML_READINESS: 0.25,
}

DIMENSION_CHECKERS = {
    DimensionEnum.COMPLETENESS: ["missing_values"],
    DimensionEnum.UNIQUENESS: ["duplicate_rows"],
    DimensionEnum.VALIDITY: ["outliers", "constant_columns", "suspicious_values"],
    DimensionEnum.CONSISTENCY: ["high_correlation", "data_types"],
    DimensionEnum.ML_READINESS: ["class_imbalance", "data_leakage"],
}


class ScoringEngine:
    """Calculates weighted composite data quality score and structures remediation plans."""

    @staticmethod
    def get_grade(score: float) -> str:
        """Assigns an intuitive letter grade based on score."""
        if score >= 95.0:
            return "A+"
        elif score >= 90.0:
            return "A"
        elif score >= 80.0:
            return "B"
        elif score >= 70.0:
            return "C"
        elif score >= 60.0:
            return "D"
        else:
            return "F"

    def calculate_dimension_scores(self, checker_results: Dict[str, CheckerResult]) -> Dict[str, DimensionScore]:
        """Aggregates checker scores into the 5 core quality dimensions."""
        dim_scores: Dict[str, DimensionScore] = {}

        for dimension, weight in DIMENSION_WEIGHTS.items():
            checker_keys = DIMENSION_CHECKERS.get(dimension, [])
            dim_results = [checker_results[k] for k in checker_keys if k in checker_results]

            if not dim_results:
                score = 100.0
                issues_count = {}
            else:
                # Average score of checkers within this dimension
                score = round(sum(cr.score for cr in dim_results) / len(dim_results), 1)
                
                # Count issues by severity in this dimension
                issues_count = {
                    SeverityEnum.CRITICAL.value: 0,
                    SeverityEnum.HIGH.value: 0,
                    SeverityEnum.MEDIUM.value: 0,
                    SeverityEnum.LOW.value: 0,
                    SeverityEnum.INFO.value: 0,
                }
                for cr in dim_results:
                    for issue in cr.issues:
                        sev = issue.severity.value
                        issues_count[sev] = issues_count.get(sev, 0) + 1

            weighted = round(score * weight, 2)
            grade = self.get_grade(score)

            dim_scores[dimension.value] = DimensionScore(
                dimension=dimension,
                score=score,
                weight=weight,
                weighted_score=weighted,
                grade=grade,
                issues_count=issues_count
            )

        return dim_scores

    def calculate_overall_score(self, dimension_scores: Dict[str, DimensionScore]) -> Tuple[float, str]:
        """Calculates total composite score (0-100) and overall letter grade."""
        total_score = sum(ds.weighted_score for ds in dimension_scores.values())
        total_score = round(max(0.0, min(100.0, total_score)), 1)
        grade = self.get_grade(total_score)
        return total_score, grade

    def generate_summary_alerts(
        self,
        checker_results: Dict[str, CheckerResult],
        overall_score: float
    ) -> List[str]:
        """Generates punchy, executive-level summary alert items for the report."""
        alerts = []

        # Missing values alert
        if "missing_values" in checker_results:
            mv = checker_results["missing_values"]
            mv_pct = mv.metrics.get("overall_missing_percentage", 0.0)
            mv_cols = mv.metrics.get("columns_with_missing", 0)
            if mv_pct > 0:
                alerts.append(f"⚠ {mv_pct}% missing values across {mv_cols} columns")

        # Duplicates alert
        if "duplicate_rows" in checker_results:
            dup = checker_results["duplicate_rows"]
            dup_cnt = dup.metrics.get("exact_duplicates_count", 0)
            dup_pct = dup.metrics.get("exact_duplicates_percentage", 0.0)
            if dup_cnt > 0:
                alerts.append(f"⚠ {dup_cnt:,} duplicate rows ({dup_pct}%)")

        # Class imbalance alert
        if "class_imbalance" in checker_results:
            ci = checker_results["class_imbalance"]
            for issue in ci.issues:
                if issue.severity in [SeverityEnum.CRITICAL, SeverityEnum.HIGH]:
                    min_pct = ci.metrics.get("minority_percentage", 0)
                    ratio = ci.metrics.get("imbalance_ratio", 0)
                    alerts.append(f"⚠ Severe class imbalance ({ratio}:1 ratio, minority {min_pct}%)")
                    break

        # Data Leakage alert
        if "data_leakage" in checker_results:
            dl = checker_results["data_leakage"]
            leaky = dl.metrics.get("leaky_columns", [])
            if leaky:
                cols_str = ", ".join(f"'{c}'" for c in leaky[:2])
                alerts.append(f"⚠ Critical data leakage in {cols_str}")

        # Outliers alert
        if "outliers" in checker_results:
            out = checker_results["outliers"]
            ext_cols = out.metrics.get("extreme_outlier_features", 0)
            if ext_cols > 0:
                alerts.append(f"⚠ Extreme outliers found in {ext_cols} numerical features")

        # Suspicious values alert
        if "suspicious_values" in checker_results:
            sv = checker_results["suspicious_values"]
            susp_cols = sv.metrics.get("suspicious_columns_count", 0)
            if susp_cols > 0:
                alerts.append(f"⚠ {susp_cols} suspicious columns with sentinels / invalid bounds")

        # High correlation alert
        if "high_correlation" in checker_results:
            hc = checker_results["high_correlation"]
            pair_cnt = hc.metrics.get("high_corr_pairs_count", 0)
            if pair_cnt > 0:
                alerts.append(f"⚠ {pair_cnt} highly correlated feature pairs (|r| >= 0.85)")

        # Data types alert
        if "data_types" in checker_results:
            dt = checker_results["data_types"]
            dt_cnt = dt.metrics.get("type_mismatches_count", 0)
            if dt_cnt > 0:
                alerts.append(f"⚠ {dt_cnt} columns with incorrect string-wrapped types (currency / dates)")

        # Constant columns alert
        if "constant_columns" in checker_results:
            cc = checker_results["constant_columns"]
            const_cnt = cc.metrics.get("constant_columns_count", 0)
            if const_cnt > 0:
                alerts.append(f"⚠ {const_cnt} constant columns with zero predictive variance")

        if not alerts:
            alerts.append("✓ Dataset meets high quality standards across all evaluated checks.")

        return alerts

    def generate_recommendations(self, all_issues: List[Issue]) -> List[Dict[str, Any]]:
        """Transforms audit issues into an ordered, prioritized action plan with code snippets."""
        severity_order = {
            SeverityEnum.CRITICAL: 0,
            SeverityEnum.HIGH: 1,
            SeverityEnum.MEDIUM: 2,
            SeverityEnum.LOW: 3,
            SeverityEnum.INFO: 4,
        }

        # Sort issues by severity then dimension
        sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x.severity, 99))
        recommendations = []

        for i, issue in enumerate(sorted_issues, start=1):
            if issue.severity == SeverityEnum.INFO:
                continue

            rec = {
                "step": i,
                "issue_id": issue.id,
                "title": issue.title,
                "dimension": issue.dimension.value,
                "severity": issue.severity.value,
                "affected_columns": issue.affected_columns,
                "action": issue.remediation_suggestion,
                "code_snippet": issue.suggested_code_snippet,
                "auto_fixable": issue.auto_fixable,
                "fix_action": issue.fix_action
            }
            recommendations.append(rec)

        return recommendations

    def build_audit_report(self, raw_audit: Dict[str, Any]) -> AuditReport:
        """Assembles the complete, finalized AuditReport model."""
        checker_results = raw_audit["checker_results"]
        all_issues = raw_audit["all_issues"]

        # 1. Dimension scores
        dim_scores = self.calculate_dimension_scores(checker_results)

        # 2. Overall score & grade
        overall_score, grade = self.calculate_overall_score(dim_scores)

        # 3. Summary alerts
        summary_alerts = self.generate_summary_alerts(checker_results, overall_score)

        # 4. Severity counts
        severity_counts = {
            SeverityEnum.CRITICAL.value: 0,
            SeverityEnum.HIGH.value: 0,
            SeverityEnum.MEDIUM.value: 0,
            SeverityEnum.LOW.value: 0,
            SeverityEnum.INFO.value: 0,
        }
        for issue in all_issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # 5. Recommendations
        recommendations = self.generate_recommendations(all_issues)

        return AuditReport(
            audit_id=f"audit_{uuid.uuid4().hex[:8]}",
            dataset_name=raw_audit.get("dataset_name", "Dataset"),
            total_rows=raw_audit["total_rows"],
            total_columns=raw_audit["total_columns"],
            memory_usage_mb=raw_audit["memory_usage_mb"],
            target_column=raw_audit.get("target_column"),
            overall_score=overall_score,
            grade=grade,
            dimension_scores=dim_scores,
            summary_alerts=summary_alerts,
            total_issues_count=len(all_issues),
            severity_counts=severity_counts,
            checker_results=checker_results,
            column_profiles=raw_audit["column_profiles"],
            recommended_actions=recommendations,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
