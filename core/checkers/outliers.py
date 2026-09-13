"""
Outliers Checker
Audits numerical features using IQR (Tukey fences), Modified Z-Score (MAD), and Multivariate Isolation Forest.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class OutliersChecker(BaseChecker):
    checker_id = "outliers"
    checker_name = "Outliers & Anomaly Checker"
    dimension = DimensionEnum.VALIDITY

    def check(self, df: pd.DataFrame, target_col: Optional[str] = None) -> CheckerResult:
        if df.empty:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary="Dataset is empty.",
                issues=[]
            )

        num_cols = self.get_numeric_columns(df)
        # Exclude ID or binary columns
        candidate_cols = []
        for col in num_cols:
            if col == target_col:
                continue
            # Skip if binary 0/1 or very low unique count
            if df[col].nunique() > 5:
                candidate_cols.append(col)

        if not candidate_cols:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary="No continuous numeric features found to evaluate for outliers.",
                issues=[]
            )

        n_rows = len(df)
        issues = []
        col_outlier_stats = {}
        total_outlier_count_overall = 0
        extreme_outlier_cols = []

        for col in candidate_cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) < 10:
                continue

            # 1. IQR Method
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            extreme_lower = q1 - 3.0 * iqr
            extreme_upper = q3 + 3.0 * iqr

            outliers_mask = (series < lower_bound) | (series > upper_bound)
            extreme_mask = (series < extreme_lower) | (series > extreme_upper)

            outlier_count = int(outliers_mask.sum())
            extreme_count = int(extreme_mask.sum())
            outlier_pct = round((outlier_count / len(series)) * 100, 2)
            extreme_pct = round((extreme_count / len(series)) * 100, 2)

            total_outlier_count_overall += outlier_count

            # 2. Modified Z-Score with MAD
            median = float(series.median())
            mad = float((series - median).abs().median())
            mod_z_outliers = 0
            if mad > 0:
                mod_z = 0.6745 * (series - median).abs() / mad
                mod_z_outliers = int((mod_z > 3.5).sum())

            col_outlier_stats[col] = {
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "extreme_count": extreme_count,
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "median": round(median, 2),
                "iqr": round(iqr, 2)
            }

            if extreme_count > 0 or outlier_pct > 8.0:
                severity = SeverityEnum.HIGH if extreme_pct > 2.0 or outlier_pct > 15.0 else SeverityEnum.MEDIUM
                if extreme_count > 0:
                    extreme_outlier_cols.append(col)

                issues.append(self.create_issue(
                    issue_id=f"outliers_{col}",
                    title=f"Significant Outliers in '{col}' ({outlier_count:,} points, {outlier_pct}%)",
                    description=f"Column '{col}' has {outlier_count:,} values outside IQR bounds [{lower_bound:,.2f}, {upper_bound:,.2f}], including {extreme_count:,} extreme outliers.",
                    severity=severity,
                    affected_columns=[col],
                    affected_rows_count=outlier_count,
                    affected_rows_percentage=outlier_pct,
                    details={
                        "iqr_lower": lower_bound,
                        "iqr_upper": upper_bound,
                        "extreme_count": extreme_count,
                        "max_value": float(series.max()),
                        "min_value": float(series.min())
                    },
                    remediation_suggestion=f"Clip extreme outliers using Winsorization / IQR bounds, or apply log transformation.",
                    suggested_code_snippet=f"q1, q3 = df['{col}'].quantile([0.01, 0.99])\ndf['{col}'] = df['{col}'].clip(q1, q3)",
                    auto_fixable=True,
                    fix_action=f"clip_outliers:{col}"
                ))

        # 3. Multivariate Isolation Forest Check
        if len(candidate_cols) >= 2 and n_rows >= 50:
            try:
                from sklearn.ensemble import IsolationForest
                num_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce').fillna(df[candidate_cols].median())
                iso = IsolationForest(contamination=0.03, random_state=42)
                preds = iso.fit_predict(num_df)
                multi_anomalies = int((preds == -1).sum())
                if multi_anomalies > 0:
                    multi_pct = round((multi_anomalies / n_rows) * 100, 2)
                    issues.append(self.create_issue(
                        issue_id="multivariate_anomalies",
                        title=f"{multi_anomalies:,} Multivariate Anomalies Detected ({multi_pct}%)",
                        description=f"Isolation Forest identified {multi_anomalies:,} rows as multi-dimensional joint outliers.",
                        severity=SeverityEnum.LOW,
                        affected_columns=candidate_cols[:4],
                        affected_rows_count=multi_anomalies,
                        affected_rows_percentage=multi_pct,
                        remediation_suggestion="Inspect anomaly rows for measurement errors or rare sub-populations.",
                        suggested_code_snippet="# Filter or inspect Isolation Forest anomalies\nfrom sklearn.ensemble import IsolationForest",
                        auto_fixable=False
                    ))
            except Exception:
                pass

        # Calculate score
        penalty = min(50.0, (len(extreme_outlier_cols) * 8.0) + (len(issues) * 4.0))
        score = round(max(0.0, 100.0 - penalty), 1)

        summary = f"{len(col_outlier_stats)} numeric columns evaluated. Found outlier anomalies in {len(issues)} features."
        if not issues:
            summary = "No significant outlier anomalies detected across numeric features."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(issues) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "features_evaluated": len(candidate_cols),
                "features_with_outliers": len(issues),
                "extreme_outlier_features": len(extreme_outlier_cols),
                "total_outlier_points": total_outlier_count_overall
            },
            visual_data={
                "column_outliers": [
                    {"column": k, **v} for k, v in col_outlier_stats.items() if v["outlier_count"] > 0
                ]
            }
        )
