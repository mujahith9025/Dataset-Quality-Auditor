"""
Missing Values Checker
Audits total missing rates, per-column null distributions, blank whitespace, and missing patterns.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class MissingValuesChecker(BaseChecker):
    checker_id = "missing_values"
    checker_name = "Missing Values Checker"
    dimension = DimensionEnum.COMPLETENESS

    def check(self, df: pd.DataFrame, target_col: Optional[str] = None) -> CheckerResult:
        if df.empty:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=False,
                score=0.0,
                summary="Dataset is empty.",
                issues=[self.create_issue(
                    issue_id="empty_dataset",
                    title="Empty Dataset",
                    description="The dataset contains zero rows.",
                    severity=SeverityEnum.CRITICAL,
                    remediation_suggestion="Provide a non-empty dataset."
                )]
            )

        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        
        # 1. Standard null counts + whitespace blanks
        null_counts = df.isna().sum()
        
        # Detect whitespace-only strings
        whitespace_counts = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            ws_count = df[col].astype(str).str.strip().eq("").sum() - df[col].isna().sum()
            if ws_count > 0:
                whitespace_counts[col] = int(ws_count)
                null_counts[col] += ws_count

        total_missing_cells = int(null_counts.sum())
        overall_missing_pct = round((total_missing_cells / total_cells) * 100, 2)
        
        issues = []
        col_missing_breakdown = {}
        columns_to_drop = []
        columns_to_impute = []

        # 2. Inspect per column
        for col in df.columns:
            count = int(null_counts[col])
            pct = round((count / n_rows) * 100, 2)
            col_missing_breakdown[col] = {
                "missing_count": count,
                "missing_percentage": pct,
                "whitespace_count": whitespace_counts.get(col, 0)
            }

            if pct > 60.0:
                columns_to_drop.append(col)
                issues.append(self.create_issue(
                    issue_id=f"excessive_missing_{col}",
                    title=f"Excessive Missing Values in '{col}' ({pct}%)",
                    description=f"Column '{col}' has {count:,} missing values ({pct}%), which exceeds the 60% threshold for reliable imputation.",
                    severity=SeverityEnum.HIGH if col != target_col else SeverityEnum.CRITICAL,
                    affected_columns=[col],
                    affected_rows_count=count,
                    affected_rows_percentage=pct,
                    details={"missing_percentage": pct, "missing_count": count},
                    remediation_suggestion=f"Drop column '{col}' as it lacks sufficient signal, or collect additional data.",
                    suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                    auto_fixable=True,
                    fix_action=f"drop_column:{col}"
                ))
            elif pct >= 20.0:
                columns_to_impute.append(col)
                issues.append(self.create_issue(
                    issue_id=f"high_missing_{col}",
                    title=f"High Missing Rate in '{col}' ({pct}%)",
                    description=f"Column '{col}' has {count:,} missing values ({pct}%). This may introduce bias if not properly handled.",
                    severity=SeverityEnum.MEDIUM if col != target_col else SeverityEnum.HIGH,
                    affected_columns=[col],
                    affected_rows_count=count,
                    affected_rows_percentage=pct,
                    details={"missing_percentage": pct, "missing_count": count},
                    remediation_suggestion=f"Use iterative/MICE imputation, KNN imputation, or create a 'missing_indicator' column.",
                    suggested_code_snippet=f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())  # Or mode for categorical",
                    auto_fixable=True,
                    fix_action=f"impute:{col}"
                ))
            elif pct > 0.0:
                columns_to_impute.append(col)
                issues.append(self.create_issue(
                    issue_id=f"moderate_missing_{col}",
                    title=f"Missing Values in '{col}' ({pct}%)",
                    description=f"Column '{col}' contains {count:,} missing values ({pct}%).",
                    severity=SeverityEnum.LOW,
                    affected_columns=[col],
                    affected_rows_count=count,
                    affected_rows_percentage=pct,
                    details={"missing_percentage": pct, "missing_count": count},
                    remediation_suggestion=f"Impute using median (numeric) or mode (categorical), or drop missing rows.",
                    suggested_code_snippet=f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())",
                    auto_fixable=True,
                    fix_action=f"impute:{col}"
                ))

        # 3. Check for target column missingness (CRITICAL)
        if target_col and target_col in df.columns:
            target_nulls = int(null_counts[target_col])
            if target_nulls > 0:
                t_pct = round((target_nulls / n_rows) * 100, 2)
                issues.insert(0, self.create_issue(
                    issue_id="target_missing_values",
                    title=f"Target Column '{target_col}' Has {target_nulls:,} Missing Labels ({t_pct}%)",
                    description=f"Supervised learning models cannot train on unlabeled target rows.",
                    severity=SeverityEnum.CRITICAL,
                    affected_columns=[target_col],
                    affected_rows_count=target_nulls,
                    affected_rows_percentage=t_pct,
                    details={"target_missing_count": target_nulls, "target_missing_percentage": t_pct},
                    remediation_suggestion=f"Drop all rows where target column '{target_col}' is NaN.",
                    suggested_code_snippet=f"df = df.dropna(subset=['{target_col}'])",
                    auto_fixable=True,
                    fix_action=f"drop_target_na:{target_col}"
                ))

        # 4. Check for rows with high missingness (>50% missing columns)
        row_null_counts = df.isna().sum(axis=1)
        sparse_rows = int((row_null_counts > (n_cols * 0.5)).sum())
        if sparse_rows > 0:
            s_pct = round((sparse_rows / n_rows) * 100, 2)
            issues.append(self.create_issue(
                issue_id="sparse_rows_detected",
                title=f"{sparse_rows:,} Severely Incomplete Rows ({s_pct}%)",
                description=f"{sparse_rows} rows have more than 50% of their features missing.",
                severity=SeverityEnum.MEDIUM,
                affected_rows_count=sparse_rows,
                affected_rows_percentage=s_pct,
                details={"sparse_rows_count": sparse_rows},
                remediation_suggestion="Remove severely degraded rows before model training.",
                suggested_code_snippet=f"df = df.dropna(thresh=int(df.shape[1] * 0.5))",
                auto_fixable=True,
                fix_action="drop_sparse_rows"
            ))

        # Calculate Score
        # Perfect is 100. Deduct based on overall missing pct and affected columns
        score = max(0.0, 100.0 - (overall_missing_pct * 3.5) - (len(columns_to_drop) * 10))
        score = round(min(100.0, score), 1)

        summary = f"{overall_missing_pct}% total missing values across dataset. {len(columns_to_impute)} columns affected."
        if overall_missing_pct == 0.0:
            summary = "No missing values found across all columns (100% complete)."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=overall_missing_pct == 0.0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "overall_missing_percentage": overall_missing_pct,
                "total_missing_cells": total_missing_cells,
                "total_cells": total_cells,
                "columns_with_missing": len(columns_to_impute),
                "columns_to_drop_count": len(columns_to_drop),
                "whitespace_only_detected": sum(whitespace_counts.values())
            },
            visual_data={
                "column_breakdown": [
                    {"column": col, "missing_pct": data["missing_percentage"], "missing_count": data["missing_count"]}
                    for col, data in col_missing_breakdown.items() if data["missing_count"] > 0
                ]
            }
        )
