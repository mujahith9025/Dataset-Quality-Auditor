"""
Constant & Quasi-Constant Columns Checker
Identifies zero-variance features and quasi-constant columns (>99% dominant value) that add no predictive signal.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class ConstantColumnsChecker(BaseChecker):
    checker_id = "constant_columns"
    checker_name = "Constant & Quasi-Constant Columns Checker"
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

        n_rows = len(df)
        issues = []
        constant_cols = []
        quasi_constant_cols = []

        for col in df.columns:
            if col == target_col:
                continue

            non_null_count = df[col].count()
            if non_null_count == 0:
                # All null column
                constant_cols.append(col)
                issues.append(self.create_issue(
                    issue_id=f"all_null_column_{col}",
                    title=f"All-Null Column '{col}' (0 non-null values)",
                    description=f"Column '{col}' is 100% empty and carries zero information.",
                    severity=SeverityEnum.HIGH,
                    affected_columns=[col],
                    affected_rows_count=n_rows,
                    affected_rows_percentage=100.0,
                    remediation_suggestion=f"Drop column '{col}' from the dataset.",
                    suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                    auto_fixable=True,
                    fix_action=f"drop_column:{col}"
                ))
                continue

            n_unique = df[col].nunique(dropna=True)

            # 1. Zero Variance / Constant Column (1 unique value)
            if n_unique == 1:
                constant_cols.append(col)
                dominant_val = str(df[col].dropna().iloc[0])
                issues.append(self.create_issue(
                    issue_id=f"constant_column_{col}",
                    title=f"Constant Column '{col}' (Zero Variance)",
                    description=f"Column '{col}' has only 1 unique value ('{dominant_val}') across all {non_null_count:,} rows. It contributes zero variance or predictive value.",
                    severity=SeverityEnum.MEDIUM,
                    affected_columns=[col],
                    affected_rows_count=n_rows,
                    affected_rows_percentage=100.0,
                    details={"unique_value": dominant_val},
                    remediation_suggestion=f"Drop constant column '{col}'.",
                    suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                    auto_fixable=True,
                    fix_action=f"drop_column:{col}"
                ))
                continue

            # 2. Quasi-Constant Column (>99% single value)
            if n_rows >= 50:
                top_freq = df[col].value_counts(dropna=True).iloc[0]
                top_val = str(df[col].value_counts(dropna=True).index[0])
                dominant_pct = round((top_freq / non_null_count) * 100, 2)

                if dominant_pct >= 99.0 and n_unique <= 5:
                    quasi_constant_cols.append(col)
                    issues.append(self.create_issue(
                        issue_id=f"quasi_constant_{col}",
                        title=f"Quasi-Constant Column '{col}' ({dominant_pct}% Dominant)",
                        description=f"Column '{col}' is dominated by value '{top_val}' in {top_freq:,} out of {non_null_count:,} rows ({dominant_pct}%).",
                        severity=SeverityEnum.LOW,
                        affected_columns=[col],
                        affected_rows_count=top_freq,
                        affected_rows_percentage=dominant_pct,
                        details={"dominant_value": top_val, "dominant_percentage": dominant_pct},
                        remediation_suggestion=f"Consider removing '{col}' using VarianceThreshold to reduce noise.",
                        suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                        auto_fixable=True,
                        fix_action=f"drop_column:{col}"
                    ))

        # Calculate score
        penalty = min(50.0, (len(constant_cols) * 10.0) + (len(quasi_constant_cols) * 4.0))
        score = round(max(0.0, 100.0 - penalty), 1)

        summary = f"Checked {len(df.columns)} columns. Found {len(constant_cols)} constant and {len(quasi_constant_cols)} quasi-constant columns."
        if not issues:
            summary = "All features exhibit healthy variance (no constant or quasi-constant columns)."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(issues) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "constant_columns_count": len(constant_cols),
                "quasi_constant_columns_count": len(quasi_constant_cols),
                "constant_columns": constant_cols,
                "quasi_constant_columns": quasi_constant_cols
            }
        )
