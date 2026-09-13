"""
Duplicate Rows & Key Checker
Detects exact full-row duplicates, ID uniqueness violations, and near-duplicate records.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class DuplicateRowsChecker(BaseChecker):
    checker_id = "duplicate_rows"
    checker_name = "Duplicate Rows & Keys Checker"
    dimension = DimensionEnum.UNIQUENESS

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
        
        # 1. Exact Full-Row Duplicates
        duplicate_mask = df.duplicated(keep="first")
        exact_dup_count = int(duplicate_mask.sum())
        exact_dup_pct = round((exact_dup_count / n_rows) * 100, 2)

        if exact_dup_count > 0:
            severity = SeverityEnum.HIGH if exact_dup_pct > 5.0 else SeverityEnum.MEDIUM
            issues.append(self.create_issue(
                issue_id="exact_duplicates_found",
                title=f"{exact_dup_count:,} Exact Duplicate Rows ({exact_dup_pct}%)",
                description=f"The dataset contains {exact_dup_count:,} identical rows. Exact duplicates can artificially inflate model performance or cause data leakage between train/test splits.",
                severity=severity,
                affected_rows_count=exact_dup_count,
                affected_rows_percentage=exact_dup_pct,
                details={"duplicate_count": exact_dup_count, "duplicate_percentage": exact_dup_pct},
                remediation_suggestion="Remove duplicate rows to preserve statistical independence and prevent train/test contamination.",
                suggested_code_snippet="df = df.drop_duplicates(keep='first')",
                auto_fixable=True,
                fix_action="drop_exact_duplicates"
            ))

        # 2. Check for Potential ID Column Uniqueness Violations
        # Identify columns named like id, uuid, key, code
        id_cols = [
            c for c in df.columns 
            if any(k in c.lower() for k in ["_id", "id_", "identifier", "uuid", "guid", "key", "pk"]) 
            or c.lower() in ["id", "pk", "ssn", "email", "phone"]
        ]

        id_violations = []
        for id_col in id_cols:
            n_unique = df[id_col].nunique(dropna=True)
            non_null_count = df[id_col].count()
            if non_null_count > 0 and n_unique < non_null_count:
                dup_id_count = non_null_count - n_unique
                dup_id_pct = round((dup_id_count / non_null_count) * 100, 2)
                id_violations.append(id_col)
                issues.append(self.create_issue(
                    issue_id=f"duplicate_key_{id_col}",
                    title=f"Non-Unique Primary Key in '{id_col}' ({dup_id_count:,} duplicates)",
                    description=f"Column '{id_col}' appears to be a unique identifier but contains {dup_id_count:,} non-unique values ({dup_id_pct}%).",
                    severity=SeverityEnum.HIGH,
                    affected_columns=[id_col],
                    affected_rows_count=dup_id_count,
                    affected_rows_percentage=dup_id_pct,
                    details={"unique_keys": n_unique, "total_keys": int(non_null_count)},
                    remediation_suggestion=f"Verify data ingestion pipeline or deduplicate records using '{id_col}'.",
                    suggested_code_snippet=f"df = df.drop_duplicates(subset=['{id_col}'], keep='first')",
                    auto_fixable=True,
                    fix_action=f"drop_duplicates_by_key:{id_col}"
                ))

        # 3. Feature Subset Duplicates (excluding ID columns)
        non_id_cols = [c for c in df.columns if c not in id_cols]
        feature_dups_count = 0
        if len(non_id_cols) >= 3 and len(id_cols) > 0:
            feature_dup_mask = df.duplicated(subset=non_id_cols, keep="first")
            feature_dups_count = int(feature_dup_mask.sum())
            if feature_dups_count > exact_dup_count:
                diff_count = feature_dups_count - exact_dup_count
                diff_pct = round((diff_count / n_rows) * 100, 2)
                if diff_pct > 2.0:
                    issues.append(self.create_issue(
                        issue_id="feature_subset_duplicates",
                        title=f"{diff_count:,} Duplicate Feature Rows with Distinct IDs ({diff_pct}%)",
                        description=f"{diff_count:,} records have identical feature values despite having different IDs.",
                        severity=SeverityEnum.LOW,
                        affected_columns=non_id_cols[:5],
                        affected_rows_count=diff_count,
                        affected_rows_percentage=diff_pct,
                        details={"subset_duplicate_count": feature_dups_count},
                        remediation_suggestion="Inspect whether these are legitimate recurring observations or repeated records.",
                        suggested_code_snippet=f"df = df.drop_duplicates(subset={non_id_cols}, keep='first')",
                        auto_fixable=False
                    ))

        # Scoring
        score = max(0.0, 100.0 - (exact_dup_pct * 4.0) - (len(id_violations) * 12.0))
        score = round(min(100.0, score), 1)

        summary = f"{exact_dup_count:,} duplicate rows detected ({exact_dup_pct}%)."
        if exact_dup_count == 0 and not id_violations:
            summary = "No duplicate rows or ID conflicts detected (100% unique)."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=exact_dup_count == 0 and len(id_violations) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "exact_duplicates_count": exact_dup_count,
                "exact_duplicates_percentage": exact_dup_pct,
                "id_columns_inspected": len(id_cols),
                "id_violations_count": len(id_violations)
            }
        )
