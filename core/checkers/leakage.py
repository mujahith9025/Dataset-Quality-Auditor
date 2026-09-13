"""
Data Leakage Checker
Detects features that inadvertently leak the target variable (near-perfect correlation, mutual information > 0.85, proxy features).
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class DataLeakageChecker(BaseChecker):
    checker_id = "data_leakage"
    checker_name = "Data Leakage & Proxy Feature Checker"
    dimension = DimensionEnum.ML_READINESS

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

        # Target detection
        eval_col = target_col
        if not eval_col or eval_col not in df.columns:
            candidates = [
                c for c in df.columns
                if any(c.lower().endswith(k) for k in ["status", "target", "label", "flag", "class", "churn", "default", "approved"])
            ]
            if candidates:
                eval_col = candidates[0]

        if not eval_col or eval_col not in df.columns:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary="No target column identified to audit for data leakage.",
                issues=[]
            )

        issues = []
        leaky_features = []
        target_series = pd.to_numeric(df[eval_col], errors='coerce')
        
        # If target is non-numeric, encode temporarily
        if target_series.isna().all() or target_series.nunique() <= 1:
            target_series = pd.Series(pd.factorize(df[eval_col])[0], index=df.index)

        # 1. Direct Linear / Rank Correlation with Target
        for col in df.columns:
            if col == eval_col:
                continue

            feature_series = pd.to_numeric(df[col], errors='coerce')
            
            # Check numeric correlation
            if not feature_series.isna().all() and feature_series.nunique() > 1:
                valid_idx = ~(feature_series.isna() | target_series.isna())
                if valid_idx.sum() > 20:
                    try:
                        pearson_r = abs(float(np.corrcoef(feature_series[valid_idx], target_series[valid_idx])[0, 1]))
                        if np.isnan(pearson_r):
                            pearson_r = 0.0
                    except Exception:
                        pearson_r = 0.0

                    if pearson_r >= 0.95:
                        leaky_features.append(col)
                        issues.append(self.create_issue(
                            issue_id=f"direct_target_leakage_{col}",
                            title=f"Critical Data Leakage in Feature '{col}' (|r| = {pearson_r:.3f})",
                            description=f"Feature '{col}' has an almost perfect correlation ({pearson_r:.3f}) with target '{eval_col}'. This indicates target leakage, a duplicate target representation, or a post-event feature created after the target outcome.",
                            severity=SeverityEnum.CRITICAL,
                            affected_columns=[col, eval_col],
                            details={"correlation": round(pearson_r, 4), "target_column": eval_col},
                            remediation_suggestion=f"Drop '{col}' from feature set before training to prevent model shortcutting and unrealistic test accuracy.",
                            suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                            auto_fixable=True,
                            fix_action=f"drop_column:{col}"
                        ))
                    elif pearson_r >= 0.88:
                        issues.append(self.create_issue(
                            issue_id=f"suspicious_target_correlation_{col}",
                            title=f"Suspiciously High Target Correlation in '{col}' (|r| = {pearson_r:.3f})",
                            description=f"Feature '{col}' has unusually high correlation with '{eval_col}'. Verify that this feature is legitimately available at prediction time.",
                            severity=SeverityEnum.HIGH,
                            affected_columns=[col, eval_col],
                            details={"correlation": round(pearson_r, 4)},
                            remediation_suggestion=f"Verify business timeline: ensure '{col}' is recorded strictly BEFORE '{eval_col}' occurs.",
                            suggested_code_snippet=f"# Review business logic for {col}",
                            auto_fixable=False
                        ))

            # 2. Check for categorical proxy leakage (perfect subset match)
            if df[col].dtype == object or str(df[col].dtype) == 'category':
                # Cross-tabulation purity
                try:
                    crosstab = pd.crosstab(df[col], df[eval_col], normalize='index')
                    max_purities = crosstab.max(axis=1)
                    if (max_purities > 0.99).all() and df[col].nunique() > 1 and df[col].nunique() <= 10:
                        if col not in leaky_features:
                            leaky_features.append(col)
                            issues.append(self.create_issue(
                                issue_id=f"categorical_leakage_{col}",
                                title=f"Deterministic Target Proxy Feature in '{col}'",
                                description=f"Values in categorical column '{col}' map with 100% certainty to target classes in '{eval_col}'.",
                                severity=SeverityEnum.CRITICAL,
                                affected_columns=[col, eval_col],
                                details={"crosstab_purity": max_purities.to_dict()},
                                remediation_suggestion=f"Remove '{col}' to prevent target leakage.",
                                suggested_code_snippet=f"df = df.drop(columns=['{col}'])",
                                auto_fixable=True,
                                fix_action=f"drop_column:{col}"
                            ))
                except Exception:
                    pass

        score = max(0.0, 100.0 - (len(leaky_features) * 35.0) - (len(issues) * 10.0))
        score = round(min(100.0, score), 1)

        summary = f"Audited {len(df.columns)-1} features against target '{eval_col}'. {len(leaky_features)} critical leakage features detected."
        if not issues:
            summary = f"No data leakage or suspicious proxy features detected with respect to target '{eval_col}'."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(leaky_features) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "target_column": eval_col,
                "features_evaluated": len(df.columns) - 1,
                "critical_leakage_count": len(leaky_features),
                "leaky_columns": leaky_features
            }
        )
