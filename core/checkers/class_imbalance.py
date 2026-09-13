"""
Class Imbalance & Distribution Checker
Evaluates target feature class distributions, entropy, minority-to-majority ratios, and suggests sampling strategies.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class ClassImbalanceChecker(BaseChecker):
    checker_id = "class_imbalance"
    checker_name = "Class Imbalance & Distribution Checker"
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

        # 1. Determine target column (provided or auto-detected)
        eval_col = target_col
        if not eval_col or eval_col not in df.columns:
            # Try auto-detecting candidate target columns: binary or low-cardinality categorical columns ending in _status, target, label, flag, class, churn, default
            candidates = [
                c for c in df.columns
                if any(c.lower().endswith(k) for k in ["status", "target", "label", "flag", "class", "churn", "default", "approved", "readmitted"])
                and 2 <= df[c].nunique() <= 10
            ]
            if candidates:
                eval_col = candidates[0]
            else:
                # If still none, find any binary 0/1 column
                binary_cols = [c for c in df.columns if df[c].nunique() == 2]
                if binary_cols:
                    eval_col = binary_cols[-1]

        if not eval_col or eval_col not in df.columns:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary="No classification target column specified or detected. Skipping class imbalance check.",
                issues=[self.create_issue(
                    issue_id="no_target_specified",
                    title="No Target Column Specified",
                    description="Specify a target column to audit class imbalance and machine learning readiness.",
                    severity=SeverityEnum.INFO,
                    remediation_suggestion="Pass target_col parameter to evaluate model-specific imbalance."
                )]
            )

        series = df[eval_col].dropna()
        n_unique = series.nunique()

        if n_unique > 25:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary=f"Column '{eval_col}' is continuous ({n_unique} unique values). Imbalance checks apply to categorical/discrete targets.",
                issues=[]
            )

        value_counts = series.value_counts()
        total_samples = len(series)
        proportions = (value_counts / total_samples).round(4)
        
        maj_count = int(value_counts.iloc[0])
        min_count = int(value_counts.iloc[-1])
        maj_class = str(value_counts.index[0])
        min_class = str(value_counts.index[-1])

        imbalance_ratio = round(min_count / maj_count, 3) if maj_count > 0 else 1.0
        min_pct = round((min_count / total_samples) * 100, 2)
        maj_pct = round((maj_count / total_samples) * 100, 2)

        # Shannon Entropy
        probs = proportions.values
        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        max_entropy = float(np.log2(n_unique)) if n_unique > 1 else 1.0
        normalized_entropy = round(entropy / max_entropy, 3) if max_entropy > 0 else 1.0

        issues = []
        score = 100.0

        if imbalance_ratio < 0.15 or min_pct < 10.0:
            # Severe Class Imbalance
            severity = SeverityEnum.CRITICAL if min_pct < 5.0 or imbalance_ratio < 0.08 else SeverityEnum.HIGH
            score = 60.0 if severity == SeverityEnum.HIGH else 45.0
            
            issues.append(self.create_issue(
                issue_id="severe_class_imbalance",
                title=f"Severe Class Imbalance in '{eval_col}' ({imbalance_ratio}:1 ratio, minority {min_pct}%)",
                description=f"Class distribution is heavily skewed: Majority class '{maj_class}' represents {maj_pct}%, while minority class '{min_class}' represents only {min_pct}% ({min_count:,} samples). Standard accuracy will be misleading.",
                severity=severity,
                affected_columns=[eval_col],
                affected_rows_count=min_count,
                affected_rows_percentage=min_pct,
                details={
                    "target_column": eval_col,
                    "imbalance_ratio": imbalance_ratio,
                    "minority_class": min_class,
                    "minority_count": min_count,
                    "minority_percentage": min_pct,
                    "majority_class": maj_class,
                    "majority_count": maj_count,
                    "majority_percentage": maj_pct,
                    "entropy": round(entropy, 2),
                    "normalized_entropy": normalized_entropy,
                    "distribution": {str(k): int(v) for k, v in value_counts.items()}
                },
                remediation_suggestion="Apply SMOTE / ADASYN oversampling, class_weight='balanced' in model loss, or use PR-AUC / F1-score evaluation metrics.",
                suggested_code_snippet=f"# Handle imbalance with imbalanced-learn\nfrom imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state=42)\nX_res, y_res = smote.fit_resample(X, y)",
                auto_fixable=False
            ))
        elif imbalance_ratio < 0.35:
            # Moderate Class Imbalance
            score = 80.0
            issues.append(self.create_issue(
                issue_id="moderate_class_imbalance",
                title=f"Moderate Class Imbalance in '{eval_col}' ({imbalance_ratio}:1 ratio)",
                description=f"Minority class '{min_class}' has {min_count:,} samples ({min_pct}%).",
                severity=SeverityEnum.MEDIUM,
                affected_columns=[eval_col],
                affected_rows_count=min_count,
                affected_rows_percentage=min_pct,
                details={"imbalance_ratio": imbalance_ratio, "minority_percentage": min_pct},
                remediation_suggestion="Use stratified cross-validation (StratifiedKFold) and tune decision threshold.",
                suggested_code_snippet="from sklearn.model_selection import StratifiedKFold\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
                auto_fixable=False
            ))

        summary = f"Target '{eval_col}' audited: {n_unique} classes. Minority ratio is {imbalance_ratio}:1 ({min_pct}%)."
        if not issues:
            summary = f"Target '{eval_col}' is well-balanced across all {n_unique} classes (Entropy: {normalized_entropy})."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(issues) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "target_column": eval_col,
                "classes_count": n_unique,
                "imbalance_ratio": imbalance_ratio,
                "minority_percentage": min_pct,
                "majority_percentage": maj_pct,
                "entropy": round(entropy, 3),
                "normalized_entropy": normalized_entropy
            },
            visual_data={
                "distribution": [
                    {"label": str(k), "count": int(v), "percentage": round(float(v / total_samples) * 100, 2)}
                    for k, v in value_counts.items()
                ]
            }
        )
