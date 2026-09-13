"""
Highly Correlated Features & Multicollinearity Checker
Computes pairwise correlation matrix, flags redundant collinear features (|r| > 0.85), and estimates multicollinearity.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class HighCorrelationChecker(BaseChecker):
    checker_id = "high_correlation"
    checker_name = "High Correlation & Multicollinearity Checker"
    dimension = DimensionEnum.CONSISTENCY

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
        if target_col and target_col in num_cols:
            # Exclude target column from feature-feature multicollinearity
            num_cols = [c for c in num_cols if c != target_col]

        if len(num_cols) < 2:
            return CheckerResult(
                checker_id=self.checker_id,
                checker_name=self.checker_name,
                dimension=self.dimension,
                passed=True,
                score=100.0,
                summary="Fewer than 2 numerical features available to compute correlation.",
                issues=[]
            )

        # Compute correlation matrix
        corr_matrix = df[num_cols].corr().fillna(0.0)
        
        issues = []
        high_corr_pairs = []
        redundant_cols = set()

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col_a = num_cols[i]
                col_b = num_cols[j]
                r_val = float(corr_matrix.loc[col_a, col_b])
                abs_r = abs(r_val)

                if abs_r >= 0.85:
                    high_corr_pairs.append({
                        "feature_1": col_a,
                        "feature_2": col_b,
                        "correlation": round(r_val, 4),
                        "abs_correlation": round(abs_r, 4)
                    })
                    redundant_cols.add(col_b)

                    severity = SeverityEnum.HIGH if abs_r >= 0.95 else SeverityEnum.MEDIUM
                    issues.append(self.create_issue(
                        issue_id=f"collinear_{col_a}_{col_b}",
                        title=f"High Correlation between '{col_a}' and '{col_b}' (|r| = {abs_r:.3f})",
                        description=f"Features '{col_a}' and '{col_b}' share a Pearson correlation of {r_val:.3f}. High multicollinearity destabilizes linear models and creates redundant feature weights.",
                        severity=severity,
                        affected_columns=[col_a, col_b],
                        details={"correlation": round(r_val, 4), "abs_correlation": round(abs_r, 4)},
                        remediation_suggestion=f"Drop one of the collinear features (e.g. '{col_b}') or apply PCA / feature engineering.",
                        suggested_code_snippet=f"df = df.drop(columns=['{col_b}'])",
                        auto_fixable=True,
                        fix_action=f"drop_column:{col_b}"
                    ))

        # Calculate score
        penalty = min(50.0, len(high_corr_pairs) * 12.0)
        score = round(max(0.0, 100.0 - penalty), 1)

        summary = f"Evaluated {len(num_cols)} numerical features. Found {len(high_corr_pairs)} highly collinear pairs (|r| >= 0.85)."
        if not high_corr_pairs:
            summary = "No high feature multicollinearity detected (all pairwise |r| < 0.85)."

        # Format matrix for UI visualization (capped at 15 columns for visual cleanliness)
        vis_cols = num_cols[:15]
        matrix_data = {
            "columns": vis_cols,
            "values": [[round(float(corr_matrix.loc[r, c]), 2) for c in vis_cols] for r in vis_cols]
        }

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(high_corr_pairs) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "features_evaluated": len(num_cols),
                "high_corr_pairs_count": len(high_corr_pairs),
                "redundant_features_count": len(redundant_cols),
                "high_corr_pairs": high_corr_pairs
            },
            visual_data={"correlation_matrix": matrix_data}
        )
