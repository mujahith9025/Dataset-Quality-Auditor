"""
Suspicious & Sentinel Values Checker
Detects hidden sentinels (-999, 9999, '?', 'N/A'), domain range violations (negative ages/prices), and anomalous tokens.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import re
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class SuspiciousValuesChecker(BaseChecker):
    checker_id = "suspicious_values"
    checker_name = "Suspicious & Sentinel Values Checker"
    dimension = DimensionEnum.VALIDITY

    KNOWN_TEXT_SENTINELS = {
        "?", "??", "???", "na", "n/a", "none", "null", "nil", "unknown", 
        "missing", "nan", "undefined", "#n/a", "#value!", "#ref!", "test", "xxx", "-"
    }
    
    KNOWN_NUMERIC_SENTINELS = {-999, -9999, -1, 999, 9999, 99999, 8888, -99, 99}

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
        suspicious_columns_count = 0
        total_suspicious_values = 0

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            col_lower = col.lower()
            col_issues = []

            # 1. Inspect Object/Categorical columns for Text Sentinels
            if series.dtype == object or str(series.dtype) == 'category':
                str_series = series.astype(str).str.strip().str.lower()
                sentinel_mask = str_series.isin(self.KNOWN_TEXT_SENTINELS)
                text_sentinel_count = int(sentinel_mask.sum())
                
                if text_sentinel_count > 0:
                    pct = round((text_sentinel_count / len(series)) * 100, 2)
                    found_sentinels = list(set(series[sentinel_mask].astype(str).str.strip().tolist()))[:5]
                    col_issues.append({
                        "type": "text_sentinel",
                        "count": text_sentinel_count,
                        "pct": pct,
                        "examples": found_sentinels
                    })
                    issues.append(self.create_issue(
                        issue_id=f"text_sentinels_{col}",
                        title=f"Disguised Null Sentinels in '{col}' ({text_sentinel_count:,} values)",
                        description=f"Column '{col}' contains pseudo-null placeholder strings: {found_sentinels} ({pct}% of rows).",
                        severity=SeverityEnum.HIGH if pct > 5.0 else SeverityEnum.MEDIUM,
                        affected_columns=[col],
                        affected_rows_count=text_sentinel_count,
                        affected_rows_percentage=pct,
                        details={"sentinels_found": found_sentinels, "count": text_sentinel_count},
                        remediation_suggestion=f"Replace sentinel strings in '{col}' with np.nan so pandas recognizes them as missing.",
                        suggested_code_snippet=f"df['{col}'] = df['{col}'].replace({found_sentinels}, np.nan)",
                        auto_fixable=True,
                        fix_action=f"replace_sentinels_with_nan:{col}"
                    ))

            # 2. Inspect Numeric columns for Numeric Sentinels & Impossible Domain Ranges
            num_series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(num_series) > 0 and (pd.api.types.is_numeric_dtype(df[col]) or len(num_series) > len(series) * 0.7):
                
                # A. Numeric Sentinels (-999, 9999, etc.)
                for sentinel in self.KNOWN_NUMERIC_SENTINELS:
                    # Only flag if the sentinel is an extreme outlier compared to median
                    if (num_series == sentinel).any():
                        s_count = int((num_series == sentinel).sum())
                        median = float(num_series.median())
                        if abs(sentinel - median) > 20 and (sentinel < 0 or sentinel >= 999):
                            s_pct = round((s_count / len(num_series)) * 100, 2)
                            issues.append(self.create_issue(
                                issue_id=f"numeric_sentinel_{col}_{sentinel}",
                                title=f"Numeric Sentinel '{sentinel}' in '{col}' ({s_count:,} entries)",
                                description=f"Found {s_count:,} occurrences of placeholder code {sentinel} in column '{col}' ({s_pct}% of rows).",
                                severity=SeverityEnum.HIGH,
                                affected_columns=[col],
                                affected_rows_count=s_count,
                                affected_rows_percentage=s_pct,
                                details={"sentinel_value": sentinel, "count": s_count},
                                remediation_suggestion=f"Convert sentinel value {sentinel} to np.nan before modeling.",
                                suggested_code_snippet=f"df['{col}'] = df['{col}'].replace({sentinel}, np.nan)",
                                auto_fixable=True,
                                fix_action=f"replace_sentinels_with_nan:{col}"
                            ))

                # B. Domain Semantic Validity: Age / Years cannot be negative or > 125
                if any(k in col_lower for k in ["age", "years_old", "patient_age"]):
                    neg_ages = int((num_series < 0).sum())
                    exp_ages = int((num_series > 125).sum())
                    if neg_ages > 0 or exp_ages > 0:
                        total_bad = neg_ages + exp_ages
                        bad_pct = round((total_bad / len(num_series)) * 100, 2)
                        issues.append(self.create_issue(
                            issue_id=f"invalid_age_range_{col}",
                            title=f"Impossible Age Values in '{col}' ({total_bad:,} rows)",
                            description=f"Column '{col}' has {neg_ages} negative values and {exp_ages} values > 125 years (e.g. {sorted(list(num_series[num_series < 0]) + list(num_series[num_series > 125]))[:4]}).",
                            severity=SeverityEnum.HIGH,
                            affected_columns=[col],
                            affected_rows_count=total_bad,
                            affected_rows_percentage=bad_pct,
                            details={"negative_count": neg_ages, "excessive_count": exp_ages},
                            remediation_suggestion=f"Clean invalid age values in '{col}' by filtering to valid biological bounds [0, 115] or setting to NaN.",
                            suggested_code_snippet=f"df.loc[(df['{col}'] < 0) | (df['{col}'] > 120), '{col}'] = np.nan",
                            auto_fixable=True,
                            fix_action=f"sanitize_age_bounds:{col}"
                        ))

                # C. Domain Semantic Validity: Financial quantities (price, income, salary, charges) cannot be negative
                if any(k in col_lower for k in ["income", "salary", "price", "loan_amount", "charges", "revenue", "fare", "cost", "budget"]):
                    neg_financial = int((num_series < 0).sum())
                    if neg_financial > 0:
                        bad_pct = round((neg_financial / len(num_series)) * 100, 2)
                        issues.append(self.create_issue(
                            issue_id=f"negative_financial_{col}",
                            title=f"Negative Values in Financial Column '{col}' ({neg_financial:,} rows)",
                            description=f"Found {neg_financial:,} negative amounts in '{col}' ({bad_pct}%). Non-reversal financial amounts should strictly be positive.",
                            severity=SeverityEnum.HIGH,
                            affected_columns=[col],
                            affected_rows_count=neg_financial,
                            affected_rows_percentage=bad_pct,
                            details={"negative_count": neg_financial},
                            remediation_suggestion=f"Take absolute value or set negative values in '{col}' to NaN.",
                            suggested_code_snippet=f"df['{col}'] = df['{col}'].apply(lambda x: abs(x) if pd.notnull(x) else x)",
                            auto_fixable=True,
                            fix_action=f"abs_or_nan:{col}"
                        ))

                # D. Blood pressure / BMI sanity check
                if "bp" in col_lower or "blood_pressure" in col_lower or "bmi" in col_lower:
                    neg_med = int((num_series < 0).sum())
                    if neg_med > 0:
                        issues.append(self.create_issue(
                            issue_id=f"negative_medical_{col}",
                            title=f"Negative Biometric Values in '{col}' ({neg_med:,} rows)",
                            description=f"Column '{col}' has {neg_med:,} negative readings (impossible for blood pressure / BMI).",
                            severity=SeverityEnum.CRITICAL,
                            affected_columns=[col],
                            affected_rows_count=neg_med,
                            remediation_suggestion=f"Convert invalid negative medical readings in '{col}' to absolute values or NaN.",
                            suggested_code_snippet=f"df.loc[df['{col}'] < 0, '{col}'] = np.nan",
                            auto_fixable=True,
                            fix_action=f"abs_or_nan:{col}"
                        ))

            if col_issues or any(i.affected_columns == [col] for i in issues):
                suspicious_columns_count += 1

        # Calculate score
        penalty = min(50.0, len(issues) * 10.0)
        score = round(max(0.0, 100.0 - penalty), 1)

        summary = f"Audited all columns for sentinels & domain validity. Detected suspicious values across {len(issues)} check criteria."
        if not issues:
            summary = "No suspicious sentinels (-999, 'N/A') or impossible domain values detected."

        return CheckerResult(
            checker_id=self.checker_id,
            checker_name=self.checker_name,
            dimension=self.dimension,
            passed=len(issues) == 0,
            score=score,
            summary=summary,
            issues=issues,
            metrics={
                "columns_audited": len(df.columns),
                "suspicious_issues_count": len(issues),
                "suspicious_columns_count": suspicious_columns_count
            }
        )
