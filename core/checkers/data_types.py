"""
Incorrect Data Types & Mixed Type Checker
Identifies numeric values stored as formatted strings (currency, percentages), disguised dates, and mixed column data types.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import re
from models.schemas import CheckerResult, DimensionEnum, SeverityEnum
from core.checkers.base import BaseChecker


class DataTypeChecker(BaseChecker):
    checker_id = "data_types"
    checker_name = "Incorrect & Mixed Data Types Checker"
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

        issues = []
        dtype_mismatches = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            current_dtype = str(df[col].dtype)

            # 1. Check Object/String columns for Hidden Numerics (Currency, Commas, Percentages)
            if current_dtype in ['object', 'string', 'str'] or df[col].dtype == object or str(df[col].dtype).startswith('string') or str(df[col].dtype) == 'str':
                str_samples = series.astype(str).str.strip()
                
                # Check for currency format e.g. "$1,200.50" or "€45.00"
                currency_regex = re.compile(r'^[\$€£¥₹]?\s*-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$')
                matches_curr = int(str_samples.map(lambda s: bool(currency_regex.match(s))).sum())
                
                if matches_curr > len(series) * 0.75:
                    dtype_mismatches.append(col)
                    issues.append(self.create_issue(
                        issue_id=f"currency_as_string_{col}",
                        title=f"Numeric Currency Stored as String in '{col}'",
                        description=f"Column '{col}' has object dtype but {matches_curr:,} rows ({round(matches_curr/len(series)*100, 1)}%) follow currency/percentage format (e.g. '{series.iloc[0]}').",
                        severity=SeverityEnum.HIGH,
                        affected_columns=[col],
                        affected_rows_count=int(matches_curr),
                        affected_rows_percentage=round((matches_curr / n_rows) * 100, 2),
                        details={"sample_value": str(series.iloc[0]), "current_dtype": current_dtype, "suggested_dtype": "float64"},
                        remediation_suggestion=f"Strip currency/comma symbols and cast '{col}' to float64.",
                        suggested_code_snippet='df[\'' + col + '\'] = df[\'' + col + '\'].astype(str).str.replace(r"[\\$,]", "", regex=True).astype(float)',
                        auto_fixable=True,
                        fix_action=f"clean_numeric_string:{col}"
                    ))
                    continue

                # Check for disguised Datetimes
                # Pattern: YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, etc.
                date_regex = re.compile(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$')
                matches_date = int(str_samples.map(lambda s: bool(date_regex.match(s))).sum())
                if matches_date > len(series) * 0.8:
                    dtype_mismatches.append(col)
                    issues.append(self.create_issue(
                        issue_id=f"date_as_string_{col}",
                        title=f"Datetime Stored as String in '{col}'",
                        description=f"Column '{col}' contains date strings (e.g. '{series.iloc[0]}') stored as plain text.",
                        severity=SeverityEnum.MEDIUM,
                        affected_columns=[col],
                        affected_rows_count=int(matches_date),
                        details={"sample_value": str(series.iloc[0]), "current_dtype": current_dtype, "suggested_dtype": "datetime64[ns]"},
                        remediation_suggestion=f"Parse '{col}' with pd.to_datetime() to enable temporal feature engineering.",
                        suggested_code_snippet=f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')",
                        auto_fixable=True,
                        fix_action=f"convert_to_datetime:{col}"
                    ))
                    continue

                # Check for Mixed Types in single column (e.g. integers mixed with random words)
                types_in_col = set(type(x).__name__ for x in series.head(200))
                if len(types_in_col) > 1 and not ('str' in types_in_col and len(types_in_col) == 1):
                    issues.append(self.create_issue(
                        issue_id=f"mixed_types_{col}",
                        title=f"Mixed Data Types in '{col}' ({', '.join(types_in_col)})",
                        description=f"Column '{col}' holds conflicting python types ({', '.join(types_in_col)}), which triggers runtime serialization errors.",
                        severity=SeverityEnum.HIGH,
                        affected_columns=[col],
                        details={"types_found": list(types_in_col)},
                        remediation_suggestion="Standardize column values into a uniform string or numeric format.",
                        suggested_code_snippet=f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')",
                        auto_fixable=True,
                        fix_action=f"standardize_types:{col}"
                    ))

            # 2. Check for Low Cardinality Floats that should be Integers/Categories
            elif pd.api.types.is_float_dtype(df[col]):
                # If non-null values are all whole numbers with low cardinality
                is_whole = (series % 1 == 0).all()
                n_uniq = series.nunique()
                if is_whole and n_uniq <= 5 and n_rows > 50:
                    issues.append(self.create_issue(
                        issue_id=f"float_should_be_int_{col}",
                        title=f"Float Column '{col}' Contains Only {n_uniq} Discrete Integers",
                        description=f"Column '{col}' is stored as float64 but only has integer values {sorted(list(series.unique()))}.",
                        severity=SeverityEnum.LOW,
                        affected_columns=[col],
                        details={"unique_values": sorted(list(series.unique()))},
                        remediation_suggestion=f"Cast '{col}' to Int64 (nullable integer) or category.",
                        suggested_code_snippet=f"df['{col}'] = df['{col}'].astype('Int64')",
                        auto_fixable=True,
                        fix_action=f"cast_to_int:{col}"
                    ))

        # Calculate score
        penalty = min(50.0, (len(dtype_mismatches) * 12.0) + (len(issues) * 6.0))
        score = round(max(0.0, 100.0 - penalty), 1)

        summary = f"Audited data types across {len(df.columns)} columns. Found {len(issues)} type anomalies."
        if not issues:
            summary = "All column data types are optimal and consistent with their contents."

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
                "type_mismatches_count": len(dtype_mismatches),
                "total_type_issues": len(issues)
            }
        )
