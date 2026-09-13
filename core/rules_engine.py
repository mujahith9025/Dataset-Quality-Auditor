"""
Dataset Quality Auditor — Custom Business Rules & Assertions Engine
Declarative data assertion engine allowing domain experts and engineers to enforce custom
business rules, regex constraints, range bounds, and cross-column logic on tabular data.
"""

from typing import Dict, Any, List, Optional, Union
import re
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class RuleDefinition(BaseModel):
    rule_id: str
    rule_type: str  # "range", "not_null", "allowed_values", "regex", "unique", "comparison", "type_check"
    column: str
    description: str
    severity: str = "HIGH"  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    # Parameters for specific rule types:
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: List[str] = Field(default_factory=list)
    regex_pattern: Optional[str] = None
    comparison_column: Optional[str] = None
    comparison_operator: Optional[str] = None  # ">=", "<=", ">", "<", "=="
    expected_type: Optional[str] = None  # "numeric", "string", "datetime", "boolean"


class RuleExecutionResult(BaseModel):
    rule: RuleDefinition
    status: str  # "PASSED" or "FAILED"
    total_evaluated: int
    passed_count: int
    failed_count: int
    pass_rate_pct: float
    violating_indices: List[int] = Field(default_factory=list)
    sample_violations: List[Dict[str, Any]] = Field(default_factory=list)
    message: str


class AssertionSuiteResult(BaseModel):
    total_rules: int
    passed_rules: int
    failed_rules: int
    overall_pass_rate: float
    status: str  # "PASSED" or "FAILED"
    results: List[RuleExecutionResult]


class RulesEngine:
    """
    Executes declarative assertions on pandas DataFrames.
    """

    def evaluate_rule(self, df: pd.DataFrame, rule: RuleDefinition) -> RuleExecutionResult:
        if rule.column not in df.columns:
            return RuleExecutionResult(
                rule=rule,
                status="FAILED",
                total_evaluated=len(df),
                passed_count=0,
                failed_count=len(df),
                pass_rate_pct=0.0,
                message=f"Column '{rule.column}' does not exist in dataset."
            )

        series = df[rule.column]
        total_rows = len(df)
        mask_passed = pd.Series(True, index=df.index)

        try:
            if rule.rule_type == "not_null":
                # Must not be null or empty string
                mask_passed = series.notna() & (series.astype(str).str.strip() != "")

            elif rule.rule_type == "range":
                num_series = pd.to_numeric(series, errors='coerce')
                valid_num = num_series.notna()
                cond = pd.Series(True, index=df.index)
                if rule.min_value is not None:
                    cond = cond & (num_series >= rule.min_value)
                if rule.max_value is not None:
                    cond = cond & (num_series <= rule.max_value)
                mask_passed = valid_num & cond

            elif rule.rule_type == "allowed_values":
                allowed_set = set(str(v).strip().lower() for v in rule.allowed_values)
                str_series = series.astype(str).str.strip().str.lower()
                mask_passed = str_series.isin(allowed_set)

            elif rule.rule_type == "regex":
                if rule.regex_pattern:
                    try:
                        compiled = re.compile(rule.regex_pattern)
                        mask_passed = series.astype(str).apply(lambda x: bool(compiled.search(str(x))) if pd.notna(x) else False)
                    except re.error as e:
                        return RuleExecutionResult(
                            rule=rule,
                            status="FAILED",
                            total_evaluated=len(df),
                            passed_count=0,
                            failed_count=len(df),
                            pass_rate_pct=0.0,
                            message=f"Invalid regular expression '{rule.regex_pattern}': {e}"
                        )

            elif rule.rule_type == "unique":
                mask_passed = ~series.duplicated(keep=False)

            elif rule.rule_type == "comparison":
                if rule.comparison_column and rule.comparison_column in df.columns:
                    col_a = pd.to_numeric(series, errors='coerce')
                    col_b = pd.to_numeric(df[rule.comparison_column], errors='coerce')
                    op = rule.comparison_operator or ">="
                    if op == ">=":
                        mask_passed = col_a >= col_b
                    elif op == "<=":
                        mask_passed = col_a <= col_b
                    elif op == ">":
                        mask_passed = col_a > col_b
                    elif op == "<":
                        mask_passed = col_a < col_b
                    elif op == "==":
                        mask_passed = col_a == col_b
                    else:
                        mask_passed = col_a >= col_b
                else:
                    mask_passed = pd.Series(False, index=df.index)

            elif rule.rule_type == "type_check":
                if rule.expected_type == "numeric":
                    mask_passed = pd.to_numeric(series, errors='coerce').notna()
                elif rule.expected_type == "datetime":
                    mask_passed = pd.to_datetime(series, errors='coerce').notna()
                else:
                    mask_passed = series.notna()

        except Exception as e:
            mask_passed = pd.Series(False, index=df.index)

        passed_count = int(mask_passed.sum())
        failed_count = total_rows - passed_count
        pass_rate = round((passed_count / total_rows) * 100.0, 2) if total_rows > 0 else 100.0

        violating_indices = df[~mask_passed].index.tolist()
        sample_violations = df.loc[violating_indices[:5]].to_dict(orient='records') if violating_indices else []

        status = "PASSED" if failed_count == 0 else "FAILED"
        if status == "PASSED":
            msg = f"All {total_rows:,} rows satisfied constraint: '{rule.description}'"
        else:
            msg = f"{failed_count:,} out of {total_rows:,} rows violated constraint ({pass_rate:.1f}% pass rate)."

        return RuleExecutionResult(
            rule=rule,
            status=status,
            total_evaluated=total_rows,
            passed_count=passed_count,
            failed_count=failed_count,
            pass_rate_pct=pass_rate,
            violating_indices=violating_indices[:500],
            sample_violations=sample_violations,
            message=msg
        )

    def run_suite(self, df: pd.DataFrame, rules: List[RuleDefinition]) -> AssertionSuiteResult:
        results = [self.evaluate_rule(df, r) for r in rules]
        passed_rules = sum(1 for r in results if r.status == "PASSED")
        failed_rules = len(results) - passed_rules
        overall_pass_rate = round((passed_rules / len(results)) * 100.0, 1) if results else 100.0

        return AssertionSuiteResult(
            total_rules=len(rules),
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            overall_pass_rate=overall_pass_rate,
            status="PASSED" if failed_rules == 0 else "FAILED",
            results=results
        )

    @staticmethod
    def get_template_rules(template_name: str, df: pd.DataFrame) -> List[RuleDefinition]:
        """
        Provides pre-built business rule templates tailored to standard tabular datasets.
        """
        cols = list(df.columns)
        rules = []

        if template_name == "fintech":
            # Financial & Lending Rules
            for c in cols:
                c_low = c.lower()
                if "age" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="fin_age_range",
                        rule_type="range",
                        column=c,
                        description="Applicant age must be between 18 and 100",
                        severity="CRITICAL",
                        min_value=18.0,
                        max_value=100.0
                    ))
                elif "income" in c_low or "salary" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="fin_income_positive",
                        rule_type="range",
                        column=c,
                        description="Income must be strictly non-negative",
                        severity="HIGH",
                        min_value=0.0
                    ))
                elif "id" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="fin_id_unique",
                        rule_type="unique",
                        column=c,
                        description="Identifier key must be globally unique",
                        severity="CRITICAL"
                    ))
        elif template_name == "ecommerce":
            # E-Commerce & Retail
            for c in cols:
                c_low = c.lower()
                if "price" in c_low or "amount" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="ecom_price_positive",
                        rule_type="range",
                        column=c,
                        description="Product price/amount must be > 0",
                        severity="CRITICAL",
                        min_value=0.01
                    ))
                elif "rating" in c_low or "score" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="ecom_rating_bounds",
                        rule_type="range",
                        column=c,
                        description="Rating must be between 1.0 and 5.0",
                        severity="MEDIUM",
                        min_value=1.0,
                        max_value=5.0
                    ))
                elif "email" in c_low:
                    rules.append(RuleDefinition(
                        rule_id="ecom_email_regex",
                        rule_type="regex",
                        column=c,
                        description="Customer email must follow valid email format",
                        severity="HIGH",
                        regex_pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"
                    ))
        else:
            # General Data Hygiene
            for c in cols[:4]:
                rules.append(RuleDefinition(
                    rule_id=f"hygiene_not_null_{c}",
                    rule_type="not_null",
                    column=c,
                    description=f"Column '{c}' must not contain missing or null values",
                    severity="MEDIUM"
                ))

        return rules
