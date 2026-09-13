"""
Base Checker Interface for Dataset Quality Auditor
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from models.schemas import CheckerResult, Issue, DimensionEnum, SeverityEnum


class BaseChecker(ABC):
    """Abstract base class for all audit checkers."""
    
    checker_id: str = "base_checker"
    checker_name: str = "Base Checker"
    dimension: DimensionEnum = DimensionEnum.COMPLETENESS

    @abstractmethod
    def check(self, df: pd.DataFrame, target_col: Optional[str] = None) -> CheckerResult:
        """Executes the audit check on the provided dataframe."""
        pass

    def create_issue(
        self,
        issue_id: str,
        title: str,
        description: str,
        severity: SeverityEnum,
        affected_columns: Optional[List[str]] = None,
        affected_rows_count: Optional[int] = None,
        affected_rows_percentage: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        remediation_suggestion: str = "",
        suggested_code_snippet: Optional[str] = None,
        auto_fixable: bool = False,
        fix_action: Optional[str] = None
    ) -> Issue:
        """Helper to create a structured Issue."""
        return Issue(
            id=f"{self.checker_id}_{issue_id}",
            dimension=self.dimension,
            checker_id=self.checker_id,
            severity=severity,
            title=title,
            description=description,
            affected_columns=affected_columns or [],
            affected_rows_count=affected_rows_count,
            affected_rows_percentage=affected_rows_percentage,
            details=details or {},
            remediation_suggestion=remediation_suggestion,
            suggested_code_snippet=suggested_code_snippet,
            auto_fixable=auto_fixable,
            fix_action=fix_action
        )

    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of truly numeric column names."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of object/category/string column names."""
        return df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
