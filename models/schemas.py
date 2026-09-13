"""
Data models and schemas for Dataset Quality Auditor
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SeverityEnum(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DimensionEnum(str, Enum):
    COMPLETENESS = "Completeness"
    UNIQUENESS = "Uniqueness"
    VALIDITY = "Validity"
    CONSISTENCY = "Consistency"
    ML_READINESS = "ML Readiness"


class Issue(BaseModel):
    id: str
    dimension: DimensionEnum
    checker_id: str
    severity: SeverityEnum
    title: str
    description: str
    affected_columns: List[str] = Field(default_factory=list)
    affected_rows_count: Optional[int] = None
    affected_rows_percentage: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    remediation_suggestion: str
    suggested_code_snippet: Optional[str] = None
    auto_fixable: bool = False
    fix_action: Optional[str] = None  # e.g., 'drop_duplicates', 'impute_median', 'drop_column', etc.


class CheckerResult(BaseModel):
    checker_id: str
    checker_name: str
    dimension: DimensionEnum
    passed: bool
    score: float = 100.0  # 0.0 to 100.0
    summary: str
    issues: List[Issue] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visual_data: Optional[Dict[str, Any]] = None  # For frontend charts/heatmaps


class ColumnProfile(BaseModel):
    name: str
    physical_dtype: str
    inferred_type: str  # numeric, categorical, datetime, text, boolean, id, constant
    total_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    is_target: bool = False
    issues_count: int = 0
    issues: List[str] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension: DimensionEnum
    score: float
    weight: float
    weighted_score: float
    grade: str
    issues_count: Dict[str, int] = Field(default_factory=dict)  # severity -> count


class AuditReport(BaseModel):
    audit_id: str
    dataset_name: str
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    target_column: Optional[str] = None
    overall_score: float
    grade: str
    dimension_scores: Dict[str, DimensionScore]
    summary_alerts: List[str]
    total_issues_count: int
    severity_counts: Dict[str, int]
    checker_results: Dict[str, CheckerResult]
    column_profiles: Dict[str, ColumnProfile]
    recommended_actions: List[Dict[str, Any]]
    created_at: str
