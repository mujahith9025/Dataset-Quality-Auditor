"""
Dataset Quality Auditor — Data Drift & Train-vs-Test Comparison Module
Detects covariate shift, concept drift, and data pipeline regressions between baseline (Train)
and target (Test/Production) datasets using PSI, KS-tests, and novel category discovery.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between two continuous distributions.
    Standard interpretation:
      PSI < 0.10: Stable / No significant change
      0.10 <= PSI < 0.25: Moderate change / Warning
      PSI >= 0.25: Significant change / Action required
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantiles based on expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(expected, percentiles)
        bin_edges = np.unique(bin_edges)  # handle duplicates
        if len(bin_edges) <= 1:
            return 0.0
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Replace zeros with a small epsilon to avoid division by zero or log(0)
    eps = 1e-4
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    # Re-normalize
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, round(psi_val, 4)))


def ks_2samp_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """
    Computes Kolmogorov-Smirnov 2-sample statistic and asymptotic p-value.
    Pure-numpy implementation to prevent external dependencies.
    """
    d1 = np.sort(data1[~np.isnan(data1)])
    d2 = np.sort(data2[~np.isnan(data2)])

    n1 = len(d1)
    n2 = len(d2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    data_all = np.concatenate([d1, d2])
    cdf1 = np.searchsorted(d1, data_all, side='right') / n1
    cdf2 = np.searchsorted(d2, data_all, side='right') / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    
    # Asymptotic p-value approximation
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    
    # Kolmogorov distribution CDF approximation
    if lambda_val <= 0:
        p_val = 1.0
    else:
        # Sum first few terms of Kolmogorov series
        p_val = 2.0 * sum(((-1) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * (lambda_val ** 2)) for k in range(1, 10))
        p_val = max(0.0, min(1.0, float(p_val)))

    return round(d_stat, 4), round(p_val, 4)


class ColumnDriftResult(BaseModel):
    column_name: str
    column_type: str  # "numeric" or "categorical"
    drift_status: str  # "STABLE", "MODERATE_DRIFT", "CRITICAL_DRIFT"
    psi_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    mean_reference: Optional[float] = None
    mean_current: Optional[float] = None
    mean_diff_pct: Optional[float] = None
    novel_categories: List[str] = Field(default_factory=list)
    total_variation_distance: Optional[float] = None
    details: str = ""


class DatasetDriftReport(BaseModel):
    reference_dataset_name: str
    current_dataset_name: str
    reference_rows: int
    current_rows: int
    shared_columns_count: int
    overall_drift_score: float  # 0 to 100 (0 = identical, 100 = total drift)
    overall_status: str  # "STABLE", "MODERATE_DRIFT", "CRITICAL_DRIFT"
    critical_columns_count: int
    moderate_columns_count: int
    stable_columns_count: int
    column_reports: Dict[str, ColumnDriftResult]
    summary_alerts: List[str] = Field(default_factory=list)


class DriftDetector:
    """
    Analyzes and compares two datasets for distributional shift, new classes, and pipeline drift.
    """

    def compare_datasets(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        ref_name: str = "Reference (Train)",
        curr_name: str = "Current (Test)"
    ) -> DatasetDriftReport:
        shared_cols = [c for c in ref_df.columns if c in curr_df.columns]
        if not shared_cols:
            raise ValueError("No common columns found between reference and current datasets.")

        column_reports: Dict[str, ColumnDriftResult] = {}
        drift_scores = []
        alerts = []

        critical_count = 0
        moderate_count = 0
        stable_count = 0

        for col in shared_cols:
            ref_series = ref_df[col].dropna()
            curr_series = curr_df[col].dropna()

            is_numeric = pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(curr_df[col])

            if is_numeric and ref_series.nunique() > 5:
                # Numeric Feature Drift Check
                ref_vals = ref_series.to_numpy(dtype=float)
                curr_vals = curr_series.to_numpy(dtype=float)

                psi = calculate_psi(ref_vals, curr_vals)
                ks_stat, ks_p = ks_2samp_test(ref_vals, curr_vals)

                m_ref = float(ref_vals.mean()) if len(ref_vals) > 0 else 0.0
                m_curr = float(curr_vals.mean()) if len(curr_vals) > 0 else 0.0
                mean_diff = ((m_curr - m_ref) / (abs(m_ref) + 1e-6)) * 100

                if psi >= 0.25 or (ks_stat > 0.35 and ks_p < 0.01):
                    status = "CRITICAL_DRIFT"
                    critical_count += 1
                    drift_scores.append(100.0)
                    alerts.append(f"🚨 **Critical Drift in '{col}'**: PSI={psi:.3f}, KS={ks_stat:.3f} (Mean shifted by {mean_diff:+.1f}%).")
                elif psi >= 0.10 or (ks_stat > 0.20 and ks_p < 0.05):
                    status = "MODERATE_DRIFT"
                    moderate_count += 1
                    drift_scores.append(50.0)
                    alerts.append(f"⚠️ **Moderate Shift in '{col}'**: PSI={psi:.3f}, KS={ks_stat:.3f}.")
                else:
                    status = "STABLE"
                    stable_count += 1
                    drift_scores.append(0.0)

                column_reports[col] = ColumnDriftResult(
                    column_name=col,
                    column_type="numeric",
                    drift_status=status,
                    psi_score=psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    mean_reference=round(m_ref, 3),
                    mean_current=round(m_curr, 3),
                    mean_diff_pct=round(mean_diff, 1),
                    details=f"Ref Mean: {m_ref:.2f} -> Curr Mean: {m_curr:.2f} (Delta {mean_diff:+.1f}%)"
                )

            else:
                # Categorical Feature Drift Check
                ref_cats = set(ref_series.astype(str).unique())
                curr_cats = set(curr_series.astype(str).unique())
                novel_cats = list(curr_cats - ref_cats)

                # Total Variation Distance on category frequency
                ref_freq = ref_series.astype(str).value_counts(normalize=True)
                curr_freq = curr_series.astype(str).value_counts(normalize=True)
                all_cats = list(ref_cats.union(curr_cats))
                tvd = 0.5 * sum(abs(ref_freq.get(c, 0.0) - curr_freq.get(c, 0.0)) for c in all_cats)
                tvd = float(round(tvd, 4))

                if len(novel_cats) > 0 or tvd > 0.35:
                    status = "CRITICAL_DRIFT" if (len(novel_cats) >= 3 or tvd > 0.5) else "MODERATE_DRIFT"
                    if status == "CRITICAL_DRIFT":
                        critical_count += 1
                        drift_scores.append(100.0)
                    else:
                        moderate_count += 1
                        drift_scores.append(50.0)
                    
                    cat_msg = f", {len(novel_cats)} unseen novel categories ({', '.join(novel_cats[:3])})" if novel_cats else ""
                    alerts.append(f"⚠️ **Categorical Shift in '{col}'**: TVD={tvd:.2f}{cat_msg}.")
                else:
                    status = "STABLE"
                    stable_count += 1
                    drift_scores.append(0.0)

                column_reports[col] = ColumnDriftResult(
                    column_name=col,
                    column_type="categorical",
                    drift_status=status,
                    novel_categories=novel_cats[:10],
                    total_variation_distance=tvd,
                    details=f"TVD: {tvd:.2f}" + (f" | Unseen: {novel_cats[:3]}" if novel_cats else " | Stable classes")
                )

        overall_score = float(round(np.mean(drift_scores) if drift_scores else 0.0, 1))
        if critical_count > 0 or overall_score >= 40:
            overall_status = "CRITICAL_DRIFT"
        elif moderate_count > 0 or overall_score >= 15:
            overall_status = "MODERATE_DRIFT"
        else:
            overall_status = "STABLE"

        if not alerts:
            alerts.append("✅ **No significant drift detected.** Feature distributions match closely between baseline and target datasets.")

        return DatasetDriftReport(
            reference_dataset_name=ref_name,
            current_dataset_name=curr_name,
            reference_rows=len(ref_df),
            current_rows=len(curr_df),
            shared_columns_count=len(shared_cols),
            overall_drift_score=overall_score,
            overall_status=overall_status,
            critical_columns_count=critical_count,
            moderate_columns_count=moderate_count,
            stable_columns_count=stable_count,
            column_reports=column_reports,
            summary_alerts=alerts
        )
