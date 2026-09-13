"""
Auto-Remediation Engine
Applies automated data cleaning recipes based on audit findings and tracks score improvements.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import re

from models.schemas import AuditReport, Issue, SeverityEnum


class DatasetCleaner:
    """Applies auto-remediation transformations to clean tabular datasets."""

    def __init__(self):
        self.log: List[str] = []

    def clean(
        self,
        df: pd.DataFrame,
        selected_actions: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        report: Optional[AuditReport] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes remediation actions on a copy of the dataframe.
        If selected_actions is None, auto-applies all safe recommended fixes from the report.
        """
        cleaned_df = df.copy()
        self.log = []
        initial_shape = df.shape

        # If no explicit actions provided, collect from report or run standard cleaning
        if selected_actions is None and report is not None:
            selected_actions = [
                rec["fix_action"] for rec in report.recommended_actions
                if rec.get("auto_fixable") and rec.get("fix_action")
            ]
        elif selected_actions is None:
            # Discover and execute all standard fixes
            from core.engine import AuditEngine
            temp_report = AuditEngine().audit(cleaned_df, target_col=target_col)
            selected_actions = [
                rec["fix_action"] for rec in temp_report.recommended_actions
                if rec.get("auto_fixable") and rec.get("fix_action")
            ]

        # De-duplicate actions while preserving order
        actions_to_run = []
        for act in selected_actions:
            if act and act not in actions_to_run:
                actions_to_run.append(act)

        # 1. First Pass: Drop Duplicates
        if "drop_exact_duplicates" in actions_to_run:
            before_cnt = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(keep="first").reset_index(drop=True)
            dropped = before_cnt - len(cleaned_df)
            if dropped > 0:
                self.log.append(f"Removed {dropped:,} exact duplicate rows.")

        # 2. Drop Target NA
        for act in actions_to_run:
            if act.startswith("drop_target_na:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    before_cnt = len(cleaned_df)
                    cleaned_df = cleaned_df.dropna(subset=[col]).reset_index(drop=True)
                    dropped = before_cnt - len(cleaned_df)
                    if dropped > 0:
                        self.log.append(f"Dropped {dropped:,} rows with missing target values in '{col}'.")

        # 3. Drop Sparse Rows (>50% missing)
        if "drop_sparse_rows" in actions_to_run:
            thresh = int(cleaned_df.shape[1] * 0.5)
            before_cnt = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(thresh=thresh).reset_index(drop=True)
            dropped = before_cnt - len(cleaned_df)
            if dropped > 0:
                self.log.append(f"Dropped {dropped:,} severely sparse rows (>50% missing columns).")

        # 4. Drop Duplicates by Key
        for act in actions_to_run:
            if act.startswith("drop_duplicates_by_key:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    before_cnt = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)
                    dropped = before_cnt - len(cleaned_df)
                    if dropped > 0:
                        self.log.append(f"Deduplicated {dropped:,} records with duplicate key in '{col}'.")

        # 5. Replace Sentinels with NaN
        for act in actions_to_run:
            if act.startswith("replace_sentinels_with_nan:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    sentinels_text = ["?", "??", "na", "n/a", "none", "null", "unknown", "missing", "#n/a", "#value!", "-", " "]
                    sentinels_num = [-999, -9999, 9999, 99999]
                    
                    # Text sentinels
                    if cleaned_df[col].dtype == object or str(cleaned_df[col].dtype).startswith("str"):
                        mask = cleaned_df[col].astype(str).str.strip().str.lower().isin(sentinels_text)
                        cleaned_df.loc[mask, col] = np.nan
                    
                    # Numeric sentinels
                    cleaned_df[col] = cleaned_df[col].replace(sentinels_num, np.nan)
                    self.log.append(f"Replaced disguised sentinels with NaN in '{col}'.")

        # 6. Sanitize Bounds & Invalid Values
        for act in actions_to_run:
            if act.startswith("sanitize_age_bounds:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    num_col = pd.to_numeric(cleaned_df[col], errors='coerce')
                    cleaned_df.loc[(num_col < 0) | (num_col > 120), col] = np.nan
                    self.log.append(f"Sanitized out-of-bound age values in '{col}'.")

            elif act.startswith("abs_or_nan:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    num_col = pd.to_numeric(cleaned_df[col], errors='coerce')
                    cleaned_df[col] = num_col.abs()
                    self.log.append(f"Converted negative financial/biometric values to positive in '{col}'.")

        # 7. Clean String-Wrapped Numbers & Datetimes
        for act in actions_to_run:
            if act.startswith("clean_numeric_string:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    cleaned_df[col] = (
                        cleaned_df[col]
                        .astype(str)
                        .str.replace(r'[\$,€£¥₹\s%]', '', regex=True)
                    )
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    self.log.append(f"Cleaned formatted currency/symbols in '{col}' and converted to numeric float64.")

            elif act.startswith("convert_to_datetime:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    self.log.append(f"Parsed text column '{col}' into datetime format.")

            elif act.startswith("cast_to_int:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').astype('Int64')
                    self.log.append(f"Cast discrete float column '{col}' to Int64.")

        # 8. Clip Extreme Outliers
        for act in actions_to_run:
            if act.startswith("clip_outliers:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns:
                    num_s = pd.to_numeric(cleaned_df[col], errors='coerce')
                    if num_s.notna().sum() > 10:
                        q_low = float(num_s.quantile(0.01))
                        q_high = float(num_s.quantile(0.99))
                        cleaned_df[col] = num_s.clip(q_low, q_high)
                        self.log.append(f"Clipped extreme outliers in '{col}' to 1st-99th percentile bounds [{q_low:,.2f}, {q_high:,.2f}].")

        # 9. Smart Imputation (Median for numeric, Mode for categorical)
        for act in actions_to_run:
            if act.startswith("impute:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns and cleaned_df[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        med_val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(med_val)
                        self.log.append(f"Imputed missing numeric values in '{col}' with median ({med_val:,.2f}).")
                    else:
                        mode_s = cleaned_df[col].mode()
                        mode_val = mode_s.iloc[0] if not mode_s.empty else "Missing"
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        self.log.append(f"Imputed missing categorical values in '{col}' with mode ('{mode_val}').")

        # 10. Drop Target Leakage / Constant / Redundant Columns (Applied last so previous steps can clean features)
        cols_to_drop = []
        for act in actions_to_run:
            if act.startswith("drop_column:"):
                col = act.split(":", 1)[1]
                if col in cleaned_df.columns and col not in cols_to_drop and col != target_col:
                    cols_to_drop.append(col)

        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            self.log.append(f"Dropped {len(cols_to_drop)} problematic columns: {cols_to_drop}")

        # Summary of transformation
        stats = {
            "initial_rows": initial_shape[0],
            "initial_columns": initial_shape[1],
            "cleaned_rows": cleaned_df.shape[0],
            "cleaned_columns": cleaned_df.shape[1],
            "rows_removed": initial_shape[0] - cleaned_df.shape[0],
            "columns_removed": initial_shape[1] - cleaned_df.shape[1],
            "actions_executed": len(self.log),
            "execution_log": self.log
        }

        return cleaned_df, stats
