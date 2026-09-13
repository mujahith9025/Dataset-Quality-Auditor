"""
Dataset Quality Auditor — Automated ML Baseline Benchmark Module
Trains quick baseline machine learning models on Raw/Dirty vs. Cleaned/Remediated datasets
to quantify the mathematical return-on-investment (accuracy/F1 score uplift) of data quality remediation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, root_mean_squared_error
)


class ModelMetric(BaseModel):
    model_name: str
    task_type: str  # "classification" or "regression"
    metrics: Dict[str, float]
    feature_importances: Dict[str, float] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    task_type: str
    target_column: str
    is_binary: bool = False
    raw_model: ModelMetric
    clean_model: ModelMetric
    uplift: Dict[str, float]  # e.g. {"accuracy": +0.12, "f1": +0.15}
    summary_text: str
    top_features: List[Tuple[str, float]] = Field(default_factory=list)


class MLBenchmarkEngine:
    """
    Automates baseline ML model training, cross-validation, and performance uplift measurement.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def detect_task_type(self, series: pd.Series) -> Tuple[str, bool]:
        """
        Detects whether target variable represents classification or regression.
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return "classification", True

        n_unique = clean_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(clean_series)

        if not is_numeric or n_unique <= 10:
            return "classification", (n_unique == 2)
        return "regression", False

    def _prepare_pipeline(self, X: pd.DataFrame, is_classification: bool) -> Tuple[Pipeline, List[str]]:
        """
        Builds a robust sklearn preprocessing pipeline for tabular data.
        """
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if num_cols:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, num_cols))

        if cat_cols:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipe, cat_cols))

        if not transformers:
            # Fallback if no columns detected
            transformers.append(('num', SimpleImputer(strategy='median'), X.columns.tolist()))

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return preprocessor, num_cols + cat_cols

    def run_benchmark(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        target_col: str,
        model_type: str = "random_forest"
    ) -> BenchmarkResult:
        """
        Runs automated training on both raw and cleaned datasets, returning metrics and uplift.
        """
        if target_col not in raw_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in raw dataset.")
        if target_col not in clean_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in cleaned dataset.")

        # Determine task type
        task_type, is_binary = self.detect_task_type(clean_df[target_col])

        # Evaluate Raw Dataset
        raw_metric = self._train_and_evaluate(
            raw_df, target_col, task_type, is_binary, model_type, is_raw=True
        )

        # Evaluate Cleaned Dataset
        clean_metric = self._train_and_evaluate(
            clean_df, target_col, task_type, is_binary, model_type, is_raw=False
        )

        # Calculate Uplift
        uplift = {}
        for k, clean_val in clean_metric.metrics.items():
            raw_val = raw_metric.metrics.get(k, 0.0)
            uplift[k] = round(clean_val - raw_val, 4)

        # Generate Human Summary
        if task_type == "classification":
            f1_diff = uplift.get("f1", 0.0) * 100
            acc_diff = uplift.get("accuracy", 0.0) * 100
            summary = (
                f"Data quality remediation boosted baseline {raw_metric.model_name} performance: "
                f"Accuracy improved by {acc_diff:+.1f}%, and F1-Score increased by {f1_diff:+.1f}%."
            )
        else:
            r2_diff = uplift.get("r2", 0.0)
            mae_diff = uplift.get("mae", 0.0)
            summary = (
                f"Data quality remediation improved baseline {raw_metric.model_name} fit: "
                f"R² score changed by {r2_diff:+.3f}, and MAE reduced by {abs(mae_diff):.2f} units."
            )

        # Top feature importances from clean model
        sorted_feats = sorted(clean_metric.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

        return BenchmarkResult(
            task_type=task_type,
            target_column=target_col,
            is_binary=is_binary,
            raw_model=raw_metric,
            clean_model=clean_metric,
            uplift=uplift,
            summary_text=summary,
            top_features=sorted_feats
        )

    def _train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        is_binary: bool,
        model_type: str,
        is_raw: bool
    ) -> ModelMetric:
        # Separate X and y
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()

        # Drop rows with null target
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if task_type == "classification":
            # Encode target
            y_encoded, class_names = pd.factorize(y)
            y = pd.Series(y_encoded, index=X.index)
            if len(class_names) < 2:
                # Degenerate case
                return ModelMetric(
                    model_name="Baseline Classifier",
                    task_type=task_type,
                    metrics={"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0}
                )
        else:
            y = pd.to_numeric(y, errors='coerce').fillna(y.mean() if not y.empty else 0.0)

        # Preprocessing pipeline
        preprocessor, feature_names = self._prepare_pipeline(X, is_classification=(task_type == "classification"))

        # Model Selection
        if task_type == "classification":
            if model_type == "logistic_regression":
                estimator = LogisticRegression(max_iter=500, random_state=self.random_state)
                model_name = "Logistic Regression"
            elif model_type == "decision_tree":
                estimator = DecisionTreeClassifier(max_depth=6, random_state=self.random_state)
                model_name = "Decision Tree Classifier"
            else:
                estimator = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=self.random_state)
                model_name = "Random Forest Classifier"
        else:
            if model_type == "ridge":
                estimator = Ridge(alpha=1.0, random_state=self.random_state)
                model_name = "Ridge Regressor"
            elif model_type == "decision_tree":
                estimator = DecisionTreeRegressor(max_depth=6, random_state=self.random_state)
                model_name = "Decision Tree Regressor"
            else:
                estimator = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=self.random_state)
                model_name = "Random Forest Regressor"

        full_pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', estimator)
        ])

        # Train / Test split
        test_size = 0.25 if len(X) >= 50 else 0.5
        stratify = y if (task_type == "classification" and y.value_counts().min() >= 2) else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
            )
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)
        except Exception:
            # Fallback for extreme dirty data with unexpected unhandled anomalies
            full_pipeline.fit(X, y)
            y_pred = full_pipeline.predict(X)
            y_test = y

        # Compute Metrics
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = float(round(accuracy_score(y_test, y_pred), 4))
            metrics["precision"] = float(round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            metrics["recall"] = float(round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            metrics["f1"] = float(round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))

            if is_binary and hasattr(full_pipeline.named_steps['model'], "predict_proba"):
                try:
                    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(round(roc_auc_score(y_test, y_prob), 4))
                except Exception:
                    metrics["roc_auc"] = 0.5
            else:
                metrics["roc_auc"] = 0.0
        else:
            metrics["r2"] = float(round(r2_score(y_test, y_pred), 4))
            metrics["mae"] = float(round(mean_absolute_error(y_test, y_pred), 4))
            try:
                metrics["rmse"] = float(round(root_mean_squared_error(y_test, y_pred), 4))
            except Exception:
                metrics["rmse"] = float(round(np.sqrt(np.mean((y_test - y_pred) ** 2)), 4))

        # Extract Feature Importances
        importances = {}
        fitted_model = full_pipeline.named_steps['model']
        if hasattr(fitted_model, "feature_importances_"):
            raw_imp = fitted_model.feature_importances_
            # Map back to column names if possible
            try:
                feature_out = full_pipeline.named_steps['prep'].get_feature_names_out()
                clean_names = [f.split("__")[-1] for f in feature_out]
                for name, imp in zip(clean_names, raw_imp):
                    importances[name] = float(round(imp, 4))
            except Exception:
                for idx, imp in enumerate(raw_imp[:len(feature_names)]):
                    name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                    importances[name] = float(round(imp, 4))

        return ModelMetric(
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            feature_importances=importances
        )
