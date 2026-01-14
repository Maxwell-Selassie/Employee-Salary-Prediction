from __future__ import annotations

"""
Inference service for Employee Salary Prediction.

Responsibilities:
- Load preprocessing pipeline (model-as-code: transformations + config)
- Load best production model from MLflow Model Registry via alias
- Provide single-row and batch prediction utilities
- Provide SHAP-based explanations for business-facing apps
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap

from utils import load_joblib
from utils.mlflow_utils import load_production_model, mlflow_stage_run


class InferenceService:
    """
    High-level inference service used by FastAPI and Streamlit.
    """

    def __init__(
        self,
        model_name: str = "EmployeeSalaryModel",
        alias: str = "production",
        preprocessing_pipeline_path: str = "artifacts/data/preprocessing_pipeline.joblib",
        target_column: str = "Current_Salary_log",
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self.target_column = target_column

        pipeline_path = Path(preprocessing_pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at {pipeline_path}. "
                "Run the preprocessing pipeline before serving."
            )

        self.pipeline_obj: Dict[str, Any] = load_joblib(pipeline_path)
        self.model = load_production_model(model_name, alias)

        # Optional cached feature names for SHAP
        self.feature_names: List[str] = self.pipeline_obj.get("features", {}).get("names", [])

        # SHAP explainer cache
        self._shap_explainer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _transform_input(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing steps used during training.
        """
        df = df_raw.copy()

        missing_handler = self.pipeline_obj.get("missing_handler")
        outlier_handler = self.pipeline_obj.get("outlier_handler")
        encoder = self.pipeline_obj.get("encoder")
        transformer = self.pipeline_obj.get("transformer")

        if missing_handler is not None:
            df = missing_handler.handle_missing(df, fit=False)
        if outlier_handler is not None:
            df = outlier_handler.handle_outliers(df, fit=False)
        if encoder is not None:
            df = encoder.encode_features(df, fit=False)
        if transformer is not None:
            df = transformer.transform_features(df, fit=False)

        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        return df

    def _get_shap_explainer(self):
        """
        Lazily construct a SHAP explainer for the loaded model.
        """
        if self._shap_explainer is not None:
            return self._shap_explainer

        try:
            self._shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to model-agnostic explainer
            background = np.zeros((10, len(self.feature_names))) if self.feature_names else np.zeros((10, 1))
            self._shap_explainer = shap.KernelExplainer(self.model.predict, background)
        return self._shap_explainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_single(self, features: Dict[str, Any]) -> float:
        """
        Predict salary for a single row of features.
        """
        df_raw = pd.DataFrame([features])
        with mlflow_stage_run("inference", "predict_single", tags={"interface": "api"}):
            X = self._transform_input(df_raw)
            y_log = self.model.predict(X)[0]
            y = float(np.exp(y_log))
        return y

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.Series:
        """
        Predict salaries for a batch of rows.
        """
        with mlflow_stage_run("inference", "predict_batch", tags={"interface": "api_or_batch"}):
            X = self._transform_input(df_raw)
            y_log = self.model.predict(X)
            y = np.exp(y_log)
        return pd.Series(y, index=df_raw.index, name="predicted_salary")

    def explain_shap(
        self,
        df_raw: pd.DataFrame,
        nsamples: int = 100,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute SHAP values for the given inputs.
        """
        X = self._transform_input(df_raw)
        explainer = self._get_shap_explainer()
        with mlflow_stage_run("inference", "shap_explain", tags={"interface": "api_or_ui"}):
            shap_values = explainer.shap_values(X, nsamples=nsamples)
        return np.array(shap_values), X

