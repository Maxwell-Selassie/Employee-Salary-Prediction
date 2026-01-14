"""
Centralized MLflow utilities for this project.

Goals:
- Enforce a single SQLite tracking backend
- Standardize per-stage experiments (EDA, preprocessing, training, evaluation, inference)
- Provide lightweight helpers for runtime logging and context tagging
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import mlflow
from mlflow.tracking import MlflowClient

from .timer import format_duration

# Single local SQLite backend (no remote tracking)
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Logical stages → dedicated experiments
EXPERIMENTS: Dict[str, str] = {
    "eda": "employee_salary_eda",
    "preprocessing": "employee_salary_preprocessing",
    "training": "employee_salary_training",
    "evaluation": "employee_salary_evaluation",
    "inference": "employee_salary_inference",
}


def set_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """
    Configure MLflow tracking URI (SQLite only).

    Args:
        tracking_uri: Optional override; defaults to local SQLite DB.

    Returns:
        The URI that was set.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    return uri


def set_experiment_for_stage(stage: str) -> str:
    """
    Set the MLflow experiment for a logical pipeline stage.

    Args:
        stage: One of: 'eda', 'preprocessing', 'training', 'evaluation', 'inference'

    Returns:
        The experiment name that was set.
    """
    if stage not in EXPERIMENTS:
        raise ValueError(f"Unknown MLflow stage '{stage}'. Expected one of {sorted(EXPERIMENTS)}")

    experiment_name = EXPERIMENTS[stage]
    mlflow.set_experiment(experiment_name)
    return experiment_name


@contextmanager
def mlflow_stage_run(
    stage: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """
    Convenience context manager that:
    - Sets the tracking URI
    - Sets the appropriate experiment for the given stage
    - Starts an MLflow run
    - Logs a standard `runtime_seconds` metric and human-readable `runtime_human`

    Usage:
        with mlflow_stage_run("eda", "initial_eda") as run_id:
            mlflow.log_param("some_param", 123)
            ...

    Args:
        stage: Logical pipeline stage.
        run_name: Descriptive run name.
        tags: Optional tags to set on the run.

    Yields:
        The active MLflow run_id.
    """
    set_tracking_uri()
    set_experiment_for_stage(stage)

    start_time = time.time()
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Set any provided tags up-front (e.g., stage, purpose, version)
        base_tags = {"stage": stage}
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)

        try:
            yield run_id
        finally:
            elapsed = time.time() - start_time
            mlflow.log_metric("runtime_seconds", elapsed)
            mlflow.set_tag("runtime_human", format_duration(elapsed))


def get_mlflow_client() -> MlflowClient:
    """
    Get an MlflowClient configured with the project tracking URI.
    """
    set_tracking_uri()
    return MlflowClient()


def load_production_model(model_name: str, alias: str = "production"):
    """
    Load a model from MLflow Model Registry using an alias (no stages).

    Args:
        model_name: Registered model name.
        alias: Model alias to resolve (e.g. 'production').

    Returns:
        Loaded model object.
    """
    set_tracking_uri()
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.sklearn.load_model(model_uri)

