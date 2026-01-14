## Employee Salary Prediction – Production-Grade ML & MLOps

This repository implements a **production-ready salary prediction system** for employees based on structured tabular data.

It goes beyond a simple notebook model by providing:

- Modular **EDA → preprocessing → training → evaluation → inference** pipelines
- End-to-end **MLflow** integration with a local **SQLite** backend
- Automatic **best-model selection**, **model registry registration**, and **alias-based serving**
- A **FastAPI** service for programmatic inference
- A **Streamlit** application for business users
- **SHAP**-based model explanations
- A growing **pytest** suite for regression and safety checks

The stack is intentionally **local-first** and cloud-agnostic; deployment to Docker/Kubernetes/cloud can be added later without changing core design.

---

## 1. Architecture Overview

### 1.1 High-Level Flow

1. **EDA (`src/eda`)**
   - Validates schema against expectations.
   - Analyzes distributions, missingness, outliers, and relationships.
   - Produces an `eda_summary_report.json` and visualization artifacts.
   - Logs metrics, artifacts, and decisions to the MLflow **`eda` experiment**.

2. **Preprocessing (`src/preprocessing`)**
   - Validates configuration and data integrity.
   - Handles duplicates, missing values, outliers, and encodes categorical features.
   - Applies deterministic feature transformations (e.g. log transforms).
   - Splits data into **train/dev/test** without leakage.
   - Saves processed datasets and a **preprocessing pipeline object** with handlers and metadata.
   - Logs row counts, split sizes, and decisions to the MLflow **`preprocessing` experiment**.

3. **Model Training & Selection (`src/model_training`)**
   - Loads processed data splits.
   - Trains multiple models: **Ridge Regression**, **Random Forest**, **XGBoost**, **LightGBM**.
   - Uses **cross-validated MSE** for model comparison.
   - Enforces a **per-model runtime constraint** (< 10 seconds) with explicit tagging when violated.
   - Logs parameters, metrics, dataset lineage, and model-as-code artifacts to the MLflow **`training` experiment**.
   - Automatically **registers the best model** and assigns MLflow alias **`production`**.

4. **Evaluation & Validation (`src/model_training/model_evaluator.py`, `model_validator.py`)**
   - Evaluates models across train/val/test splits with multiple regression metrics.
   - Checks for overfitting via train–validation gap.
   - Validates metrics against configurable thresholds.
   - (Optionally) compares new models against the current production model.
   - Logs evaluation and validation results to the MLflow **`evaluation` experiment**.

5. **Inference (`src/inference/service.py`)**
   - Loads the **preprocessing pipeline** (as code + config) from disk.
   - Loads the **best production model** from the MLflow Model Registry via alias:  
     `models:/EmployeeSalaryModel@production`.
   - Provides:
     - `predict_single(features: Dict[str, Any]) -> float`
     - `predict_batch(df: pd.DataFrame) -> pd.Series`
     - `explain_shap(df: pd.DataFrame) -> (np.ndarray, pd.DataFrame)`
   - Logs batch sizes, latency, and explanation metadata to the MLflow **`inference` experiment**.

6. **Serving Interfaces**
   - **FastAPI (`src/api/app.py`)**:
     - `/health`
     - `/predict` (single row JSON)
     - `/predict/batch` (CSV upload)
     - `/model/info` (registry metadata)
     - `/explain` (SHAP contributions)
   - **Streamlit (`src/app/streamlit_app.py`)**:
     - Manual form input for single predictions.
     - CSV batch upload.
     - SHAP-based feature contribution plots for business interpretation.

7. **Testing (`tests/`)**
   - Validates:
     - Preprocessing splits and schema.
     - API endpoints.
     - Inference and SHAP output shapes and sanity.
   - Designed to be extended toward full coverage of pipelines and MLflow logging.

---

## 2. MLflow Design & Conventions

### 2.1 Tracking Backend

- **Tracking URI**: `sqlite:///mlflow.db`
- Configured centrally in `src/utils/mlflow_utils.py` and used uniformly in all stages.
- Ensures:
  - Local, file-based tracking.
  - No remote servers or authentication.
  - Easy reproducibility on a single machine.

```python
from utils.mlflow_utils import set_tracking_uri

set_tracking_uri()  # sets sqlite:///mlflow.db
```

### 2.2 Per-Stage Experiments

Each logical stage has its own MLflow experiment:

- `employee_salary_eda`
- `employee_salary_preprocessing`
- `employee_salary_training`
- `employee_salary_evaluation`
- `employee_salary_inference`

The mapping and utilities are defined in `src/utils/mlflow_utils.py` and encapsulated via `mlflow_stage_run`:

```python
from utils.mlflow_utils import mlflow_stage_run
import mlflow

with mlflow_stage_run("training", "training_session_20260114") as run_id:
    mlflow.log_param("n_models", 4)
    ...
    mlflow.log_metric("runtime_seconds", 8.73)
```

Within each run:

- **Parameters**: configuration, hyperparameters, feature flags, dataset sizes.
- **Metrics**: performance metrics and timing (including `runtime_seconds`).
- **Artifacts**: reports, plots, joblib pipelines, and model artifacts.
- **Tags**: decisions and context notes (e.g. why a model was selected or rejected).

### 2.3 Model Registry & Aliases

- The **best model** is selected by:
  - Lowest **cross-validated MSE** (`cv_mse`).
  - Satisfying the **runtime constraint** (< 10 seconds) where possible.
- The winner is registered as `EmployeeSalaryModel` with:
  - A registered model version in the MLflow Model Registry.
  - Alias **`production`** assigned (no deprecated stages are used).
- Downstream services load via:

```python
from utils.mlflow_utils import load_production_model

model = load_production_model("EmployeeSalaryModel", alias="production")
```

This decouples training from serving:

- New training runs can register new versions and update the `production` alias.
- FastAPI / Streamlit only depend on the alias, not on a specific run ID.

---

## 3. Project Layout

Key directories and their responsibilities:

- `config/`
  - `EDA_config.yaml` – EDA and data-quality configuration.
  - `preprocessing_config.yaml` – schema, ranges, encoding, and splitting config.
  - `model_training_config.yaml` – model definitions, MLflow config, runtime constraints, and validation thresholds.

- `src/`
  - `eda/`
    - `data_overview.py` – high-level dataset introspection.
    - `data_quality.py` – missing values, duplicates, outlier detection.
    - `visualizations.py` – summary plots.
    - `eda_pipeline.py` – orchestrated EDA run with MLflow integration.
  - `preprocessing/`
    - `config_validator.py` – validates preprocessing config structure.
    - `data_validator.py` – validates schemas, ranges, integrity.
    - `handle_missing.py`, `handle_duplicates.py`, `handle_outliers.py` – core data cleaning steps.
    - `encoding.py` – categorical encoding.
    - `feature_transformations.py` – log transforms and other numeric operations.
    - `data_splitter.py` – train/dev/test split with no leakage.
    - `preprocessing_pipeline.py` – orchestrates all of the above and saves the pipeline object.
  - `model_training/`
    - `model_training_pipeline.py` – main training entrypoint; handles:
      - MLflow experiment setup.
      - Data loading.
      - Multi-model training and CV.
      - Best-model selection and registry registration.
    - `model_evaluator.py` – evaluation utilities for regression metrics.
    - `model_validator.py` – formal validation and production comparison logic.
    - `feature_importance_analyzer.py` – optional feature importance tooling.
  - `inference/`
    - `service.py` – `InferenceService` providing single/batch inference and SHAP explanations.
  - `api/`
    - `app.py` – FastAPI service that loads `InferenceService` once on startup and exposes HTTP endpoints.
  - `app/`
    - `streamlit_app.py` – business-facing visualization and prediction demo using the production model.
  - `utils/`
    - `io_utils.py` – robust CSV/YAML/JSON/joblib I/O.
    - `timer.py` – timing utilities and human-readable durations.
    - `logging_mixin.py` – standardized logging setup.
    - `mlflow_utils.py` – central MLflow configuration and helper utilities.

- `artifacts/`
  - `feature_store/expected_features.json` – expected columns used across pipelines.
  - `eda_summary_report.json` – EDA summary for quick reference.
  - `data/preprocessing_pipeline.joblib` – saved preprocessing pipeline.
  - Additional plots and data artifacts as pipelines run.

- `tests/`
  - `test_preprocessing_pipeline.py` – checks preprocessing splits and schema.
  - `test_api.py` – verifies FastAPI `/health` and `/predict` behavior.
  - `test_inference_and_shap.py` – checks inference output and SHAP contribution lengths.

---

## 4. How to Run Pipelines

Make sure dependencies are installed:

```bash
pip install -e .
```

### 4.1 EDA

```bash
python -m src.eda.eda_pipeline
```

This will:

- Run EDA against the configured dataset.
- Generate plots and a JSON summary under `artifacts/`.
- Log metrics, parameters, and artifacts to the MLflow **`eda` experiment**.

### 4.2 Preprocessing

```bash
python -m src.preprocessing.preprocessing_pipeline
```

This will:

- Load raw data and validate schema & value ranges.
- Handle duplicates, missing values, outliers.
- Split into train/dev/test.
- Apply encoding and feature transformations.
- Save:
  - `data/processed/train_set.csv`
  - `data/processed/dev_set.csv`
  - `data/processed/test_set.csv`
  - `artifacts/data/preprocessing_pipeline.joblib`
  - `data/processed/preprocessing_metadata.json`
- Log row counts and decisions to the MLflow **`preprocessing` experiment**.

### 4.3 Training & Model Registration

```bash
python -m src.model_training.model_training_pipeline
```

This will:

- Load processed train/dev/test splits.
- Train configured models (Ridge, RF, XGBoost, LightGBM).
- Perform cross-validation and log **CV MSE** and other metrics.
- Apply a **per-model runtime constraint** (< 10 seconds; violations are tagged).
- Select the best model by CV MSE (subject to runtime) and register it as:
  - Registered model: `EmployeeSalaryModel`
  - Alias: `production`
- Log all training runs to the MLflow **`training` experiment**.

You can inspect runs via:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 5. Serving: FastAPI & Streamlit

### 5.1 FastAPI Service

Start the API:

```bash
uvicorn src.api.app:app --reload
```

Key endpoints:

- `GET /health`  
  Health check; returns `{ "status": "ok" }` when the inference service has loaded.

- `GET /model/info`  
  Returns current `model_name`, `alias`, `version`, and `run_id` from the Model Registry.

- `POST /predict`  
  Single-row prediction. Example request body:

  ```json
  {
    "features": {
      "Employee_age": 30,
      "years_experience": 5,
      "Number_of_Children": 0,
      "Department": "Engineering",
      "Role": "Software Engineer",
      "performance_rating": 3
    }
  }
  ```

- `POST /predict/batch`  
  Accepts a CSV file upload (same feature columns), returns a list of predicted salaries.

- `POST /explain`  
  Returns SHAP feature contributions for a single row:

  ```json
  {
    "contributions": {
      "Employee_age": 123.4,
      "years_experience": 456.7,
      ...
    }
  }
  ```

### 5.2 Streamlit Business App

Start the UI:

```bash
streamlit run src/app/streamlit_app.py
```

Features:

- **Single Prediction Tab**
  - Manual controls for age, years of experience, department, role, etc.
  - Displays predicted salary using the **production** model.
  - Shows a SHAP bar plot for feature contributions.

- **Batch Prediction Tab**
  - Upload CSV with feature columns.
  - Displays a preview and computed salaries.
  - Allows downloading predictions as a CSV file.

Both the API and Streamlit app share the same **`InferenceService`**, which:

- Loads the preprocessing pipeline (model-as-code).
- Loads the model from MLflow using the `production` alias.

---

## 6. Testing Strategy

Run the full test suite with:

```bash
pytest
```

Current coverage includes:

- **Preprocessing**
  - `test_preprocessing_pipeline.py` – ensures non-empty train/dev/test splits and consistent schemas.

- **Inference & SHAP**
  - `test_inference_and_shap.py` – confirms:
    - Positive salary predictions for plausible inputs.
    - SHAP output length matches the number of input features.

- **FastAPI**
  - `test_api.py` – verifies:
    - `/health` returns 200 or 503 depending on readiness.
    - `/predict` endpoint can be called and returns a `prediction` when the model is available.

These tests are designed to be **fast** and **deterministic**, and serve as a foundation to:

- Add more tests around:
  - Data validation logic.
  - Feature engineering determinism across runs.
  - Model training reproducibility (e.g. stable metrics with fixed seeds).
  - MLflow logging correctness and registry alias assignment.

---

## 7. Design Decisions & Trade-Offs

### 7.1 Why MLflow This Way?

- **SQLite Backend**
  - Fits the project’s constraint of **local filesystem artifacts** and no remote tracking.
  - Keeps the setup simple and reproducible on a single machine.

- **Per-Stage Experiments**
  - EDA, preprocessing, training, evaluation, and inference each get their own experiment.
  - Encourages **observability** and **lineage tracking** for every step.
  - Makes it easy to answer questions like:
    - “What preprocessing version was used for this model?”
    - “What decisions were made during EDA for this dataset?”

- **Aliases (No Stages)**
  - Explicitly avoids deprecated MLflow stages (`Staging`, `Production`, etc.).
  - Uses only **aliases** (e.g. `production`) to represent the current serving model.
  - This pattern scales better when:
    - You have multiple environments (dev/stage/prod) represented as aliases.
    - You promote models between environments programmatically.

### 7.2 Why This Structure Scales

- **Separation of Concerns**
  - EDA, preprocessing, training, evaluation, and inference live in separate modules.
  - Changing the preprocessing logic does not require touching training or inference interfaces.
  - New models can be added to `model_training_config.yaml` without changing inference code.

- **Model-as-Code**
  - Training logs models with `code_paths=["src/"]`.
  - The preprocessing pipeline is a **Python object** containing transformers and handlers.
  - This makes it easier to:
    - Reproduce training logic exactly.
    - Inspect and evolve transformations over time.

- **API & UI Decoupled from Training**
  - FastAPI and Streamlit depend only on the **Model Registry alias**, not on run IDs.
  - New training runs update the alias; serving layers automatically use the new best model after promotion.

### 7.3 Runtime Constraints

- The **< 10 seconds per-model constraint** is a **satisficing metric**, not an absolute guarantee that every future model will be extremely fast.
- Enforcement:
  - Each model logs its `training_time_seconds`.
  - Models that exceed the threshold are tagged and, depending on configuration, can be excluded from selection.
- This balances:
  - **Exploration** (trying more complex models).
  - **Operational realism** (fast, iterative training is critical in many production settings).

---

## 8. Next Steps & Extensions

Suggested extensions (deliberately out of scope for now, but enabled by this design):

- **Deployment**
  - Dockerfile and containerization of the FastAPI app + model artifacts.
  - CI/CD pipeline to automatically retrain and promote models.

- **Monitoring**
  - Endpoint-level logging for latency and errors.
  - Data drift detection using live traffic plus EDA metrics.

- **Feature Store Integration**
  - Leveraging the existing `artifacts/feature_store/expected_features.json` as a starting point for a more formal feature store.

The current structure is intentionally designed so these features can be added **without rewriting the core pipelines** or changing how clients interact with the model.

