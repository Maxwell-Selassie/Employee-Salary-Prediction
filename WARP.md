# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository implements an end-to-end Employee Salary Prediction pipeline with three main stages:

1. **Preprocessing** (`src/preprocessing`)
   - Validates configuration and raw data.
   - Cleans data (missing values, duplicates, optional outliers).
   - Splits into train/dev/test sets with consistent feature spaces.
   - Encodes categorical features and applies feature transformations.
   - Saves processed datasets and a serialized preprocessing pipeline.

2. **Exploratory Data Analysis (EDA)** (`src/eda`)
   - Validates raw data schema against expectations.
   - Produces numeric and categorical summaries and quality checks.
   - Generates diagnostic plots (distributions, boxplots, correlations).
   - Writes an aggregated EDA summary report and artifacts under `artifacts/`.

3. **Model Training** (`src/model_training`)
   - Loads preprocessed train/dev/test splits.
   - Trains baseline and tree-based models.
   - Optionally performs hyperparameter tuning.
   - Evaluates models with regression metrics and generalization checks.
   - Logs runs and artifacts to MLflow and saves the best model and metadata.

The project is **config-driven** via YAML files under `config/`, and relies heavily on reusable utilities in `src/utils` for I/O, logging, timing, and configuration loading.


## Environment & dependencies

- **Python version**: `>=3.12` (see `pyproject.toml`).
- **Core runtime dependencies** (from `[project.dependencies]` in `pyproject.toml`):
  - `joblib`, `lightgbm`, `matplotlib`, `mlflow`, `numpy`, `pandas`, `pyyaml`,
    `scikit-learn`, `scipy`, `seaborn`, `shap`, `xgboost`.
- A typical setup flow from the project root:
  - Create/activate a virtual environment.
  - Install dependencies directly from the list above, for example:
    - `python -m pip install joblib lightgbm matplotlib mlflow numpy pandas pyyaml scikit-learn scipy seaborn shap xgboost`

There is currently no dedicated test or linting configuration in the repo.


## Core workflows & commands

All commands below assume you run them from the repository root.

### 1. Preprocessing pipeline (raw → processed data)

Module: `src/preprocessing/preprocessing_pipeline.py`

Responsibilities:
- Loads raw CSV data (default path from `config/preprocessing_config.yaml` → `data.file_path`, currently `data/raw/Employee_Complete_Dataset.csv`).
- Validates schema and value ranges (`expected_columns`, `value_ranges`).
- Handles duplicates and missing values.
- Splits into train/dev/test sets (`data_split` section).
- Applies outlier handling (if enabled), encoding, and feature transforms.
- Saves processed datasets and pipeline artifacts.

Key entrypoint:
- Script-style main at bottom of `preprocessing_pipeline.py` (`if __name__ == "__main__"`), which constructs `PreprocessingPipeline` and calls `fit_transform()`.

Common commands:
- **Run full preprocessing pipeline** (uses `config/preprocessing_config.yaml`):
  - `python src/preprocessing/preprocessing_pipeline.py`

Outputs:
- Processed CSVs under `data/processed/` (see `output.processed_dir`).
- Serialized preprocessing pipeline under `artifacts/data/preprocessing_pipeline.joblib`.
- Preprocessing metadata under `data/processed/preprocessing_metadata.json`.
- Logs under `logs/` (via `LoggerMixin` / `setup_logger`).

### 2. EDA pipeline (raw data diagnostics)

Module: `src/eda/eda_pipeline.py`

Responsibilities:
- Orchestrates:
  - `DataOverview` (`src/eda/data_overview.py`): loads config, reads raw data, validates schema, generates numeric and categorical summaries, and persists feature expectations.
  - `DataQuality` (`src/eda/data_quality.py`): missing-values analysis, duplicate detection, outlier detection with severity thresholds, and CSV artifacts.
  - `Visualizations` (`src/eda/visualizations.py`): histograms, boxplots, categorical distributions, and correlation heatmaps.
- Uses MLflow to track dataset metadata, summaries, and artifacts.
- Writes a consolidated JSON summary report.

Configuration:
- Default config path in `EDAPipeline.__init__`: `config/eda_config.yaml` (note: the actual file in `config/` is `EDA_config.yaml`, which is fine on Windows but will be case-sensitive on Linux/macOS).
- EDA config file (`config/EDA_config.yaml`) defines:
  - Raw data path: `data.raw_path` (`data/raw/Employee_Complete_Dataset.csv`).
  - Expected columns and feature store path.
  - Logging settings.
  - Output directories for plots and artifacts.
  - Data quality thresholds and visualization options.

Common commands:
- **Run full EDA pipeline** (uses `config/EDA_config.yaml` by default in code):
  - `python src/eda/eda_pipeline.py`

Outputs:
- EDA numeric and categorical summaries in CSV form under `artifacts/`.
- Data quality CSVs for missing values, duplicates (optional), and outliers under `artifacts/data/`.
- Plots under `artifacts/plots/`.
- Aggregated summary JSON (`eda_summary_report.json`) under the EDA `output.artifacts_dir` (currently `artifacts/`).
- MLflow tracking to a SQLite DB at `sqlite:///employee_salary_preds.db` (set in `EDAPipeline.execute`).

MLflow UI (EDA):
- To inspect EDA runs tracked to `employee_salary_preds.db`:
  - `mlflow ui --backend-store-uri sqlite:///employee_salary_preds.db`

### 3. Model training pipeline (processed → model & artifacts)

Module: `src/model_training/model_training_pipeline.py`

Responsibilities:
- Loads preprocessed train/dev/test splits via `TrainingDataLoader` (`src/model_training/data_loader.py`).
- Trains a baseline model plus enabled tree-based models.
- Optionally tunes top-performing models via `HyperparameterTuner` (Optuna-based, see `hyperparameter_tuning` section in config).
- Evaluates models with regression metrics via `ModelEvaluator`.
- Analyzes feature importance (native, permutation, optional SHAP) via `FeatureImportanceAnalyzer`.
- Validates model quality via `ModelValidator` before persisting or registering.
- Logs metrics, parameters, and artifacts to MLflow, and persists the best model and training metadata.

Configuration:
- Default config path: `config/model_training_config.yaml`.
- Key sections in `model_training_config.yaml`:
  - `data`: paths to `data/processed/train_set.csv`, `dev_set.csv`, `test_set.csv`, and the target column name.
  - `mlflow`: SQLite tracking URI (`sqlite:///mlruns/employee_salary_prediction.db`), experiment name, and registry options.
  - `models`: which models to train, with default hyperparameters.
  - `hyperparameter_tuning`: Optuna settings and search spaces.
  - `metrics`: primary metric (`mean_squared_error`) and additional regression metrics.
  - `model_persistence`: `models/best_model.joblib` and artifacts under `artifacts/`.
  - `plotting`: enables training-related plots under `artifacts/plots/`.

Entrypoint:
- `main()` at bottom of `model_training_pipeline.py` instantiates `ModelTrainingPipeline` with the config and calls `execute()`.

Prerequisites:
- Run the **preprocessing pipeline** first to generate the processed CSVs expected by `data.train_data`, `data.val_data`, and `data.test_data`.

Common commands:
- **Run full model training pipeline**:
  - `python src/model_training/model_training_pipeline.py`

Expected outputs:
- Trained best model saved to `models/best_model.joblib`.
- Additional artifacts (if enabled in `model_persistence.save_artifacts`) under `artifacts/` (feature names, metrics summaries, etc.).
- MLflow runs stored in `sqlite:///mlruns/employee_salary_prediction.db`.

MLflow UI (training):
- To inspect training experiments defined in `config/model_training_config.yaml`:
  - `mlflow ui --backend-store-uri sqlite:///mlruns/employee_salary_prediction.db`

### 4. Jupyter notebooks

Location: `notebooks/`.
- `01_eda.ipynb`: interactive EDA and data understanding.
- `03_model_training.ipynb`: interactive model training experiments.

Typical usage:
- Start Jupyter from the project root:
  - `jupyter lab` or `jupyter notebook`
- Then open the notebook(s) under `notebooks/`.


## High-level architecture & design notes

### Packages and layering

- **`src/utils`**
  - Shared infrastructure utilities for the entire project.
  - `io_utils.py`: CSV/YAML/JSON/joblib read/write helpers with error handling and optional dtype optimization.
  - `logging_mixin.py`:
    - `LoggerMixin`: mixin used across preprocessing, EDA, and training to create per-class loggers based on config.
    - `setup_logger`: central logging factory (rotating file handlers, console output, consistent formats) writing to `logs/`.
  - `timer.py`: time measurement utilities and `Timer` context manager used around major pipeline phases.
  - `__init__.py` re-exports these helpers so modules can import from `utils` directly.

- **`src/preprocessing`**
  - Orchestrated by `PreprocessingPipeline` in `preprocessing_pipeline.py`.
  - Key components (see their modules for details):
    - `ConfigValidator` / `DataValidator`: configuration and data integrity checks.
    - `DataSplitter`: orchestrates train/dev/test splitting, with logging and validation of split sizes and distributions.
    - `MissingHandler`, `DuplicateHandler`, `OutlierHandler`: encapsulate cleaning logic.
    - `FeatureEncoder`, `FeatureTransformer`: encoding and transformations (e.g., log transforms, one-hot encoding) applied in a leakage-safe order (fit on train, transform dev/test).
  - Outputs both tabular artifacts (processed CSVs) and a serialized pipeline + metadata for reproducibility.

- **`src/eda`**
  - Encapsulates EDA into composable stages, each with its own logger derived from `LoggerMixin`:
    - `DataOverview`:
      - Loads EDA config and raw data.
      - Validates schema against `data.expected_columns` and writes expected columns/feature store JSON.
      - Generates numeric and categorical summaries and logs them as MLflow artifacts.
    - `DataQuality`:
      - Uses `data_quality` settings from EDA config to compute missing-value severities, duplicates, and IQR-based outliers.
      - Persists diagnostic CSVs under `artifacts/data/`.
    - `Visualizations`:
      - Driven by `visualization` and `output` sections in EDA config.
      - Uses seaborn/matplotlib to produce multi-panel plots and correlation heatmaps; saves them to `artifacts/plots/` and logs to MLflow.
  - `EDAPipeline` coordinates these components, aggregates high-level results, and writes a single JSON summary for downstream reference.

- **`src/model_training`**
  - Designed as a modular training system centered on `ModelTrainingPipeline`:
    - `TrainingDataLoader`: loads processed splits, validates target column presence, and ensures feature alignment across splits.
    - `ModelTrainer`: given a model name and hyperparameters, trains scikit-learn / XGBoost / LightGBM models.
    - `HyperparameterTuner`: wraps Optuna-based tuning using search spaces declared in config.
    - `ModelEvaluator`: computes regression metrics requested in the `metrics` section and compares train vs dev for generalization.
    - `FeatureImportanceAnalyzer`: computes model-specific and generic feature importance and can generate plots.
    - `ModelValidator`: enforces thresholds from `model_validation` (MSE, R², train–test gap) before allowing persistence/registration.
  - `ModelTrainingPipeline.execute()` orchestrates:
    - Data loading → baseline model training → (optional) tuning → best-model selection → validation → persistence/registration.
  - MLflow is integrated at the pipeline and model level for parameters, metrics, plots, and model artifacts.

### Configuration-driven behavior

- **Preprocessing** (`config/preprocessing_config.yaml`):
  - Controls raw file path, expected schema, value ranges, split sizes, and all cleaning/encoding/transform behaviors.
  - Changing column names or adding new features requires keeping `expected_columns`, `value_ranges`, `encoding`, and `transformations` sections in sync with the actual data.

- **EDA** (`config/EDA_config.yaml`):
  - Governs which columns are expected, how data quality is assessed, and where EDA artifacts are written.
  - Any updates to the raw dataset schema should be reflected here first so that `DataOverview` and `DataQuality` validations remain consistent.

- **Model training** (`config/model_training_config.yaml`):
  - Drives which models are trained, how tuning is performed, which metrics and thresholds apply, and how/where models and artifacts are persisted.
  - Adjusting the primary metric or acceptable error thresholds is done here, not in the model code.

### Logging, artifacts, and data flow

- **Logging**:
  - All major classes use `LoggerMixin.setup_class_logger` with a shared `logging` section from config files.
  - Logs are written under `logs/` with rotating file handlers plus console output.

- **Data & artifact directories** (as configured):
  - Raw data: typically `data/raw/Employee_Complete_Dataset.csv`.
  - Processed data: `data/processed/`.
  - Models: `models/`.
  - General artifacts (EDA + training): `artifacts/` and subdirectories (`feature_store/`, `plots/`, `data/`).
  - Serialized preprocessing pipeline: `artifacts/data/preprocessing_pipeline.joblib`.
  - EDA summary: `artifacts/eda_summary_report.json`.

- **MLflow tracking**:
  - EDA: uses `sqlite:///employee_salary_preds.db` directly in `EDAPipeline.execute()`.
  - Training: uses the `mlflow.tracking_uri` and `mlflow.experiment_name` from `config/model_training_config.yaml` (currently `sqlite:///mlruns/employee_salary_prediction.db`).

### Entry points and execution order

For a clean end-to-end retrain on updated raw data, the typical order is:

1. **Preprocessing**: `python src/preprocessing/preprocessing_pipeline.py`
2. **(Optional) EDA**: `python src/eda/eda_pipeline.py`
3. **Model training**: `python src/model_training/model_training_pipeline.py`

`main.py` at the project root currently only prints a placeholder message and does not orchestrate these stages; modify it if you want a single CLI entrypoint that chains the full pipeline.
