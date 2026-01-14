from __future__ import annotations

"""
FastAPI application exposing the Employee Salary Prediction model.

Endpoints:
- /health          : Service health check
- /model/info      : Basic model + registry metadata
- /predict         : Single-row prediction (JSON)
- /predict/batch   : Batch prediction from CSV upload
- /explain         : SHAP explanation for a single row
"""

from io import StringIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference.service import InferenceService
from utils.mlflow_utils import get_mlflow_client, set_tracking_uri


app = FastAPI(title="Employee Salary Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    features: Dict[str, Any]


inference_service: InferenceService | None = None


@app.on_event("startup")
def on_startup() -> None:
    global inference_service
    set_tracking_uri()
    inference_service = InferenceService()


@app.get("/health")
def health() -> JSONResponse:
    if inference_service is None:
        return JSONResponse(status_code=503, content={"status": "initializing"})
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    client = get_mlflow_client()
    model_name = inference_service.model_name
    alias = inference_service.alias

    mv = client.get_model_version_by_alias(model_name, alias)

    return {
        "model_name": model_name,
        "alias": alias,
        "version": mv.version,
        "run_id": mv.run_id,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pred = inference_service.predict_single(request.features)
    return {"prediction": pred}


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> Dict[str, Any]:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    content = (await file.read()).decode("utf-8")
    df = pd.read_csv(StringIO(content))
    preds = inference_service.predict_batch(df)
    return {"predictions": preds.tolist()}


@app.post("/explain")
def explain(request: PredictRequest) -> Dict[str, Any]:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([request.features])
    shap_values, X = inference_service.explain_shap(df)
    # Use first row's contributions
    contribs = dict(zip(X.columns, shap_values[0].tolist()))
    return {"contributions": contribs}

