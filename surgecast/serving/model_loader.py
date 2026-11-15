"""Load the LightGBM champions registered in MLflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from surgecast.config import (
    CHAMPION_ALIAS,
    MODEL_NAME_15,
    MODEL_NAME_30,
    SCHEMA_VERSION,
)
from surgecast.features.compute import FEATURE_COLUMNS


def _load_one(name: str):
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    model = mlflow.lightgbm.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
    path = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="feature_columns.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = payload["columns"]
    if columns != FEATURE_COLUMNS:
        raise RuntimeError(f"feature column mismatch for {name}: model {columns} code {FEATURE_COLUMNS}")
    if int(payload.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise RuntimeError(f"schema version mismatch for {name}")
    return int(version.version), model


def use_tracking_uri(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5055")
    mlflow.set_tracking_uri(uri)
    return uri


def champion_versions(tracking_uri: str | None = None) -> tuple[int, int]:
    """Cheap alias lookup. Does not download model artifacts."""
    use_tracking_uri(tracking_uri)
    client = MlflowClient()
    v15 = int(client.get_model_version_by_alias(MODEL_NAME_15, CHAMPION_ALIAS).version)
    v30 = int(client.get_model_version_by_alias(MODEL_NAME_30, CHAMPION_ALIAS).version)
    return v15, v30


def load_champions(tracking_uri: str | None = None):
    """Return ``(version_15, model_15, version_30, model_30)``."""
    use_tracking_uri(tracking_uri)
    v15, m15 = _load_one(MODEL_NAME_15)
    v30, m30 = _load_one(MODEL_NAME_30)
    return v15, m15, v30, m30
