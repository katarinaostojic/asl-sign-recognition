"""Tiny experiment tracker: JSON files per run, plus MLflow if it is installed and enabled."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from . import config


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_run(name: str, params: dict, metrics: dict) -> None:
    d = config.RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (d / "params.json").write_text(json.dumps({"time": stamp, **params}, indent=2, default=_jsonable))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_jsonable))
    if os.environ.get("ASL_MLFLOW") == "1":  # opt-in: `set ASL_MLFLOW=1`
        try:
            import mlflow

            mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
            mlflow.set_experiment("asl-sign-recognition")
            with mlflow.start_run(run_name=name):
                mlflow.log_params({k: v for k, v in params.items() if np.isscalar(v) or isinstance(v, str)})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float, np.floating, np.integer))})
        except ImportError:
            print("ASL_MLFLOW=1 but mlflow is not installed (pip install mlflow); skipping.")


def save_predictions(name: str, test_rows, y_true, probs, detected) -> None:
    """Persist test-set predictions in a model-independent format for evaluation.

    probs has one row per test image; rows with detected == False are all zeros
    (the evaluation script decides how to score them).
    """
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PRED_DIR / f"{name}.npz",
        test_rows=np.asarray(test_rows),
        y_true=np.asarray(y_true),
        probs=np.asarray(probs, dtype=np.float32),
        detected=np.asarray(detected, dtype=bool),
    )
