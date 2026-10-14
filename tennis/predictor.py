"""ONNX inference. Predictions are symmetric by construction:
p(A beats B) = (f(A, B) + 1 - f(B, A)) / 2, so swapping players flips the
probability exactly and identical inputs give 0.5."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import onnxruntime as ort

from tennis_features import build_model_row

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "tennis_predictor.onnx"))
CARD_PATH = BASE_DIR / "model_card.json"
INPUT_NAME = "features"


@lru_cache(maxsize=1)
def _session() -> ort.InferenceSession:
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def model_card() -> Dict:
    try:
        return json.loads(CARD_PATH.read_text())
    except (OSError, ValueError):
        return {}


def model_status() -> Dict:
    try:
        _session()
        return {"model_loaded": True, "error": None}
    except Exception as exc:  # surfaced on /health
        return {"model_loaded": False, "error": str(exc)}


def win_probability(p1: Mapping, p2: Mapping) -> float:
    """Probability that p1 (season features) beats p2."""
    batch = np.asarray([build_model_row(p1, p2), build_model_row(p2, p1)], dtype=np.float32)
    proba = _session().run(["probabilities"], {INPUT_NAME: batch})[0][:, 1]
    return float((proba[0] + 1.0 - proba[1]) / 2.0)
