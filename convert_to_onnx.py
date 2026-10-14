"""Export the trained pipeline to ONNX and check it matches scikit-learn."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from tennis_features import MODEL_FEATURES
from train_model import build_dataset

BASE_DIR = Path(__file__).resolve().parent
INPUT_NAME = "features"
TOLERANCE = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=BASE_DIR / "tennis_predictor.pkl")
    parser.add_argument("--onnx-path", type=Path, default=BASE_DIR / "tennis_predictor.onnx")
    parser.add_argument("--db-path", type=Path, default=BASE_DIR / "atp_stats.db")
    parser.add_argument("--check-rows", type=int, default=1000)
    args = parser.parse_args()

    pipeline = joblib.load(args.model_path)
    onx = convert_sklearn(
        pipeline,
        initial_types=[(INPUT_NAME, FloatTensorType([None, len(MODEL_FEATURES)]))],
        options={id(pipeline): {"zipmap": False}},
        target_opset=17,
    )
    args.onnx_path.write_bytes(onx.SerializeToString())
    print(f"Saved {args.onnx_path} ({args.onnx_path.stat().st_size / 1e6:.1f} MB)")

    X, *_ = build_dataset(args.db_path)
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), size=min(args.check_rows, len(X)), replace=False)]

    session = ort.InferenceSession(str(args.onnx_path))
    onnx_proba = session.run(["probabilities"], {INPUT_NAME: sample})[0][:, 1]
    sklearn_proba = pipeline.predict_proba(sample)[:, 1]
    max_diff = float(np.max(np.abs(onnx_proba - sklearn_proba)))
    print(f"Max |ONNX - sklearn| over {len(sample)} rows: {max_diff:.2e}")
    if max_diff > TOLERANCE:
        raise SystemExit(f"ONNX export differs from sklearn by more than {TOLERANCE}.")


if __name__ == "__main__":
    main()
