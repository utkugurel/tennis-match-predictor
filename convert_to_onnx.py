import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from tennis_features import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_DIFF_FEATURES,
    MODEL_NUMERIC_FEATURES,
)

def convert():
    # Load the trained pipeline
    model_path = "tennis_predictor.pkl"
    onnx_path = "tennis_predictor.onnx"
    
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)

    # Define input types
    # The pipeline expects a DataFrame, but for ONNX we usually define inputs by column
    # However, our ColumnTransformer uses column names.
    # skl2onnx can handle this if we provide a list of (name, type) tuples.
    
    initial_types = []
    
    # Numeric features (p1_*, p2_*)
    for col in MODEL_NUMERIC_FEATURES:
        initial_types.append((f"p1_{col}", FloatTensorType([None, 1])))
        initial_types.append((f"p2_{col}", FloatTensorType([None, 1])))
        
    # Diff features
    for col in MODEL_DIFF_FEATURES:
        initial_types.append((col, FloatTensorType([None, 1])))
        
    # Categorical features (p1_*, p2_*)
    for col in MODEL_CATEGORICAL_FEATURES:
        initial_types.append((f"p1_{col}", StringTensorType([None, 1])))
        initial_types.append((f"p2_{col}", StringTensorType([None, 1])))

    print("Converting to ONNX...")
    # We need to ensure the input names match what the ColumnTransformer expects.
    # The ColumnTransformer in train_model.py uses column names strings.
    
    onx = convert_sklearn(pipeline, initial_types=initial_types)
    
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"Model saved to {onnx_path}")

if __name__ == "__main__":
    convert()
