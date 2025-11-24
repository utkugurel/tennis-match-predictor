import os
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

from tennis_features import build_match_example, get_stats_from_db

MODEL_FILENAME = "tennis_predictor.onnx"
# Use absolute path relative to this file to ensure it works wherever app is imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, MODEL_FILENAME))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

# Serve static files from the 'static_content' folder
app = Flask(__name__, static_folder="static_content", static_url_path="")

@app.route("/")
def index():
    return app.send_static_file("index.html")

try:
    # Load ONNX model
    MODEL = ort.InferenceSession(MODEL_PATH)
    MODEL_LOAD_ERROR = None
    print(f"Loaded model from {MODEL_PATH}")
except Exception as exc:  # pragma: no cover - executed at runtime
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)
    print(f"Failed to load model: {MODEL_LOAD_ERROR}")

# Cache for player names
PLAYERS_CACHE = []

def get_all_players():
    global PLAYERS_CACHE
    if PLAYERS_CACHE:
        return PLAYERS_CACHE
    
    try:
        # We need a DB connection. get_stats_from_db creates one, but we need a custom query.
        # Let's reuse the DB path logic if possible, or just hardcode/env var it.
        # The original code uses "atp_stats.db" by default.
        import sqlite3
        conn = sqlite3.connect("atp_stats.db")
        cursor = conn.execute("SELECT DISTINCT player_name FROM player_yearly_stats ORDER BY player_name")
        players = [row[0] for row in cursor.fetchall()]
        conn.close()
        PLAYERS_CACHE = players
        return players
    except Exception as e:
        print(f"Error fetching players: {e}")
        return []


def normalize_surface(surface: str) -> str:
    cleaned = (surface or "").strip().lower()
    mapping = {
        "hard": "Hard",
        "clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
    }
    return mapping.get(cleaned, surface.strip())


def parse_payload(payload: Dict) -> Tuple[Dict, Dict, str]:
    required = [
        "player1_name",
        "player1_year",
        "player2_name",
        "player2_year",
        "surface",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    p1 = {
        "name": str(payload["player1_name"]).strip(),
        "year": int(payload["player1_year"]),
    }
    p2 = {
        "name": str(payload["player2_name"]).strip(),
        "year": int(payload["player2_year"]),
    }
    surface = normalize_surface(str(payload["surface"]))

    if not p1["name"] or not p2["name"]:
        raise ValueError("Player names cannot be empty.")
    
    # Prevent same player in same year (would not give 50-50 probability)
    if p1["name"].lower() == p2["name"].lower() and p1["year"] == p2["year"]:
        raise ValueError(
            f"Cannot predict a match between {p1['name']} and themselves in the same year. "
            "Please select different players or different years."
        )

    return p1, p2, surface


def summarize_player(stats: Dict, win_probability: float) -> Dict:
    return {
        "name": stats["player_name"],
        "year": int(stats["year"]),
        "surface": stats["surface"],
        "win_probability": round(win_probability, 4),
        "season_summary": {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_pct": round(stats["win_pct"], 4),
            "matches_played": stats["matches_played"],
            "rank": stats["rank"],
        },
    }


@app.after_request
def add_cors_headers(response):  # pragma: no cover - simple header mutation
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    status = {
        "model_path": MODEL_PATH,
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "error": MODEL_LOAD_ERROR,
    }
    return jsonify(status)


@app.route("/api/players", methods=["GET"])
def players():
    return jsonify(get_all_players())

@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if MODEL is None:
        return (
            jsonify(
                {
                    "error": "Model not available.",
                    "details": MODEL_LOAD_ERROR or "Train the model first.",
                }
            ),
            503,
        )

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400

    try:
        player1, player2, surface = parse_payload(payload)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    try:
        p1_stats = get_stats_from_db(player1["name"], player1["year"], surface)
        p2_stats = get_stats_from_db(player2["name"], player2["year"], surface)
    except ValueError as err:
        return jsonify({"error": str(err)}), 404

    match_row = build_match_example(p1_stats, p2_stats)
    
    # Prepare inputs for ONNX
    # The ONNX model expects inputs as a dictionary of {name: numpy_array}
    # Each input should be a 2D array (batch_size, 1)
    inputs = {}
    for key, value in match_row.items():
        # Convert to numpy array with shape (1, 1)
        # Numeric inputs are float32, categorical are strings
        if isinstance(value, str):
            inputs[key] = np.array([[value]], dtype=object)
        else:
            inputs[key] = np.array([[value]], dtype=np.float32)

    # Run inference
    # output_names usually ["label", "probabilities"] for sklearn classifiers
    # probabilities is a list of dictionaries in onnxruntime for sklearn?
    # Actually skl2onnx usually produces "output_label" and "output_probability"
    # Let's check the outputs or assume standard skl2onnx behavior:
    # output_probability is a sequence of maps (one per example)
    
    outputs = MODEL.run(["output_probability"], inputs)
    probs = outputs[0][0] # First example's probability map
    
    # probs is a dict like {0: 0.2, 1: 0.8}
    p1_win = float(probs.get(1, 0.0))
    p2_win = float(probs.get(0, 0.0))

    response = {
        "surface": surface,
        "player1": summarize_player(p1_stats, p1_win),
        "player2": summarize_player(p2_stats, p2_win),
        "model_version": MODEL_VERSION,
    }
    return jsonify(response)


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
