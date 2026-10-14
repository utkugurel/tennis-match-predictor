"""Train the match predictor without target leakage.

Each match is described by both players' *season* stats on that surface
with the whole tournament removed (leave-one-tournament-out). Removing only
the match would still leak the result: the winner goes on to play (and often
win) later rounds of the same event, inflating their season totals. Elo
ratings are taken from just before the match; the app serves the season's
mean pre-match Elo (tennis.etl.season_elo), the same kind of number. Validation keeps both orientations of a match in
the same fold, and a time-based holdout checks generalisation to later years.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis.etl import player_match_elo, to_long
from tennis_features import MODEL_FEATURES, TOTAL_COLUMNS, build_model_row, compute_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "2.0.0"
MIN_OTHER_MATCHES = 3
TEMPORAL_SPLIT_YEAR = 2019
MAX_CALIBRATION_ERROR = 0.02
# Logistic regression is used even though the random forest scores marginally
# better (~0.5 pt accuracy): trees can't extrapolate, so they compress
# predictions for two exceptional seasons towards 50%, which is exactly the
# comparison this app is for.
FINAL_MODEL = "logistic_regression"


def load_matches(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        matches = pd.read_sql("SELECT * FROM matches", conn)
        players = pd.read_sql("SELECT player_id, hand, ht FROM players", conn)
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    return matches, players, meta


def leave_one_out_features(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per (match, player) features from season totals minus that tournament."""
    long = to_long(matches)
    season_totals = long.groupby(["player_id", "year", "surface"])[TOTAL_COLUMNS].transform("sum")
    event_totals = long.groupby(["player_id", "year", "surface", "tourney_id"])[TOTAL_COLUMNS].transform("sum")
    loo = long[["match_id", "player_id", "year"]].copy()
    loo[TOTAL_COLUMNS] = season_totals.to_numpy() - event_totals.to_numpy()
    elo = player_match_elo(matches)[["match_id", "player_id", "elo", "surface_elo"]]
    loo = loo.merge(elo, on=["match_id", "player_id"], how="left", validate="one_to_one")
    loo = loo.merge(players, on="player_id", how="left")
    loo["other_matches"] = loo["wins"] + loo["losses"]
    loo["won"] = long["wins"].to_numpy()
    records = loo.to_dict("records")
    loo["features"] = [compute_features(record) for record in records]
    return loo


def build_dataset(db_path: Path):
    matches, players, meta = load_matches(db_path)
    loo = leave_one_out_features(matches, players)

    winners = loo[loo["won"] == 1].set_index("match_id")
    losers = loo[loo["won"] == 0].set_index("match_id")
    both = winners.join(losers, lsuffix="_w", rsuffix="_l", how="inner")
    keep = (both["other_matches_w"] >= MIN_OTHER_MATCHES) & (both["other_matches_l"] >= MIN_OTHER_MATCHES)
    both = both[keep]
    print(f"{len(both)} of {len(matches)} matches have >= {MIN_OTHER_MATCHES} matches per player "
          "outside the tournament.")

    rows, labels, groups, years = [], [], [], []
    for match_id, w, l, year in zip(both.index, both["features_w"], both["features_l"], both["year_w"]):
        rows.append(build_model_row(w, l))
        rows.append(build_model_row(l, w))
        labels += [1, 0]
        groups += [match_id, match_id]
        years += [year, year]
    X = np.asarray(rows, dtype=np.float32)
    return X, np.asarray(labels), np.asarray(groups), np.asarray(years), meta


def candidate_models() -> Dict[str, Pipeline]:
    def pipeline(model):
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )

    # HistGradientBoosting scored slightly better (log loss 0.583 vs 0.586) but
    # skl2onnx 1.20 cannot export it from scikit-learn 1.7, so it is left out.
    # Forest size is capped to keep the ONNX file small for Vercel.
    return {
        "logistic_regression": pipeline(LogisticRegression(C=1.0, max_iter=2000)),
        "random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42
            )
        ),
    }


def symmetric_proba(model, X: np.ndarray) -> np.ndarray:
    """Average f(A,B) and 1 - f(B,A), as the app does. Rows come in mirrored pairs."""
    proba = model.predict_proba(X)[:, 1]
    swapped = proba.reshape(-1, 2)[:, ::-1].reshape(-1)
    return (proba + 1.0 - swapped) / 2.0


def calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            error += mask.mean() * abs(proba[mask].mean() - y[mask].mean())
    return float(error)


def evaluate(model, X, y) -> Dict[str, float]:
    proba = symmetric_proba(model, X)
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, proba >= 0.5)), 4),
        "log_loss": round(float(log_loss(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "calibration_error": round(calibration_error(y, proba), 4),
    }


def grouped_split(groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    # Keep mirrored pairs adjacent (symmetric_proba relies on it).
    return np.sort(train), np.sort(test)


def calibrated(model, X, y, groups):
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))
    return CalibratedClassifierCV(clone(model), method="isotonic", cv=folds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=BASE_DIR / "atp_stats.db")
    parser.add_argument("--model-path", type=Path, default=BASE_DIR / "tennis_predictor.pkl")
    parser.add_argument("--card-path", type=Path, default=BASE_DIR / "model_card.json")
    args = parser.parse_args()

    X, y, groups, years, meta = build_dataset(args.db_path)
    print(f"Training rows: {len(X)} ({len(X) // 2} matches), {X.shape[1]} features.")

    train_idx, test_idx = grouped_split(groups)
    results = {}
    for name, model in candidate_models().items():
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        results[name] = evaluate(fitted, X[test_idx], y[test_idx])
        print(f"{name:24s} grouped holdout: {results[name]}")

    best_name = FINAL_MODEL
    best = candidate_models()[best_name]
    use_calibration = results[best_name]["calibration_error"] > MAX_CALIBRATION_ERROR
    if use_calibration:
        cal = calibrated(best, X[train_idx], y[train_idx], groups[train_idx])
        cal_metrics = evaluate(cal.fit(X[train_idx], y[train_idx]), X[test_idx], y[test_idx])
        print(f"{best_name} + isotonic calibration: {cal_metrics}")
        use_calibration = cal_metrics["log_loss"] < results[best_name]["log_loss"]
        if use_calibration:
            results[f"{best_name}+calibration"] = cal_metrics
    print(f"Selected: {best_name}{' + calibration' if use_calibration else ''}")

    def make_final(X_fit, y_fit, groups_fit):
        if use_calibration:
            return calibrated(best, X_fit, y_fit, groups_fit).fit(X_fit, y_fit)
        return clone(best).fit(X_fit, y_fit)

    past, future = years < TEMPORAL_SPLIT_YEAR, years >= TEMPORAL_SPLIT_YEAR
    temporal = evaluate(make_final(X[past], y[past], groups[past]), X[future], y[future])
    print(f"Temporal holdout (train < {TEMPORAL_SPLIT_YEAR}, test >= {TEMPORAL_SPLIT_YEAR}): {temporal}")

    final = make_final(X, y, groups)
    joblib.dump(final, args.model_path)
    print(f"Model saved to {args.model_path}")

    selected_key = f"{best_name}+calibration" if use_calibration else best_name
    card = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name,
        "calibrated": use_calibration,
        "data_snapshot": meta.get("snapshot_id"),
        "data_manifest_sha256": meta.get("manifest_sha256"),
        "training_matches": int(len(X) // 2),
        "min_other_matches": MIN_OTHER_MATCHES,
        "features": MODEL_FEATURES,
        "metrics": {
            "grouped_holdout": results[selected_key],
            f"temporal_holdout_{TEMPORAL_SPLIT_YEAR}_plus": temporal,
        },
        "candidates": results,
    }
    args.card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"Model card saved to {args.card_path}")


if __name__ == "__main__":
    main()
