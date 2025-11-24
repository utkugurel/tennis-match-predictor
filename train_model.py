import argparse
import glob
import sqlite3
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tennis_features import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_DIFF_FEATURES,
    MODEL_NUMERIC_FEATURES,
    build_match_example,
    get_stats_from_db,
)

MATCH_COLUMNS = ["tourney_date", "surface", "winner_name", "loser_name"]


def load_match_results(pattern: str = "data/atp_matches_*.csv") -> pd.DataFrame:
    """Load historical matches from CSVs and return a cleaned DataFrame."""
    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs matching {pattern}")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, usecols=MATCH_COLUMNS)
        except ValueError:
            # Fallback for files that miss optional columns
            df = pd.read_csv(path)
            missing = [col for col in MATCH_COLUMNS if col not in df.columns]
            if missing:
                print(f"Skipping {path} (missing columns: {missing})")
                continue
            df = df[MATCH_COLUMNS]
        frames.append(df)

    if not frames:
        raise RuntimeError("CSV files were found but none contained the required columns.")

    matches = pd.concat(frames, ignore_index=True)
    matches["year"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d", errors="coerce"
    ).dt.year
    matches = matches.dropna(subset=["surface", "winner_name", "loser_name", "year"])
    matches["year"] = matches["year"].astype(int)
    matches["surface"] = matches["surface"].str.strip()
    return matches


def build_training_data(
    matches: pd.DataFrame, db_path: str = "atp_stats.db", limit: int = None
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Create the training matrix by pairing player stats with outcomes."""
    rows = []
    labels = []
    skipped = 0
    total = len(matches) if limit is None else min(limit, len(matches))

    conn = sqlite3.connect(db_path)
    try:
        iterable = matches.itertuples(index=False)
        for idx, match in enumerate(iterable, start=1):
            if limit and idx > limit:
                break
            try:
                p1_stats = get_stats_from_db(
                    match.winner_name, int(match.year), match.surface, conn=conn
                )
                p2_stats = get_stats_from_db(
                    match.loser_name, int(match.year), match.surface, conn=conn
                )
            except ValueError:
                skipped += 1
                continue

            rows.append(build_match_example(p1_stats, p2_stats))
            labels.append(1)
            rows.append(build_match_example(p2_stats, p1_stats))
            labels.append(0)

            if idx % 5000 == 0:
                print(f"Processed {idx}/{total} matches...")
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No training samples were generated. Check the database build.")

    features = pd.DataFrame(rows)
    target = pd.Series(labels, name="target")
    print(f"Generated {len(features)} samples (skipped {skipped} matches).")
    return features, target, skipped


def train_pipeline(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Train the sklearn pipeline."""
    numeric_columns = (
        [f"p1_{col}" for col in MODEL_NUMERIC_FEATURES]
        + [f"p2_{col}" for col in MODEL_NUMERIC_FEATURES]
        + MODEL_DIFF_FEATURES
    )
    categorical_columns = [f"p1_{col}" for col in MODEL_CATEGORICAL_FEATURES] + [
        f"p2_{col}" for col in MODEL_CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, val_proba)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation AUC: {auc:.3f} | Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))

    return clf


def main():
    parser = argparse.ArgumentParser(description="Train the tennis match predictor model.")
    parser.add_argument(
        "--db-path", default="atp_stats.db", help="Path to the aggregated stats database."
    )
    parser.add_argument(
        "--matches-glob", default="data/atp_matches_*.csv", help="Glob for match CSV files."
    )
    parser.add_argument(
        "--model-path", default="tennis_predictor.pkl", help="Where to save the trained model."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of matches to process (for quick iterations).",
    )
    args = parser.parse_args()

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    matches = load_match_results(args.matches_glob)
    features, target, skipped = build_training_data(matches, args.db_path, args.limit)
    print(f"Training on {len(features)} samples (skipped {skipped} matches with missing stats).")

    pipeline = train_pipeline(features, target)
    joblib.dump(pipeline, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
