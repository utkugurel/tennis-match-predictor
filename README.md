# Tennis Match Predictor

🎾 **[Live Demo](https://tennis-match-predictor.vercel.app/)** 🎾

Compare prime versions of any two ATP players on a given surface. This project aggregates historical season-long player stats, trains a Random Forest model, and serves predictions through a Flask API consumed by a single-page Tailwind UI.

## Repository Layout

| File | Purpose |
| --- | --- |
| `build_database.py` | ETL script that ingests `data/atp_matches_*.csv`, pivots winner/loser rows, and stores yearly surface stats in `atp_stats.db`. |
| `tennis_features.py` | Shared feature engineering helpers used by both training and inference (`get_stats_from_db`, `build_match_example`). |
| `train_model.py` | Builds the training matrix from historical matches, fits a `Pipeline(StandardScaler + OneHotEncoder + LogisticRegression)`, saves `tennis_predictor.pkl`. |
| `app.py` | Flask backend exposing `/predict` and `/health`, loads the trained pipeline, and responds with win probabilities plus season summaries. |
| `index.html` | Tailwind-based UI that posts to the API and visualizes the probabilities. |

## Prerequisites

1. Python 3.10+
2. [OPTIONAL] `virtualenv` or `venv`
3. ATP CSVs already present in `data/` (repo includes them)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — Build the SQLite Stats Database

```bash
python build_database.py
```

This creates/overwrites `atp_stats.db` with the `player_yearly_stats` table. Expect the step to take several minutes due to the number of CSVs.

## Phase 2 — Train the Model

```bash
python train_model.py \
  --db-path atp_stats.db \
  --matches-glob "data/atp_matches_*.csv" \
  --model-path tennis_predictor.pkl
```

Training iterates through every valid match, generates two samples (winner vs loser and vice versa), fits the pipeline, prints validation metrics, and writes `tennis_predictor.pkl`.

**Faster smoke test:** pass `--limit 20000` (matches) to iterate quickly while validating the pipeline end-to-end.

## Phase 3 — Run the API

```bash
export FLASK_APP=app.py
export MODEL_PATH=tennis_predictor.pkl
flask run --host 0.0.0.0 --port 5000
```

`/predict` accepts:

```json
{
  "player1_name": "Roger Federer",
  "player1_year": 2015,
  "player2_name": "Rafael Nadal",
  "player2_year": 2007,
  "surface": "Hard"
}
```

## Phase 4 — Frontend

Open `index.html` in your browser. If your backend runs on a non-default host/port, set `window.API_URL` before including the script or adjust the constant inside the file.

## Testing the Model & API

1. **Unit-style check:** ensure the helper module imports and compiles.
   ```bash
   python -m py_compile tennis_features.py train_model.py app.py
   ```
2. **Predictive sanity check:** after training, run a quick curl request.
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"player1_name":"Roger Federer","player1_year":2015,"player2_name":"Rafael Nadal","player2_year":2007,"surface":"Hard"}'
   ```
3. **Browser test:** load the UI, input two players/years/surface, verify probabilities render and match the API response.

## Troubleshooting

- **Missing player/year/surface**: ensure you ran `build_database.py` after updating CSVs, and confirm the combination exists in `player_yearly_stats`.
- **Model load error**: delete/replace `tennis_predictor.pkl` and re-run `train_model.py`.
- **Slow preprocessing**: use the `--limit` flag while iterating, then remove it for a full training pass once satisfied.

## Data Source

Match data provided by [Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
