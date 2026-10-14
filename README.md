# Tennis Match Predictor

🎾 **[Live Demo](https://tennis-match-predictor.vercel.app/)** 🎾

Compare any two ATP seasons on any surface. The app predicts who would win and shows a dashboard comparing the two players: serve and return profiles, career trends, surface breakdowns and their real head-to-head record. Everything is written in Python with [Plotly Dash](https://dash.plotly.com/) and runs on Vercel.

## Features

- **Win probability**: a model trained on real ATP matches. It's symmetric: swapping the players flips the probability exactly.
- **Season comparison**: stat tiles, a serve/return radar chart (percentiles against all ATP seasons on that surface) and side-by-side stat bars.
- **Career trends**: win %, ranking, serve and return stats year by year, with the selected seasons highlighted.
- **Surfaces**: win % by surface, plus a year × surface heatmap for each player.
- **Head-to-head**: overall and per-surface record, a timeline and the full match list.

## Tech stack

- **App**: Plotly Dash, dash-bootstrap-components, Plotly (all Python)
- **Inference**: ONNX Runtime on a read-only SQLite database
- **Training**: pandas, scikit-learn, skl2onnx
- **Deployment**: Vercel (Python serverless function running Dash's Flask server)

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Dash app: layout, callbacks, `/health` and `/api/predict` |
| `api/index.py` | Alternative Vercel entry point (same WSGI server as `app.app`) |
| `assets/style.css` | Dark theme styling (served by Dash automatically) |
| `tennis/db.py` | Read-only queries (players, seasons, careers, head-to-head) |
| `tennis/predictor.py` | ONNX inference that gives the same result whichever player is first |
| `tennis/charts.py` | Plotly figure builders |
| `tennis_features.py` | Feature engineering shared by training and inference |
| `tennis/snapshot.py`, `scripts/make_manifest.py` | Integrity checks for the raw data snapshot |
| `tennis/etl.py`, `build_database.py` | Validation and building `atp_stats.db` |
| `train_model.py` | Training and evaluation, writes `model_card.json` |
| `convert_to_onnx.py` | ONNX export with a check that it matches the scikit-learn model |
| `tests/` | pytest suite |

## Getting started

```bash
git clone https://github.com/utkugurel/tennis-match-predictor.git
cd tennis-match-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime-only: requirements.txt
python app.py                         # http://localhost:8050
```

The repository includes the built database (`atp_stats.db`) and model (`tennis_predictor.onnx`), so the app runs without the raw data.

## Rebuilding from raw data

The raw CSVs are not in this repository (see [Data source](#data-source) and [Raw data snapshot](#raw-data-snapshot)). With the snapshot in `data/`:

```bash
python build_database.py     # verifies data/MANIFEST.json, validates rows, writes atp_stats.db
python train_model.py        # trains, evaluates, writes tennis_predictor.pkl + model_card.json
python convert_to_onnx.py    # writes tennis_predictor.onnx and checks it matches the .pkl
pytest
```

`build_database.py` refuses to run if any raw file is missing or its checksum differs from `data/MANIFEST.json`. It also drops and reports rows that fail consistency checks: impossible stats, a player playing themselves, duplicates, walkovers, and matches with no surface. Duplicate player ids that are clearly the same person (same name, country and birth date) are merged. The resulting validation report is stored in the database's `meta` table.

## Raw data snapshot

The original upstream repository is gone, so the raw data (snapshot `sackmann-atp-2024-12-30`, 67 files, about 110 MB) exists only in the maintainer's copies listed below. A fresh clone doesn't include it, so `build_database.py` can't run there until the data is restored. The app itself still works, because `atp_stats.db` and the model are committed.

**Status:**
- **Local archive:** `archive/sackmann-atp-2024-12-30.zip` (24.7 MB, SHA-256 `ccb1572aa39901f3db3182a2ebe55b9e2ec18954e3eaa3749cf04c2c9f5218b2`). It contains `data/` including `MANIFEST.json`. `archive/` is git-ignored and excluded from Vercel deployments, so it is never committed or published.
- **Off-machine copy:** handled privately by the maintainer (private cloud storage). Keep the raw files out of public redistribution, since the dataset author withdrew the original.

To restore the data (on a fresh clone, copy the zip in from the cloud backup first), unzip it into `data/`:

```bash
unzip archive/sackmann-atp-2024-12-30.zip -d data/
```

`data/MANIFEST.json` (committed) is the reference: it holds each file's size, SHA-256 and line count. `build_database.py` verifies every file against the manifest before building. Never regenerate the manifest (`scripts/make_manifest.py --force`) to make a mismatch go away; that would bless modified data.

## How the model works

- **Player strength**: Elo ratings, both overall and for the chosen surface. They are computed match by match over the whole history, so they react to results immediately and account for opponent strength.
- **Season stats**: each player's season totals on the surface: win %, matches played, age, height, and (when recorded) serve and return stats such as 1st/2nd serve points won, break points saved and converted, and return points won. The model sees the *differences* between the two players, plus handedness.
- **ATP ranking is deliberately not used.** It reflects the previous 12 months, so a player returning from injury looks weaker than their season was. Nadal in 2013 went 39–2 on clay with an average ranking of about 5. Ranking is still shown in the app.
- **No leakage**: during training, each player's season totals leave out the whole tournament the match belongs to. Leaving out only the match isn't enough, because the winner goes on to play later rounds of that event. Elo is taken from just before the match. In the app, the season's year-end Elo is used.
- **Missing data**: serve and return stats exist mostly from 1991 onwards. Earlier seasons get explicit "no stats" flags instead of fake zeros.
- **Model**: logistic regression. A random forest scores marginally better on held-out matches, but trees compress predictions for two exceptional seasons towards 50%, and that comparison is what this app is for.
- **Validation**: both orientations of a match (A vs B and B vs A) are always in the same split. There is also a separate time-based holdout: trained on seasons before 2019, tested on 2019+.

Current metrics are in [`model_card.json`](model_card.json): about 68% accuracy (AUC ≈ 0.75) on held-out matches and about 65% on 2019+. Earlier versions reported AUC 0.94, but that came from the leakage described above.

**Limitation**: Elo is relative to the players of its time, and the data starts in 1968, so ratings from the first few years are still settling. Comparisons across distant eras are rough.

## API

All JSON endpoints send CORS headers (`Access-Control-Allow-Origin: *`) and answer `OPTIONS` preflight requests, so they can be called from other origins.

**`POST /api/predict`** (alias: `POST /predict`)

```bash
curl -X POST http://localhost:8050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"player1_name": "Roger Federer", "player1_year": 2006,
       "player2_name": "Rafael Nadal", "player2_year": 2008, "surface": "Grass"}'
```

Names must match a player's full name exactly (case and accents are ignored) or their label from `/api/players`, e.g. `"Fred Hemmes (NED, b. 1981)"`. Possible errors:
- `400`: bad payload or unknown surface.
- `404`: unknown name (the response includes up to 5 `suggestions`), or no season for that player/year/surface.
- `409`: several players share the name (the response lists the `candidates` labels).
- `503`: the model could not be loaded.

**`GET /api/players?q=nad&limit=25`** returns `[{"player_id", "name", "label"}]` matching the search text. Without `q`, it returns the most active players. `limit` defaults to 100, maximum 200. (Before v2 this endpoint returned a plain list of every name.)

**`GET /health`** reports whether the model loaded, the model version and the data snapshot.

## Deployment

Vercel's Python runtime loads the WSGI app from `app.py` (`app:app`): `app` is Dash's Flask server, and the Dash object itself is `dash_app`. `vercel.json` rewrites every path to the function, and `api/index.py` exposes the same server. `.vercelignore` keeps raw data, training code and tests out of the deployment; only `requirements.txt` (runtime dependencies) is installed.

## Data source

Match data is by **Jeff Sackmann / [Tennis Abstract](http://www.tennisabstract.com/)**, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The original `JeffSackmann/tennis_atp` GitHub repository is no longer available (it has returned 404 since mid-2026). This project uses a frozen local snapshot (`sackmann-atp-2024-12-30`, ATP tour-level matches 1968–2024). Its files are pinned by checksum in [`data/MANIFEST.json`](data/MANIFEST.json). The raw files are not redistributed here; only derived statistics (the database and model) are published, under the same licence.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**; see [LICENSE](LICENSE).

**Important**: this project is for **educational and research purposes only** and may not be used commercially, in line with the data source licence.

**Player names**: professional tennis player names are used for factual, statistical and analytical purposes only. No endorsement by any player is implied.

## Author

Created by [Utku Gürel](https://github.com/utkugurel)
