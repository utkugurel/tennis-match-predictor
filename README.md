# Tennis Match Predictor

🎾 **[Live Demo](https://tennis-match-predictor.vercel.app/)** 🎾

Compare prime versions of any two ATP players on a given surface. This project aggregates historical season-long player stats, trains a Random Forest model, and serves predictions through a Flask API consumed by a single-page Tailwind UI.

## Features

- 🤖 **AI-Powered Predictions**: Random Forest classifier trained on historical ATP match data
- 📊 **Comprehensive Stats**: Season-long player statistics including serve, return, and surface performance
- 🎨 **Modern UI**: Clean, responsive interface with player autocomplete
- ⚡ **Fast Inference**: ONNX-optimised model for quick predictions
- 🌐 **Deployed on Vercel**: Serverless architecture with automatic deployments

## Tech Stack

- **Backend**: Flask, ONNX Runtime, SQLite
- **Frontend**: Vanilla JavaScript, Tailwind CSS
- **ML**: Scikit-Learn, Random Forest Classifier
- **Deployment**: Vercel (serverless functions)

## Project Structure

| File | Purpose |
| --- | --- |
| `build_database.py` | ETL script that ingests `data/atp_matches_*.csv`, pivots winner/loser rows, and stores yearly surface stats in `atp_stats.db` |
| `tennis_features.py` | Shared feature engineering helpers used by both training and inference |
| `train_model.py` | Builds the training matrix from historical matches, fits a Random Forest pipeline, and saves the model |
| `convert_to_onnx.py` | Converts the trained Scikit-Learn model to ONNX format for optimised inference |
| `app.py` | Flask backend exposing `/api/predict`, `/api/players`, and `/health` endpoints |
| `static_content/index.html` | Tailwind-based UI with autocomplete and real-time predictions |

## Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- ATP match data CSVs (included in `data/` directory)

### Installation

```bash
# Clone the repository
git clone https://github.com/utkugurel/tennis-match-predictor.git
cd tennis-match-predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Building the Database

The first step is to process the raw CSV match data into an SQLite database:

```bash
python build_database.py
```

This creates `atp_stats.db` with the `player_yearly_stats` table. The process takes several minutes due to the volume of historical data.

## Training the Model

Train the Random Forest classifier on historical match data:

```bash
python train_model.py \
  --db-path atp_stats.db \
  --matches-glob "data/atp_matches_*.csv" \
  --model-path tennis_predictor.pkl
```

The script:
- Generates training samples (winner vs loser and vice versa)
- Fits a pipeline with StandardScaler, OneHotEncoder, and RandomForestClassifier
- Prints validation metrics (AUC, accuracy, classification report)
- Saves the trained model to `tennis_predictor.pkl`

**Quick test**: Use `--limit 1000` to train on a subset of matches for faster iteration.

## Converting to ONNX

For optimised inference in production, convert the model to ONNX format:

```bash
python convert_to_onnx.py
```

This creates `tennis_predictor.onnx`, which is significantly faster and has smaller dependencies than the Scikit-Learn model.

## Running Locally

Start the Flask development server:

```bash
export FLASK_APP=app.py
flask run
```

The app will be available at `http://localhost:5000`. The frontend automatically loads player names for autocomplete and comes pre-filled with an example match-up.

### API Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player1_name": "Roger Federer",
    "player1_year": 2015,
    "player2_name": "Rafael Nadal",
    "player2_year": 2007,
    "surface": "Hard"
  }'
```

## Deployment

The app is configured for Vercel with serverless functions:

1. Connect your GitHub repository to Vercel
2. Vercel automatically detects the configuration from `vercel.json`
3. Push to `main` branch to trigger automatic deployments

The `data/` directory is excluded via `.vercelignore` to stay within size limits.

## Model Performance

The Random Forest classifier achieves:
- **Validation AUC**: ~0.94
- **Accuracy**: ~85%

Key features include:
- Player statistics (serve %, return %, break points, etc.)
- Surface-specific performance
- Relative differences (rank, age, height)

## Troubleshooting

**Missing player/year/surface combination**  
Ensure you've run `build_database.py` and that the player competed in that year on that surface.

**Model load error**  
Delete `tennis_predictor.pkl` or `tennis_predictor.onnx` and retrain the model.

**Slow training**  
Use the `--limit` flag to train on a subset of matches during development.

## Data Source

Match data provided by [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp), licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** - see the [LICENSE](LICENSE) file for details.

**Important**: This project is for **educational and research purposes only**. It may not be used for commercial purposes in compliance with the data source license.

**Player Names**: Professional tennis player names are used for factual, statistical, and analytical purposes only. No endorsement by any player is implied.

## Author

Created by [Utku Gürel](https://github.com/utkugurel)

