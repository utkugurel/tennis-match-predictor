import numpy as np
import pandas as pd

from tennis.etl import compute_elo, player_match_elo, season_elo, to_long
from tennis_features import MATCH_STATS


def _matches():
    """Three matches for player 1 in one season, two on clay, one on hard."""
    rows = [
        (1, "t1", 2020, "Clay", 1, 2),
        (2, "t1", 2020, "Clay", 1, 3),
        (3, "t2", 2020, "Hard", 3, 1),
    ]
    df = pd.DataFrame(rows, columns=["match_id", "tourney_id", "year", "surface", "winner_id", "loser_id"])
    for side in ("w", "l"):
        for col in MATCH_STATS:
            df[f"{side}_{col}"] = 10.0
    for side in ("winner", "loser"):
        for col in ("rank", "rank_points", "age"):
            df[f"{side}_{col}"] = 1.0
    return df


def test_served_season_elo_is_mean_of_training_feature():
    matches = _matches()
    matches = matches.merge(compute_elo(matches), on="match_id")
    per_match = player_match_elo(matches)  # what train_model uses per match
    served = season_elo(per_match)  # what build_database stores for the app

    training_rows = per_match[(per_match.player_id == 1) & (per_match.surface == "Clay")]
    served_row = served[(served.player_id == 1) & (served.year == 2020) & (served.surface == "Clay")].iloc[0]
    assert served_row["elo"] == round(training_rows["elo"].mean(), 1)
    assert served_row["surface_elo"] == round(training_rows["surface_elo"].mean(), 1)
    # Pre-match: the first match is played at the starting rating.
    assert per_match[(per_match.match_id == 1) & (per_match.player_id == 1)]["elo"].iloc[0] == 1500


def test_missing_opponent_stats_are_not_counted():
    matches = _matches().iloc[[0]].copy()
    for col in MATCH_STATS:
        matches[f"l_{col}"] = np.nan
    long = to_long(matches)
    assert (long["serve_matches"] == 0).all()
    stat_columns = [c for c in long.columns if c.startswith(("p_", "o_"))]
    assert (long[stat_columns] == 0).all().all()
