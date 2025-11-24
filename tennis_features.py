import math
import sqlite3
from typing import Any, Dict, List, Optional


MODEL_NUMERIC_FEATURES: List[str] = [
    "ht",
    "age",
    "rank",
    "rank_points",
    "matches_played",
    "wins",
    "losses",
    "win_pct",
    "ace_rate",
    "df_rate",
    "first_serve_pct",
    "first_serve_win_pct",
    "second_serve_win_pct",
    "service_points_won_pct",
    "bp_save_pct",
    "return_points_won_pct",
    "bp_conversion_pct",
    "ace_per_match",
    "df_per_match",
    "serve_games_per_match",
    "bp_faced_per_match",
    "bp_created_per_match",
]

MODEL_CATEGORICAL_FEATURES: List[str] = ["hand"]

MODEL_BASE_FEATURES: List[str] = list(MODEL_NUMERIC_FEATURES) + list(
    MODEL_CATEGORICAL_FEATURES
)

MODEL_DIFF_FEATURES: List[str] = ["log_rank_diff", "age_diff", "height_diff"]


def _safe_number(value: Any) -> float:
    """Convert None/NaN/str numbers to a float, defaulting to 0."""
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return number


def _safe_divide(numerator: Any, denominator: Any) -> float:
    top = _safe_number(numerator)
    bottom = _safe_number(denominator)
    if bottom == 0:
        return 0.0
    return top / bottom


def _normalize_hand(hand_value: Optional[str]) -> str:
    hand = (hand_value or "").strip().upper()
    return hand if hand in {"L", "R", "U"} else "U"


def record_to_feature_dict(record: sqlite3.Row) -> Dict[str, Any]:
    """Convert a SQLite row into the feature dictionary used by the model."""
    matches = _safe_number(record["matches_played"])
    wins = _safe_number(record["wins"])
    losses = _safe_number(record["losses"])
    p_svpt = _safe_number(record["p_svpt"])
    p_1stin = _safe_number(record["p_1stIn"])
    p_1stwon = _safe_number(record["p_1stWon"])
    p_2ndwon = _safe_number(record["p_2ndWon"])
    p_sg = _safe_number(record["p_SvGms"])
    p_bp_faced = _safe_number(record["p_bpFaced"])
    p_bp_saved = _safe_number(record["p_bpSaved"])
    o_svpt = _safe_number(record["o_svpt"])
    o_1stwon = _safe_number(record["o_1stWon"])
    o_2ndwon = _safe_number(record["o_2ndWon"])
    o_bp_faced = _safe_number(record["o_bpFaced"])
    o_bp_saved = _safe_number(record["o_bpSaved"])
    p_aces = _safe_number(record["p_ace"])
    p_df = _safe_number(record["p_df"])

    second_serve_points = max(p_svpt - p_1stin, 0.0)
    opp_service_points_won = o_1stwon + o_2ndwon
    opp_break_points_converted = max(o_bp_faced - o_bp_saved, 0.0)

    features: Dict[str, Any] = {
        "player_name": record["player_name"],
        "year": _safe_number(record["year"]),
        "surface": record["surface"],
        "hand": _normalize_hand(record["p_hand"]),
        "ht": _safe_number(record["p_ht"]),
        "age": _safe_number(record["p_age"]),
        "rank": _safe_number(record["p_rank"]),
        "rank_points": _safe_number(record["p_rank_points"]),
        "matches_played": matches,
        "wins": wins,
        "losses": losses,
        "win_pct": _safe_divide(wins, wins + losses),
        "ace_rate": _safe_divide(p_aces, p_svpt),
        "df_rate": _safe_divide(p_df, p_svpt),
        "first_serve_pct": _safe_divide(p_1stin, p_svpt),
        "first_serve_win_pct": _safe_divide(p_1stwon, p_1stin),
        "second_serve_win_pct": _safe_divide(p_2ndwon, second_serve_points),
        "service_points_won_pct": _safe_divide(p_1stwon + p_2ndwon, p_svpt),
        "bp_save_pct": _safe_divide(p_bp_saved, p_bp_faced),
        "return_points_won_pct": _safe_divide(
            max(o_svpt - opp_service_points_won, 0.0), o_svpt
        ),
        "bp_conversion_pct": _safe_divide(opp_break_points_converted, o_bp_faced),
        "ace_per_match": _safe_divide(p_aces, matches),
        "df_per_match": _safe_divide(p_df, matches),
        "serve_games_per_match": _safe_divide(p_sg, matches),
        "bp_faced_per_match": _safe_divide(p_bp_faced, matches),
        "bp_created_per_match": _safe_divide(o_bp_faced, matches),
    }
    return features


def get_stats_from_db(
    player_name: str,
    year: int,
    surface: str,
    db_path: str = "atp_stats.db",
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """
    Retrieve a player's season-long stats for a given year and surface and
    return the engineered features the model expects.
    """
    surface_clean = (surface or "").strip()
    player_clean = (player_name or "").strip()
    if not player_clean or not surface_clean:
        raise ValueError("player_name and surface are required.")

    close_conn = False
    connection = conn
    previous_factory = None
    if connection is None:
        connection = sqlite3.connect(db_path)
        close_conn = True
    else:
        previous_factory = connection.row_factory
    connection.row_factory = sqlite3.Row

    query = """
        SELECT *
        FROM player_yearly_stats
        WHERE lower(player_name) = lower(?)
          AND year = ?
          AND upper(surface) = upper(?)
        LIMIT 1
    """

    cursor = connection.execute(query, (player_clean, int(year), surface_clean))
    record = cursor.fetchone()

    if previous_factory is not None:
        connection.row_factory = previous_factory

    if close_conn:
        connection.close()

    if record is None:
        raise ValueError(
            f"No stats found for {player_name} ({year}, {surface_clean}). "
            "Run build_database.py and ensure the player/year/surface combination exists."
        )

    return record_to_feature_dict(record)


def build_match_example(p1_stats: Dict[str, Any], p2_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine two player feature dictionaries into the prefixed format the
    training pipeline expects (p1_* vs p2_* features).
    """
    row: Dict[str, Any] = {}
    
    # Calculate differences before prefixing
    p1_rank = _safe_number(p1_stats.get("rank", 0))
    p2_rank = _safe_number(p2_stats.get("rank", 0))
    # Avoid log(0) or log(negative) issues by using a small epsilon if rank is 0 (unranked)
    # However, unranked usually means high rank number effectively. Let's treat 0 as a very high number for log diff?
    # Or just use the raw values if they are safe.
    # A safer approach for log rank: if rank > 0, use log(rank). If 0, treat as e.g. 2000 (low rank).
    def get_log_rank(r):
        return math.log(r) if r > 0 else math.log(2000)

    row["log_rank_diff"] = get_log_rank(p1_rank) - get_log_rank(p2_rank)
    row["age_diff"] = _safe_number(p1_stats.get("age", 0)) - _safe_number(p2_stats.get("age", 0))
    row["height_diff"] = _safe_number(p1_stats.get("ht", 0)) - _safe_number(p2_stats.get("ht", 0))

    for key in MODEL_BASE_FEATURES:
        row[f"p1_{key}"] = p1_stats.get(key, 0)
        row[f"p2_{key}"] = p2_stats.get(key, 0)
    return row
