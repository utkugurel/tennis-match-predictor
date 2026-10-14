"""Feature engineering shared by training (train_model.py) and inference.

Works on plain mappings so it runs without pandas at serve time. A player's
"season totals" are summed per (player, year, surface); during training the
match being predicted is subtracted from them first (leave-one-out).
"""

import math
import unicodedata
from typing import Any, Dict, List, Mapping, Optional

SURFACES: List[str] = ["Hard", "Clay", "Grass", "Carpet"]


def fold_name(text: Optional[str]) -> str:
    """Lower-case, accent-free, trimmed form of a name, for search and matching."""
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


# Per-match stat columns from the raw data, summed into season totals for the
# player ("p_") and for their opponents ("o_").
MATCH_STATS: List[str] = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
]

# Every additive column in a season-totals record.
TOTAL_COLUMNS: List[str] = (
    ["wins", "losses", "serve_matches"]
    + [f"p_{col}" for col in MATCH_STATS]
    + [f"o_{col}" for col in MATCH_STATS]
    + ["rank_sum", "rank_n", "rank_points_sum", "rank_points_n", "age_sum", "age_n"]
)

UNRANKED = 2000.0

# Stats that only exist when match stats were recorded (mostly 1991 onwards).
SERVE_FEATURES: List[str] = [
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

# ATP ranking and ranking points are deliberately not model inputs: they
# describe the previous 12 months, so a player returning from injury (Nadal
# 2013) looks weaker than their season was. Elo ratings react to results
# immediately and account for opponent strength. Training uses the rating
# just before each match; serving uses the season's mean pre-match rating
# (tennis.etl.season_elo), so both are the same kind of number.
PLAYER_NUMERIC_FEATURES: List[str] = [
    "elo",
    "surface_elo",
    "age",
    "ht",
    "matches_played",
    "win_pct_smoothed",
] + SERVE_FEATURES

HANDS = ("L", "R", "U")

# Model input, in order: p1 - p2 differences, then per-player flags for
# whether serve stats exist (the differences are NaN otherwise and get
# imputed), then one-hot hands (R is the baseline). Only differences are used so the model sees no "slot" information
# beyond handedness; predict() additionally averages both orderings.
MODEL_FEATURES: List[str] = (
    [f"diff_{name}" for name in PLAYER_NUMERIC_FEATURES]
    + ["p1_has_serve_stats", "p2_has_serve_stats"]
    + ["p1_hand_L", "p1_hand_U", "p2_hand_L", "p2_hand_U"]
)


def _num(value: Any) -> Optional[float]:
    """Convert to float; None/NaN/garbage become None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _ratio(numerator: Any, denominator: Any) -> float:
    top, bottom = _num(numerator), _num(denominator)
    if top is None or not bottom:
        return math.nan
    return top / bottom


def _mean(total: Any, count: Any) -> Optional[float]:
    total, count = _num(total), _num(count)
    if not count:
        return None
    return total / count


def normalize_hand(hand_value: Optional[str]) -> str:
    hand = (hand_value or "").strip().upper() if isinstance(hand_value, str) else ""
    return hand if hand in HANDS else "U"


def compute_features(totals: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn a season-totals record into display stats and model features.

    `totals` needs TOTAL_COLUMNS plus `hand`, `ht`, `elo` and `surface_elo`.
    Serve/return stats are NaN (not 0) when the season has no recorded
    match stats.
    """
    t = {col: _num(totals.get(col)) or 0.0 for col in TOTAL_COLUMNS}
    wins, losses = t["wins"], t["losses"]
    matches = wins + losses
    serve_matches = t["serve_matches"]
    has_stats = serve_matches > 0 and t["p_svpt"] > 0

    rank = _mean(t["rank_sum"], t["rank_n"])
    rank_points = _mean(t["rank_points_sum"], t["rank_points_n"])
    age = _mean(t["age_sum"], t["age_n"])

    features: Dict[str, Any] = {
        "hand": normalize_hand(totals.get("hand")),
        "elo": _num(totals.get("elo")) or math.nan,
        "surface_elo": _num(totals.get("surface_elo")) or math.nan,
        "ht": _num(totals.get("ht")) or math.nan,
        "age": age if age is not None else math.nan,
        "rank": rank,
        "log_rank": math.log(rank if rank and rank > 0 else UNRANKED),
        "rank_points": rank_points or 0.0,
        "wins": wins,
        "losses": losses,
        "matches_played": matches,
        "win_pct": _ratio(wins, matches) if matches else 0.0,
        # Shrunk towards 50% so a 2-0 season doesn't look like a 100% player.
        "win_pct_smoothed": (wins + 2.0) / (matches + 4.0),
        "has_serve_stats": 1.0 if has_stats else 0.0,
        "serve_matches": serve_matches,
    }

    if has_stats:
        svpt, first_in = t["p_svpt"], t["p_1stIn"]
        opp_svpt = t["o_svpt"]
        features.update(
            {
                "ace_rate": _ratio(t["p_ace"], svpt),
                "df_rate": _ratio(t["p_df"], svpt),
                "first_serve_pct": _ratio(first_in, svpt),
                "first_serve_win_pct": _ratio(t["p_1stWon"], first_in),
                "second_serve_win_pct": _ratio(t["p_2ndWon"], max(svpt - first_in, 0.0)),
                "service_points_won_pct": _ratio(t["p_1stWon"] + t["p_2ndWon"], svpt),
                "bp_save_pct": _ratio(t["p_bpSaved"], t["p_bpFaced"]),
                "return_points_won_pct": _ratio(
                    max(opp_svpt - t["o_1stWon"] - t["o_2ndWon"], 0.0), opp_svpt
                ),
                "bp_conversion_pct": _ratio(
                    max(t["o_bpFaced"] - t["o_bpSaved"], 0.0), t["o_bpFaced"]
                ),
                "ace_per_match": _ratio(t["p_ace"], serve_matches),
                "df_per_match": _ratio(t["p_df"], serve_matches),
                "serve_games_per_match": _ratio(t["p_SvGms"], serve_matches),
                "bp_faced_per_match": _ratio(t["p_bpFaced"], serve_matches),
                "bp_created_per_match": _ratio(t["o_bpFaced"], serve_matches),
            }
        )
    else:
        features.update({name: math.nan for name in SERVE_FEATURES})
    return features


def build_model_row(p1: Mapping[str, Any], p2: Mapping[str, Any]) -> List[float]:
    """Model input vector (ordered as MODEL_FEATURES) for p1 vs p2."""
    row: List[float] = []
    for name in PLAYER_NUMERIC_FEATURES:
        a, b = _num(p1.get(name)), _num(p2.get(name))
        row.append(math.nan if a is None or b is None else a - b)
    for player in (p1, p2):
        row.append(1.0 if player.get("has_serve_stats") else 0.0)
    for player in (p1, p2):
        hand = normalize_hand(player.get("hand"))
        row.append(1.0 if hand == "L" else 0.0)
        row.append(1.0 if hand == "U" else 0.0)
    return row
