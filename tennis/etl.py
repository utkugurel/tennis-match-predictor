"""Pandas helpers shared by build_database.py and train_model.py (not used at serve time)."""

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tennis_features import MATCH_STATS, SURFACES, TOTAL_COLUMNS

MATCH_FILE_RE = re.compile(r"^atp_matches_\d{4}\.csv$")
MAX_INVALID_SHARE = 0.005
# Chronological order of rounds within an event (match_num alone isn't reliable).
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "ER": 3, "RR": 4, "R128": 5, "R64": 6, "R32": 7,
               "R16": 8, "QF": 9, "SF": 10, "BR": 11, "F": 12}
ELO_START = 1500.0

MATCH_COLUMNS = [
    "tourney_id", "tourney_name", "tourney_level", "tourney_date", "surface",
    "match_num", "round", "best_of", "score", "minutes",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_rank", "loser_rank_points",
] + [f"{side}_{col}" for side in ("w", "l") for col in MATCH_STATS]


def match_files(data_dir: Path):
    return sorted(p for p in Path(data_dir).iterdir() if MATCH_FILE_RE.match(p.name))


def load_raw_matches(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in match_files(data_dir):
        df = pd.read_csv(path, low_memory=False)
        for col in MATCH_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[MATCH_COLUMNS])
    if not frames:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv files in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def validate_matches(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop unusable or inconsistent rows and report how many and why.

    Raises if more than MAX_INVALID_SHARE of rows break consistency rules
    (missing surfaces and walkovers are counted separately: they are
    excluded, not wrong).
    """
    df = raw.copy()
    report: Dict[str, int] = {"input_rows": len(df)}

    date = pd.to_datetime(df["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce")
    df["tourney_date"] = date
    df["year"] = date.dt.year

    surface = df["surface"].astype("string").str.strip().str.title()
    df["surface"] = surface.where(surface.isin(SURFACES))

    invalid = pd.Series(False, index=df.index)

    def flag(name: str, mask: pd.Series):
        nonlocal invalid
        mask = mask.fillna(False).astype(bool)
        report[name] = int((mask & ~invalid).sum())
        invalid |= mask

    flag("bad_date", date.isna())
    flag("missing_player_id", df["winner_id"].isna() | df["loser_id"].isna())
    flag("winner_equals_loser", df["winner_id"] == df["loser_id"])
    flag("duplicate_match", df.duplicated(["tourney_id", "match_num"], keep="first"))
    for side in ("w", "l"):
        stats = df[[f"{side}_{c}" for c in MATCH_STATS]]
        flag(f"{side}_negative_stat", (stats < 0).any(axis=1))
        flag(f"{side}_1stIn_gt_svpt", df[f"{side}_1stIn"] > df[f"{side}_svpt"])
        flag(f"{side}_1stWon_gt_1stIn", df[f"{side}_1stWon"] > df[f"{side}_1stIn"])
        flag(
            f"{side}_2ndWon_gt_2nd_serves",
            df[f"{side}_2ndWon"] > df[f"{side}_svpt"] - df[f"{side}_1stIn"],
        )
        flag(f"{side}_bpSaved_gt_bpFaced", df[f"{side}_bpSaved"] > df[f"{side}_bpFaced"])

    report["invalid_rows"] = int(invalid.sum())
    share = invalid.mean()
    if share > MAX_INVALID_SHARE:
        raise RuntimeError(f"{share:.2%} of rows failed validation: {report}")

    missing_surface = df["surface"].isna() & ~invalid
    report["missing_surface"] = int(missing_surface.sum())

    # Walkovers were never played: they don't count in head-to-heads or stats.
    score = df["score"].astype("string").str.strip().str.upper()
    walkover = score.isin(["W/O", "WALKOVER"]).fillna(False) & ~invalid & ~missing_surface
    report["walkover"] = int(walkover.sum())

    clean = df[~invalid & ~missing_surface & ~walkover].copy()
    clean["winner_id"] = clean["winner_id"].astype(int)
    clean["loser_id"] = clean["loser_id"].astype(int)
    clean["year"] = clean["year"].astype(int)
    clean["_round_order"] = clean["round"].map(ROUND_ORDER).fillna(4)
    clean = (
        clean.sort_values(["tourney_date", "tourney_id", "_round_order", "match_num"])
        .drop(columns="_round_order")
        .reset_index(drop=True)
    )
    clean.insert(0, "match_id", np.arange(1, len(clean) + 1))
    report["output_rows"] = len(clean)
    return clean, report


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, player) holding that match's contribution to TOTAL_COLUMNS."""
    sides = []
    for me, opp, my_stats, opp_stats, won in (
        ("winner", "loser", "w", "l", 1),
        ("loser", "winner", "l", "w", 0),
    ):
        part = pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "tourney_id": matches["tourney_id"].to_numpy(),
                "player_id": matches[f"{me}_id"].to_numpy(),
                "opponent_id": matches[f"{opp}_id"].to_numpy(),
                "year": matches["year"].to_numpy(),
                "surface": matches["surface"].to_numpy(),
                "wins": won,
                "losses": 1 - won,
            }
        )
        has_stats = (matches[f"{my_stats}_svpt"].fillna(0).to_numpy() > 0) & (
            matches[f"{opp_stats}_svpt"].fillna(0).to_numpy() > 0
        )
        part["serve_matches"] = has_stats.astype(int)
        for col in MATCH_STATS:
            # Only count stats from matches where both sides were recorded.
            part[f"p_{col}"] = np.where(has_stats, matches[f"{my_stats}_{col}"].fillna(0), 0.0)
            part[f"o_{col}"] = np.where(has_stats, matches[f"{opp_stats}_{col}"].fillna(0), 0.0)
        for name, source in (
            ("rank", f"{me}_rank"),
            ("rank_points", f"{me}_rank_points"),
            ("age", f"{me}_age"),
        ):
            values = matches[source].to_numpy(dtype=float)
            present = ~np.isnan(values)
            part[f"{name}_sum"] = np.where(present, values, 0.0)
            part[f"{name}_n"] = present.astype(int)
        sides.append(part)
    long = pd.concat(sides, ignore_index=True)
    return long[["match_id", "tourney_id", "player_id", "opponent_id", "year", "surface"] + TOTAL_COLUMNS]


def aggregate_seasons(long: pd.DataFrame) -> pd.DataFrame:
    return long.groupby(["player_id", "year", "surface"], as_index=False)[TOTAL_COLUMNS].sum()


def _k_factor(matches_played: int) -> float:
    # FiveThirtyEight's tennis K-factor: new players move fast, veterans slowly.
    return 250.0 / (matches_played + 5) ** 0.4


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Overall and surface-specific Elo ratings, processed in match order.

    Returns, per match_id, both players' ratings *before* the match:
    winner_elo, loser_elo, winner_surface_elo, loser_surface_elo.
    """
    overall, surface_elo = {}, {}
    played, surface_played = {}, {}
    pre = np.empty((len(matches), 4))

    for i, (surface, winner, loser) in enumerate(
        zip(matches["surface"], matches["winner_id"], matches["loser_id"])
    ):
        w_s, l_s = (winner, surface), (loser, surface)
        rw, rl = overall.get(winner, ELO_START), overall.get(loser, ELO_START)
        sw, sl = surface_elo.get(w_s, ELO_START), surface_elo.get(l_s, ELO_START)
        pre[i] = rw, rl, sw, sl

        expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        overall[winner] = rw + _k_factor(played.get(winner, 0)) * (1 - expected)
        overall[loser] = rl - _k_factor(played.get(loser, 0)) * (1 - expected)
        played[winner] = played.get(winner, 0) + 1
        played[loser] = played.get(loser, 0) + 1

        expected = 1.0 / (1.0 + 10 ** ((sl - sw) / 400.0))
        surface_elo[w_s] = sw + _k_factor(surface_played.get(w_s, 0)) * (1 - expected)
        surface_elo[l_s] = sl - _k_factor(surface_played.get(l_s, 0)) * (1 - expected)
        surface_played[w_s] = surface_played.get(w_s, 0) + 1
        surface_played[l_s] = surface_played.get(l_s, 0) + 1

    pre_match = pd.DataFrame(
        pre.round(), columns=["winner_elo", "loser_elo", "winner_surface_elo", "loser_surface_elo"]
    ).astype(int)
    pre_match.insert(0, "match_id", matches["match_id"].to_numpy())
    return pre_match


def player_match_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, player) with that player's pre-match ratings.

    `matches` must carry the compute_elo columns. This is the training
    feature, and season_elo() averages it for serving, so both sides use the
    same kind of number.
    """
    sides = []
    for side in ("winner", "loser"):
        sides.append(
            pd.DataFrame(
                {
                    "match_id": matches["match_id"].to_numpy(),
                    "player_id": matches[f"{side}_id"].to_numpy(),
                    "year": matches["year"].to_numpy(),
                    "surface": matches["surface"].to_numpy(),
                    "elo": matches[f"{side}_elo"].to_numpy(dtype=float),
                    "surface_elo": matches[f"{side}_surface_elo"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(sides, ignore_index=True)


def season_elo(per_match: pd.DataFrame) -> pd.DataFrame:
    """Mean pre-match Elo per (player_id, year, surface): the season strength
    served by the app. Matches the training feature in kind (a pre-match
    rating), unlike a season-end rating that already includes the results."""
    return (
        per_match.groupby(["player_id", "year", "surface"], as_index=False)[["elo", "surface_elo"]]
        .mean()
        .round(1)
    )
