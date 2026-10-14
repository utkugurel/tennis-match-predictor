"""Read-only queries against atp_stats.db used by the app."""

import os
import sqlite3
from bisect import bisect_left
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from tennis_features import SERVE_FEATURES, SURFACES, TOTAL_COLUMNS, compute_features, fold_name

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "atp_stats.db"))
PRIME_MIN_MATCHES = 10

_SUM_TOTALS = ", ".join(f'SUM("{col}") AS "{col}"' for col in TOTAL_COLUMNS)


def _connect() -> sqlite3.Connection:
    # immutable=1: the file never changes at runtime (Vercel's disk is read-only).
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql: str, params=()) -> List[Dict]:
    with closing(_connect()) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _with_features(record: Dict, player: Dict) -> Dict:
    merged = {**record, "hand": player.get("hand"), "ht": player.get("ht")}
    return {**record, **compute_features(merged)}


@lru_cache(maxsize=1)
def meta() -> Dict[str, str]:
    return {row["key"]: row["value"] for row in _query("SELECT key, value FROM meta")}


def search_players(text: str, limit: int = 25) -> List[Dict]:
    """Dropdown options for players whose name contains `text`."""
    needle = fold_name(text)
    if len(needle) < 2:
        return []
    # Typed % and _ are literal characters, not LIKE wildcards.
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = _query(
        """
        SELECT player_id, label FROM players
        WHERE search_name LIKE ? ESCAPE '\\'
        ORDER BY (' ' || search_name LIKE ? ESCAPE '\\') DESC, total_matches DESC
        LIMIT ?
        """,
        (f"%{escaped}%", f"% {escaped}%", limit),
    )
    return [{"label": row["label"], "value": row["player_id"]} for row in rows]


@lru_cache(maxsize=4)
def top_players(limit: int) -> List[Dict]:
    """Players with the most matches, as dropdown options."""
    rows = _query(
        "SELECT player_id, label FROM players ORDER BY total_matches DESC LIMIT ?", (int(limit),)
    )
    return [{"label": row["label"], "value": row["player_id"]} for row in rows]


@lru_cache(maxsize=1)
def _exact_name_index() -> Dict[str, List[int]]:
    """Folded name and folded label -> player ids."""
    index: Dict[str, List[int]] = {}
    for row in _query("SELECT player_id, name, label FROM players"):
        for key in {fold_name(row["name"]), fold_name(row["label"])}:
            index.setdefault(key, []).append(row["player_id"])
    return index


def players_by_exact_name(text: str) -> List[Dict]:
    """Players whose name or label equals `text`, ignoring case and accents.

    More than one result means the name is shared; the caller must ask for a
    label to tell them apart.
    """
    ids = _exact_name_index().get(fold_name(text), [])
    return [player(pid) for pid in sorted(set(ids))]


@lru_cache(maxsize=2048)
def player(player_id: int) -> Optional[Dict]:
    rows = _query("SELECT * FROM players WHERE player_id = ?", (int(player_id),))
    return rows[0] if rows else None


@lru_cache(maxsize=4096)
def seasons_for(player_id: int, surface: str) -> List[Dict]:
    """Seasons (newest first) the player has on `surface`, with match counts."""
    return _query(
        """
        SELECT year, wins, losses, matches_played FROM player_yearly_stats
        WHERE player_id = ? AND surface = ?
        ORDER BY year DESC
        """,
        (int(player_id), surface),
    )


def prime_season(player_id: int, surface: str) -> Optional[int]:
    """Best season on a surface by smoothed win %, preferring >= 10 matches."""
    seasons = seasons_for(player_id, surface)
    if not seasons:
        return None
    eligible = [s for s in seasons if s["matches_played"] >= PRIME_MIN_MATCHES] or seasons
    best = max(eligible, key=lambda s: ((s["wins"] + 2) / (s["matches_played"] + 4), s["matches_played"]))
    return best["year"]


@lru_cache(maxsize=4096)
def season(player_id: int, year: int, surface: str) -> Optional[Dict]:
    """Season totals plus engineered features for one player/year/surface."""
    info = player(player_id)
    rows = _query(
        "SELECT * FROM player_yearly_stats WHERE player_id = ? AND year = ? AND surface = ?",
        (int(player_id), int(year), surface),
    )
    if not rows or info is None:
        return None
    return _with_features(rows[0], info)


@lru_cache(maxsize=1024)
def career(player_id: int, surface: Optional[str] = None) -> List[Dict]:
    """Per-year features, on one surface or summed over all surfaces."""
    info = player(player_id)
    if info is None:
        return []
    where, params = "player_id = ?", [int(player_id)]
    # `elo` is the year-end rating, identical across a year's surface rows.
    elo_columns = "MAX(elo) AS elo"
    if surface:
        where += " AND surface = ?"
        params.append(surface)
        elo_columns += ", MAX(surface_elo) AS surface_elo"
    rows = _query(
        f"""
        SELECT year, {elo_columns}, {_SUM_TOTALS} FROM player_yearly_stats
        WHERE {where} GROUP BY year ORDER BY year
        """,
        params,
    )
    return [_with_features(row, info) for row in rows]


@lru_cache(maxsize=1024)
def surface_splits(player_id: int, year: Optional[int] = None) -> List[Dict]:
    """Per-surface features for a whole career or a single year."""
    info = player(player_id)
    if info is None:
        return []
    where, params = "player_id = ?", [int(player_id)]
    if year is not None:
        where += " AND year = ?"
        params.append(int(year))
    rows = _query(
        f"""
        SELECT surface, {_SUM_TOTALS} FROM player_yearly_stats
        WHERE {where} GROUP BY surface
        """,
        params,
    )
    order = {name: i for i, name in enumerate(SURFACES)}
    return sorted((_with_features(row, info) for row in rows), key=lambda r: order.get(r["surface"], 9))


@lru_cache(maxsize=1024)
def year_surface_grid(player_id: int) -> List[Dict]:
    return _query(
        """
        SELECT year, surface, wins, losses, matches_played FROM player_yearly_stats
        WHERE player_id = ? ORDER BY year
        """,
        (int(player_id),),
    )


@lru_cache(maxsize=512)
def head_to_head(player_a: int, player_b: int) -> List[Dict]:
    return _query(
        """
        SELECT tourney_date, year, tourney_name, tourney_level, round, surface, score,
               winner_id, loser_id, best_of
        FROM matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        ORDER BY tourney_date, match_id
        """,
        (int(player_a), int(player_b), int(player_b), int(player_a)),
    )


@lru_cache(maxsize=8)
def _reference_distribution(surface: str) -> Dict[str, List[float]]:
    """Sorted values of each serve/return stat over all seasons on a surface
    with match stats and at least PRIME_MIN_MATCHES matches."""
    rows = _query(
        f"""
        SELECT * FROM player_yearly_stats
        WHERE surface = ? AND serve_matches >= ? AND p_svpt > 0
        """,
        (surface, PRIME_MIN_MATCHES),
    )
    features = [compute_features(row) for row in rows]
    return {
        name: sorted(f[name] for f in features if f[name] == f[name])
        for name in SERVE_FEATURES
    }


def percentile(stat: str, value: float, surface: str) -> Optional[float]:
    """Share (0-100) of reference seasons on `surface` below `value`."""
    values = _reference_distribution(surface).get(stat) or []
    if not values or value is None or value != value:
        return None
    return 100.0 * bisect_left(values, value) / len(values)


# Data quirks in Grand Slam names (1968 "Australian Chps.", the extra
# December 1977 Australian Open, inconsistent "Us Open").
SLAM_NAMES = {
    "Australian Chps.": "Australian Open",
    "Australian Open-2": "Australian Open (Dec)",
    "Us Open": "US Open",
}
# Tennis was only a demonstration event at the 1984 Olympics.
FIRST_OLYMPIC_MEDAL_YEAR = 1988


@lru_cache(maxsize=4096)
def season_achievements(player_id: int, year: int, surface: str) -> Dict:
    """Titles and record vs top-10 opponents for one season on a surface.

    Titles count tour-level finals won (Grand Slams, Masters, Tour Finals,
    other tour events and the Olympics), not Davis Cup ties. Top 10 uses the
    opponent's ranking at the event; `top10` is None when no opponent had a
    ranking recorded (mostly before 1973).
    """
    pid = int(player_id)
    row = _query(
        """
        SELECT
          SUM(winner_id = :p AND round = 'F' AND tourney_level != 'D') AS titles,
          SUM(winner_id = :p AND loser_rank <= 10) AS top10_wins,
          SUM(loser_id = :p AND winner_rank <= 10) AS top10_losses,
          SUM(CASE WHEN winner_id = :p THEN loser_rank IS NOT NULL
                   ELSE winner_rank IS NOT NULL END) AS ranked_opponents
        FROM matches
        WHERE (winner_id = :p OR loser_id = :p) AND year = :y AND surface = :s
        """,
        {"p": pid, "y": int(year), "s": surface},
    )[0]
    top10 = None
    if row["ranked_opponents"]:
        top10 = (int(row["top10_wins"] or 0), int(row["top10_losses"] or 0))
    return {"titles": int(row["titles"] or 0), "top10": top10}


@lru_cache(maxsize=4096)
def year_honours(player_id: int, year: int) -> Dict:
    """Grand Slams won and Olympic medal in a calendar year (all surfaces)."""
    pid, yr = int(player_id), int(year)
    slams = [
        SLAM_NAMES.get(r["tourney_name"], r["tourney_name"])
        for r in _query(
            """
            SELECT tourney_name FROM matches
            WHERE winner_id = ? AND year = ? AND tourney_level = 'G' AND round = 'F'
            ORDER BY tourney_date
            """,
            (pid, yr),
        )
    ]

    medal = None
    if yr >= FIRST_OLYMPIC_MEDAL_YEAR:
        rows = _query(
            """
            SELECT tourney_id, round, winner_id FROM matches
            WHERE (winner_id = :p OR loser_id = :p) AND year = :y
              AND tourney_name LIKE '%Olympics' AND round IN ('F', 'BR', 'SF')
            """,
            {"p": pid, "y": yr},
        )
        by_round = {r["round"]: r for r in rows}
        if "F" in by_round:
            medal = "Gold" if by_round["F"]["winner_id"] == pid else "Silver"
        elif "BR" in by_round:
            medal = "Bronze" if by_round["BR"]["winner_id"] == pid else None
        elif "SF" in by_round and by_round["SF"]["winner_id"] != pid:
            # Before 1996 there was no bronze-medal match: both losing
            # semi-finalists received bronze.
            had_bronze_match = _query(
                "SELECT 1 FROM matches WHERE tourney_id = ? AND round = 'BR' LIMIT 1",
                (by_round["SF"]["tourney_id"],),
            )
            medal = None if had_bronze_match else "Bronze"
    return {"slams": slams, "olympic_medal": medal}
