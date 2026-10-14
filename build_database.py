"""Build atp_stats.db from the verified raw data snapshot in data/.

Tables:
  players              one row per player (id, name, display label, bio)
  matches              every valid tour-level match, with per-match stats
  player_yearly_stats  season totals per (player_id, year, surface), plus the
                       mean pre-match overall and surface Elo over those matches
  meta                 snapshot id, manifest hash, build time, git commit
"""

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tennis.etl import (
    aggregate_seasons,
    compute_elo,
    load_raw_matches,
    player_match_elo,
    season_elo,
    to_long,
    validate_matches,
)
from tennis_features import TOTAL_COLUMNS, fold_name
from tennis.snapshot import DATA_DIR, MANIFEST_PATH, manifest_hash, verify_snapshot

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "atp_stats.db"
TOTAL_COLUMNS_INT = [c for c in TOTAL_COLUMNS if c not in ("age_sum", "rank_points_sum")]

MATCH_TABLE_COLUMNS = [
    "match_id", "tourney_id", "tourney_name", "tourney_level", "tourney_date", "year",
    "surface", "round", "best_of", "score", "minutes",
    "winner_id", "loser_id",
    "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
    "winner_age", "loser_age",
    "winner_elo", "loser_elo", "winner_surface_elo", "loser_surface_elo",
] + [f"{side}_{col}" for side in ("w", "l") for col in
     ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]]


def build_players(matches: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Players appearing in `matches`, with a unique display label."""
    appearances = pd.concat(
        [
            matches[["tourney_date", "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc"]]
            .set_axis(["date", "player_id", "name", "hand", "ht", "ioc"], axis=1),
            matches[["tourney_date", "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc"]]
            .set_axis(["date", "player_id", "name", "hand", "ht", "ioc"], axis=1),
        ]
    ).sort_values("date")
    latest = appearances.groupby("player_id").agg(
        name=("name", "last"),
        hand=("hand", "last"),
        ht=("ht", "median"),
        ioc=("ioc", "last"),
        first_year=("date", lambda d: d.min().year),
        last_year=("date", lambda d: d.max().year),
        total_matches=("date", "size"),
    )

    bio = pd.read_csv(data_dir / "atp_players.csv", dtype={"dob": "string"}, low_memory=False).set_index("player_id")
    players = latest.join(bio[["hand", "dob", "ioc", "height"]], rsuffix="_bio")
    players["hand"] = players["hand_bio"].where(players["hand_bio"].isin(["L", "R"]), players["hand"])
    players["ht"] = players["height"].where(players["height"].notna(), players["ht"])
    players["ioc"] = players["ioc"].fillna(players["ioc_bio"])
    players["birth_year"] = pd.to_numeric(players["dob"].str[:4], errors="coerce")
    players = players.reset_index()

    duplicated = players["name"].duplicated(keep=False)
    detail = players["ioc"].fillna("?") + players["birth_year"].map(
        lambda y: f", b. {int(y)}" if pd.notna(y) else ""
    )
    players["label"] = np.where(duplicated, players["name"] + " (" + detail + ")", players["name"])
    still_dup = players["label"].duplicated(keep=False)
    span = players["first_year"].astype(str).where(
        players["first_year"] == players["last_year"],
        players["first_year"].astype(str) + "–" + players["last_year"].astype(str),
    )
    players.loc[still_dup, "label"] = players.loc[still_dup, "label"].str[:-1] + ", " + span[still_dup] + ")"
    still_dup = players["label"].duplicated(keep=False)
    players.loc[still_dup, "label"] = (
        players.loc[still_dup, "label"] + " #" + players.loc[still_dup, "player_id"].astype(str)
    )
    players["search_name"] = players["name"].map(fold_name)
    return players[
        ["player_id", "name", "label", "search_name", "hand", "ht", "ioc", "dob",
         "birth_year", "first_year", "last_year", "total_matches"]
    ]


def merge_duplicate_ids(matches: pd.DataFrame, players: pd.DataFrame):
    """Map ids that are clearly the same person (same name, country and birth
    date) onto one canonical id: the one with the most recent appearance."""
    known = players[players["dob"].notna()]
    groups = known.groupby(["name", "ioc", "dob"])["player_id"]
    aliases = {}
    for _, ids in groups:
        if len(ids) < 2:
            continue
        ordered = players.set_index("player_id").loc[ids.tolist()].sort_values("last_year")
        canonical = ordered.index[-1]
        aliases.update({int(pid): int(canonical) for pid in ordered.index[:-1]})
    if aliases:
        matches = matches.copy()
        matches["winner_id"] = matches["winner_id"].replace(aliases)
        matches["loser_id"] = matches["loser_id"].replace(aliases)
        same = matches["winner_id"] == matches["loser_id"]
        matches = matches[~same]
    return matches, aliases


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_database(db_path: Path = DEFAULT_DB_PATH, data_dir: Path = DATA_DIR) -> None:
    print("Verifying raw data snapshot...")
    manifest = verify_snapshot(data_dir)
    print(f"Snapshot {manifest['snapshot_id']} OK ({len(manifest['files'])} files).")

    raw = load_raw_matches(data_dir)
    matches, report = validate_matches(raw)
    print("Validation report:", json.dumps(report))

    matches, aliases = merge_duplicate_ids(matches, build_players(matches, data_dir))
    report["merged_duplicate_ids"] = len(aliases)
    print(f"Merged {len(aliases)} duplicate player ids: {aliases}")
    players = build_players(matches, data_dir)
    matches = matches.merge(compute_elo(matches), on="match_id")
    seasons = (
        aggregate_seasons(to_long(matches))
        .merge(players[["player_id", "hand", "ht"]], on="player_id", how="left")
        .merge(season_elo(player_match_elo(matches)), on=["player_id", "year", "surface"], how="left")
    )
    seasons["matches_played"] = seasons["wins"] + seasons["losses"]

    match_table = matches[MATCH_TABLE_COLUMNS].copy()
    match_table["tourney_date"] = match_table["tourney_date"].dt.strftime("%Y-%m-%d")
    # Store counts as INTEGER (nullable) rather than REAL to keep the file small.
    int_columns = [c for c in MATCH_TABLE_COLUMNS[10:] if not c.endswith(("_age", "_elo"))] + ["best_of"]
    match_table[int_columns] = match_table[int_columns].round().astype("Int64")
    seasons[TOTAL_COLUMNS_INT] = seasons[TOTAL_COLUMNS_INT].round().astype("int64")

    meta = {
        "snapshot_id": manifest["snapshot_id"],
        "manifest_sha256": manifest_hash(MANIFEST_PATH),
        "attribution": manifest["attribution"],
        "license": manifest["license"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "validation_report": json.dumps(report),
        "min_year": str(int(seasons["year"].min())),
        "max_year": str(int(seasons["year"].max())),
    }

    tmp_path = Path(f"{db_path}.tmp")
    tmp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp_path)
    try:
        players.to_sql("players", conn, index=False)
        match_table.to_sql("matches", conn, index=False)
        seasons.to_sql("player_yearly_stats", conn, index=False)
        pd.DataFrame(list(meta.items()), columns=["key", "value"]).to_sql("meta", conn, index=False)
        conn.executescript(
            """
            CREATE UNIQUE INDEX idx_players_id ON players(player_id);
            CREATE INDEX idx_matches_wl ON matches(winner_id, loser_id);
            CREATE INDEX idx_matches_lw ON matches(loser_id, winner_id);
            CREATE UNIQUE INDEX idx_seasons_key ON player_yearly_stats(player_id, year, surface);
            """
        )
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    tmp_path.replace(db_path)

    print(
        f"Built {db_path.name}: {len(players)} players, {len(match_table)} matches, "
        f"{len(seasons)} player-seasons ({db_path.stat().st_size / 1e6:.1f} MB)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    build_database(args.db_path)


if __name__ == "__main__":
    main()
