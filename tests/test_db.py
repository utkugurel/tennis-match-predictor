import sqlite3

from tennis import db


def _scalar(sql):
    with sqlite3.connect(db.DB_PATH) as conn:
        return conn.execute(sql).fetchone()[0]


def test_only_known_surfaces():
    assert _scalar(
        "SELECT COUNT(*) FROM player_yearly_stats WHERE surface NOT IN ('Hard','Clay','Grass','Carpet')"
    ) == 0


def test_season_key_is_unique():
    assert _scalar(
        "SELECT COUNT(*) FROM (SELECT 1 FROM player_yearly_stats "
        "GROUP BY player_id, year, surface HAVING COUNT(*) > 1)"
    ) == 0


def test_no_walkovers():
    assert _scalar("SELECT COUNT(*) FROM matches WHERE TRIM(UPPER(score)) IN ('W/O','WALKOVER')") == 0


def test_player_labels_are_unique():
    assert _scalar("SELECT COUNT(*) - COUNT(DISTINCT label) FROM players") == 0


def test_search_ranks_famous_word_prefix_first():
    assert db.search_players("fed")[0]["label"] == "Roger Federer"
    assert db.search_players("x") == []


def test_season_queries():
    seasons = db.seasons_for(104745, "Clay")
    assert {s["year"] for s in seasons} >= {2005, 2010}
    assert db.prime_season(104745, "Clay") in {s["year"] for s in seasons}
    s = db.season(104745, 2006, "Clay")
    assert (s["wins"], s["losses"]) == (26, 0)


def test_head_to_head_matches_official_record():
    matches = db.head_to_head(104745, 103819)
    nadal_wins = sum(m["winner_id"] == 104745 for m in matches)
    assert (nadal_wins, len(matches) - nadal_wins) == (24, 16)


def test_pre_open_era_player_has_no_serve_stats():
    s = db.season(100029, 1969, "Grass")
    assert s["has_serve_stats"] == 0.0


def test_meta_records_snapshot():
    assert db.meta()["snapshot_id"] == "sackmann-atp-2024-12-30"


def test_elo_ratings_are_stored():
    s = db.season(104745, 2013, "Clay")
    assert s["surface_elo"] > 2400 and s["elo"] > 2300
    career = db.career(104745, "Clay")
    assert all("surface_elo" in row for row in career)


def test_served_elo_matches_mean_pre_match_elo_in_matches_table():
    # Nadal, 2013, clay: the app's season Elo must equal the mean of the
    # pre-match ratings the model was trained on.
    with sqlite3.connect(db.DB_PATH) as conn:
        elo, surface_elo = conn.execute(
            """
            SELECT AVG(CASE WHEN winner_id = 104745 THEN winner_elo ELSE loser_elo END),
                   AVG(CASE WHEN winner_id = 104745 THEN winner_surface_elo ELSE loser_surface_elo END)
            FROM matches
            WHERE (winner_id = 104745 OR loser_id = 104745) AND year = 2013 AND surface = 'Clay'
            """
        ).fetchone()
    s = db.season(104745, 2013, "Clay")
    assert s["elo"] == round(elo, 1)
    assert s["surface_elo"] == round(surface_elo, 1)


def test_search_treats_like_wildcards_literally():
    assert db.search_players("__") == []
    assert db.search_players("%%") == []
    assert db.search_players("n_dal") == []
    assert db.search_players("\\\\") == []


def _player_id(name):
    return _scalar(f"SELECT player_id FROM players WHERE name = '{name}'")


def test_season_titles_and_top10_record():
    nadal_2013_clay = db.season_achievements(104745, 2013, "Clay")
    assert nadal_2013_clay["titles"] == 6  # Sao Paulo, Acapulco, Barcelona, Madrid, Rome, RG
    wins, losses = nadal_2013_clay["top10"]
    assert wins > losses
    # No rankings recorded before 1973.
    assert db.season_achievements(100029, 1969, "Grass")["top10"] is None


def test_grand_slams_and_olympic_medals():
    assert db.year_honours(100029, 1969)["slams"] == [
        "Australian Open", "Roland Garros", "Wimbledon", "US Open"
    ]
    assert db.year_honours(104745, 2008) == {"slams": ["Roland Garros", "Wimbledon"], "olympic_medal": "Gold"}
    assert db.year_honours(103819, 2012)["olympic_medal"] == "Silver"
    assert db.year_honours(104925, 2008)["olympic_medal"] == "Bronze"  # won the bronze match
    # 1988: no bronze-medal match, both losing semi-finalists got bronze.
    assert db.year_honours(_player_id("Brad Gilbert"), 1988)["olympic_medal"] == "Bronze"
    assert db.year_honours(_player_id("Stefan Edberg"), 1988)["olympic_medal"] == "Bronze"
    assert db.year_honours(104745, 2013)["olympic_medal"] is None
