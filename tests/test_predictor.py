import math

import pytest

from tennis import db, predictor

NADAL, FEDERER, DJOKOVIC, MURRAY, LAVER = 104745, 103819, 104925, 104918, 100029

PAIRS = [
    (NADAL, 2008, FEDERER, 2006, "Clay"),
    (DJOKOVIC, 2015, MURRAY, 2016, "Hard"),
    (LAVER, 1969, FEDERER, 2006, "Grass"),
]


@pytest.mark.parametrize("a, ay, b, by, surface", PAIRS)
def test_swapping_players_flips_probability(a, ay, b, by, surface):
    sa, sb = db.season(a, ay, surface), db.season(b, by, surface)
    p_ab = predictor.win_probability(sa, sb)
    p_ba = predictor.win_probability(sb, sa)
    assert 0 < p_ab < 1
    assert math.isclose(p_ab + p_ba, 1.0, abs_tol=1e-6)


def test_identical_seasons_are_a_coin_flip():
    s = db.season(NADAL, 2010, "Clay")
    assert math.isclose(predictor.win_probability(s, s), 0.5, abs_tol=1e-6)


def test_model_card_matches_data_snapshot():
    assert predictor.model_card()["data_snapshot"] == db.meta()["snapshot_id"]


def test_dominant_season_beats_weaker_one_despite_lower_ranking():
    # Nadal 2013 on clay (39-2, avg rank ~5 after injury) vs Federer 2013 on
    # clay (12-5, avg rank ~3.5): the stronger season must be favoured.
    p = predictor.win_probability(db.season(NADAL, 2013, "Clay"), db.season(FEDERER, 2013, "Clay"))
    assert p > 0.7
