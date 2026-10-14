import math

from tennis_features import MODEL_FEATURES, SERVE_FEATURES, build_model_row, compute_features

BASE = {
    "wins": 8, "losses": 2, "serve_matches": 10,
    "p_ace": 50, "p_df": 20, "p_svpt": 700, "p_1stIn": 420, "p_1stWon": 320, "p_2ndWon": 150,
    "p_SvGms": 110, "p_bpSaved": 30, "p_bpFaced": 45,
    "o_ace": 30, "o_df": 25, "o_svpt": 690, "o_1stWon": 280, "o_2ndWon": 130, "o_SvGms": 108,
    "o_bpSaved": 40, "o_bpFaced": 70,
    "rank_sum": 50, "rank_n": 10, "rank_points_sum": 30000, "rank_points_n": 10,
    "age_sum": 250, "age_n": 10, "hand": "R", "ht": 185,
}


def test_rates_are_computed_from_totals():
    f = compute_features(BASE)
    assert f["win_pct"] == 0.8
    assert f["rank"] == 5
    assert math.isclose(f["first_serve_pct"], 420 / 700)
    assert math.isclose(f["second_serve_win_pct"], 150 / 280)
    assert math.isclose(f["return_points_won_pct"], (690 - 410) / 690)
    assert f["has_serve_stats"] == 1.0


def test_missing_serve_stats_are_nan_not_zero():
    old = dict(BASE, serve_matches=0, p_svpt=0)
    f = compute_features(old)
    assert f["has_serve_stats"] == 0.0
    assert all(math.isnan(f[name]) for name in SERVE_FEATURES)


def test_missing_rank_counts_as_unranked():
    f = compute_features(dict(BASE, rank_sum=0, rank_n=0))
    assert f["rank"] is None
    assert math.isclose(f["log_rank"], math.log(2000))


def test_model_row_is_antisymmetric_in_differences():
    a = compute_features(BASE)
    b = compute_features(dict(BASE, wins=3, losses=7, rank_sum=300, hand="L"))
    ab, ba = build_model_row(a, b), build_model_row(b, a)
    assert len(ab) == len(MODEL_FEATURES)
    n_diff = sum(name.startswith("diff_") for name in MODEL_FEATURES)
    for x, y in zip(ab[:n_diff], ba[:n_diff]):
        assert (math.isnan(x) and math.isnan(y)) or math.isclose(x, -y, abs_tol=1e-12)
