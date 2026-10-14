import app as app_module


def client():
    return app_module.server.test_client()


def test_health():
    body = client().get("/health").get_json()
    assert body["model_loaded"] is True
    assert body["data_snapshot"]


def test_predict_api_is_symmetric():
    c = client()
    a = c.post("/api/predict", json={"player1_name": "Rafael Nadal", "player1_year": 2008,
                                     "player2_name": "Roger Federer", "player2_year": 2006,
                                     "surface": "clay"}).get_json()
    b = c.post("/api/predict", json={"player1_name": "Roger Federer", "player1_year": 2006,
                                     "player2_name": "Rafael Nadal", "player2_year": 2008,
                                     "surface": "Clay"}).get_json()
    assert abs(a["player1"]["win_probability"] - b["player2"]["win_probability"]) < 1e-4


def test_predict_api_rejects_bad_payloads():
    c = client()
    assert c.post("/api/predict", json=None).status_code == 400
    assert c.post("/api/predict", json=[1, 2]).status_code == 400
    assert c.post("/api/predict", json={"player1_name": "x", "player1_year": None,
                                        "player2_name": "y", "player2_year": 2000,
                                        "surface": "Clay"}).status_code == 400
    assert c.post("/api/predict", json={"player1_name": "Rafael Nadal", "player1_year": 2008,
                                        "player2_name": "Roger Federer", "player2_year": 2006,
                                        "surface": "Ice"}).status_code == 400
    assert c.post("/api/predict", json={"player1_name": "Rafael Nadal", "player1_year": 1990,
                                        "player2_name": "Roger Federer", "player2_year": 2006,
                                        "surface": "Clay"}).status_code == 404


def test_dash_callbacks_render():
    hero, tiles, radar, bars = app_module.update_prediction(104745, 2008, 103819, 2006, "Clay")
    assert len(tiles) == 2 and radar.data and bars.data
    tiles, timeline, table = app_module.update_h2h(104745, 103819)
    assert timeline.data and table is not None


def _predict(c, name1, name2="Roger Federer", url="/api/predict"):
    return c.post(url, json={"player1_name": name1, "player1_year": 2008,
                             "player2_name": name2, "player2_year": 2006, "surface": "Clay"})


def test_partial_name_is_not_resolved_to_someone_else():
    resp = _predict(client(), "Rafael Nadal", "Federer")
    assert resp.status_code == 404
    body = resp.get_json()
    assert "Roger Federer" in body["suggestions"]


def test_exact_name_resolves():
    resp = _predict(client(), "rafael nadal")
    assert resp.status_code == 200
    assert resp.get_json()["player1"]["name"] == "Rafael Nadal"


def test_accent_insensitive_exact_name_resolves():
    resp = _predict(client(), "Rafael Nadál")
    assert resp.status_code == 200
    assert resp.get_json()["player1"]["name"] == "Rafael Nadal"


def test_shared_name_asks_for_a_label():
    c = client()
    body = c.post("/api/predict", json={"player1_name": "Fred Hemmes", "player1_year": 1970,
                                        "player2_name": "Roger Federer", "player2_year": 2006,
                                        "surface": "Clay"})
    assert body.status_code == 409
    assert len(body.get_json()["candidates"]) == 2


def test_cors_preflight_on_predict():
    resp = client().options("/api/predict")
    assert resp.status_code == 204
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in resp.headers["Access-Control-Allow-Methods"]


def test_cors_headers_not_added_to_dash_routes():
    assert "Access-Control-Allow-Origin" not in client().get("/_dash-layout").headers


def test_predict_alias_behaves_like_api_predict():
    c = client()
    a = _predict(c, "Rafael Nadal").get_json()
    b = _predict(c, "Rafael Nadal", url="/predict").get_json()
    assert a == b


def test_players_endpoint():
    c = client()
    found = c.get("/api/players?q=nad").get_json()
    assert {"player_id", "name", "label"} <= set(found[0])
    assert any(p["name"] == "Rafael Nadal" for p in found)
    unfiltered = c.get("/api/players").get_json()
    assert 0 < len(unfiltered) <= 100
    assert c.get("/api/players?limit=abc").status_code == 400


def test_predict_returns_json_503_when_model_missing(monkeypatch, tmp_path):
    from tennis import predictor

    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "missing.onnx")
    predictor._session.cache_clear()
    try:
        resp = _predict(client(), "Rafael Nadal")
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "Model not available."
    finally:
        monkeypatch.undo()
        predictor._session.cache_clear()


def test_career_chart_with_only_one_player():
    fig = app_module.update_career(104745, None, 2008, None, "win_pct", "All")
    names = {trace.name for trace in fig.data if trace.showlegend is not False}
    assert names == {"Rafael Nadal"}
    fig = app_module.update_career(None, 103819, None, 2006, "win_pct", "Clay")
    assert fig.data[0].line.color == "#3b82f6"  # Player 2 keeps its colour


def test_new_player_starts_on_prime_season():
    from tennis import db

    federer_clay = {s["year"] for s in db.seasons_for(103819, "Clay")}
    prime = db.prime_season(103819, "Clay")
    assert 2008 in federer_clay and prime != 2008
    # Switching Nadal (2008) -> Federer: start on Federer's prime, not 2008.
    assert app_module.choose_season("player", 2008, 2008, federer_clay, 103819, "Clay") == (prime, prime)


def test_picked_year_survives_a_surface_round_trip():
    from tennis import db

    years = {s: {r["year"] for r in db.seasons_for(104745, s)} for s in ("Clay", "Carpet")}
    assert 2010 in years["Clay"] and 2010 not in years["Carpet"]
    # User picks Nadal 2010 on clay.
    value, remembered = app_module.choose_season("season", 2010, 2013, years["Clay"], 104745, "Clay")
    assert (value, remembered) == (2010, 2010)
    # Switch to carpet: no 2010 season there, so show the closest one...
    value, remembered = app_module.choose_season("surface", value, remembered, years["Carpet"], 104745, "Carpet")
    assert value == app_module.closest_season(years["Carpet"], 2010) and remembered == 2010
    # ...and switching back to clay restores 2010.
    value, remembered = app_module.choose_season("surface", value, remembered, years["Clay"], 104745, "Clay")
    assert (value, remembered) == (2010, 2010)


def test_initial_load_uses_default_matchup():
    from tennis import db

    for prefix, pid in (("p1", app_module.DEFAULT_P1), ("p2", app_module.DEFAULT_P2)):
        years = {r["year"] for r in db.seasons_for(pid, app_module.DEFAULT_SURFACE)}
        year = app_module.DEFAULT_YEARS[prefix]
        assert app_module.choose_season(None, None, year, years, pid, app_module.DEFAULT_SURFACE) == (year, year)
    assert (app_module.DEFAULT_YEARS, app_module.DEFAULT_SURFACE) == ({"p1": 2013, "p2": 2011}, "Clay")


def test_closest_season():
    assert app_module.closest_season({2005, 2009, 2012}, 2010) == 2009
    assert app_module.closest_season({2008, 2012}, 2010) == 2012  # tie -> later year
    assert app_module.closest_season(set(), 2010) is None


def test_theme_switch_reaches_every_chart_callback():
    from tennis import charts

    light_ink = charts.THEMES["light"]["ink"]
    _, _, radar, bars = app_module.update_prediction(104745, 2008, 103819, 2006, "Clay", False)
    career = app_module.update_career(104745, 103819, 2008, 2006, "win_pct", "All", False)
    surface_bars, heat1, heat2 = app_module.update_surfaces(104745, 103819, 2008, 2006, "career", False)
    _, timeline, _ = app_module.update_h2h(104745, 103819, False)
    for fig in (radar, bars, career, surface_bars, heat1, heat2, timeline):
        assert fig.layout.font.color == light_ink


def test_season_card_shows_titles_top10_and_honours():
    card = str(app_module.season_tiles(_season(104745, 2008, "Clay"), 0, 2008))
    assert "Clay titles" in card and "Vs top 10" in card
    assert "Roland Garros, Wimbledon" in card and "Olympic gold" in card
    assert "Elo" not in card
    quiet = str(app_module.honours_line(103819, 1999))
    assert "No Grand Slam titles or Olympic medals in 1999" in quiet


def _season(pid, year, surface):
    from tennis import db

    return db.season(pid, year, surface)


def test_vercel_entrypoint_is_wsgi():
    # Vercel loads `app` from app.py (app:app); it must be the Flask WSGI app.
    import flask

    assert isinstance(app_module.app, flask.Flask)
    assert app_module.app is app_module.server


def test_no_vercel_rewrite_hides_the_request_path():
    # A catch-all rewrite to another path makes Flask see that path instead of
    # the real one, so every URL falls through to Dash's page (see README).
    import json
    from pathlib import Path

    config = Path(app_module.__file__).with_name("vercel.json")
    if config.exists():
        assert not json.loads(config.read_text()).get("rewrites")
