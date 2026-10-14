import plotly.graph_objects as go

from tennis import charts, db

IDS = (104745, 103819)


def _named(pid, year, surface):
    return {**db.season(pid, year, surface), "display_name": f"{pid} {year}"}


def test_season_charts_have_one_trace_per_player():
    players = [_named(IDS[0], 2008, "Clay"), _named(IDS[1], 2006, "Clay")]
    pct = [{stat: db.percentile(stat, p.get(stat), "Clay") for stat, _ in charts.RADAR_STATS} for p in players]
    for fig in (charts.radar(players, pct, "Clay"), charts.comparison_bars(players)):
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


def test_career_surface_and_h2h_figures():
    careers = [db.career(pid) for pid in IDS]
    assert len(charts.career_lines(careers, ["A", "B"], "win_pct", [2008, 2006]).data) >= 2
    assert len(charts.surface_bars([db.surface_splits(pid) for pid in IDS], ["A", "B"]).data) == 2
    assert len(charts.surface_heatmap(db.year_surface_grid(IDS[0]), "A", 0).data) == 1
    assert len(charts.h2h_timeline(db.head_to_head(*IDS), IDS, ["A", "B"]).data) == 2


def test_no_h2h_gives_message():
    fig = charts.h2h_timeline([], IDS, ["A", "B"])
    assert fig.layout.annotations


def test_light_theme_changes_ink_not_player_colours():
    players = [_named(IDS[0], 2008, "Clay"), _named(IDS[1], 2006, "Clay")]
    dark = charts.comparison_bars(players)
    light = charts.comparison_bars(players, theme="light")
    assert dark.layout.font.color == charts.THEMES["dark"]["ink"]
    assert light.layout.font.color == charts.THEMES["light"]["ink"]
    assert [t.marker.color for t in dark.data] == [t.marker.color for t in light.data]
    heat = charts.surface_heatmap(db.year_surface_grid(IDS[0]), "A", 0, theme="light")
    assert heat.data[0].colorscale[0][1] == charts.THEMES["light"]["ramps"][0][0][1]
