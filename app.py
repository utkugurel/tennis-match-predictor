"""Tennis Match Predictor: Dash app, served on Vercel through `app` (its Flask server)."""

import os
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from flask import jsonify, request

from tennis import charts, db, predictor
from tennis_features import SURFACES

# Opening matchup: Nadal 2013 vs Federer 2011 on clay.
DEFAULT_P1, DEFAULT_P2, DEFAULT_SURFACE = 104745, 103819, "Clay"
DEFAULT_YEARS = {"p1": 2013, "p2": 2011}
SMALL_SAMPLE = 10

# Vercel Analytics only works on Vercel (it posts to /_vercel/insights).
ANALYTICS = (
    '<script defer src="https://va.vercel-scripts.com/v1/script.js"></script>'
    if os.environ.get("VERCEL") else ""
)
FAVICON = (
    "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>"
    "<text y=%22.9em%22 font-size=%2290%22>🎾</text></svg>"
)

dash_app = Dash(
    __name__,
    title="Tennis Match Predictor",
    update_title=None,
    external_stylesheets=[
        # Plain Bootstrap 5.3: its colour modes (data-bs-theme) drive light/dark.
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
# Applies the saved theme before the first paint (no dark/light flash).
THEME_BOOT = (
    "<script>try{var t=localStorage.getItem('tmp-theme');"
    "if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-bs-theme',t);}}"
    "catch(e){}</script>"
)

dash_app.index_string = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  {{%metas%}}
  {THEME_BOOT}
  <title>{{%title%}}</title>
  <link rel="icon" href="{FAVICON}">
  {{%css%}}
  {ANALYTICS}
</head>
<body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""
# WSGI entry points. Vercel's Python runtime loads `app` from this module
# (app:app), so `app` must be the Flask server, not the Dash object.
server = dash_app.server
app = server


# ---------------------------------------------------------------- helpers

def _player_option(player_id: Optional[int]) -> List[Dict]:
    info = db.player(player_id) if player_id else None
    return [{"label": info["label"], "value": info["player_id"]}] if info else []


def _season_or_none(player_id, year, surface) -> Optional[Dict]:
    if not player_id or not year or not surface:
        return None
    return db.season(int(player_id), int(year), surface)


def _display_name(player_id: int, year: Optional[int] = None) -> str:
    info = db.player(player_id)
    name = info["name"] if info else "?"
    return f"{name} {year}" if year else name


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None and value == value else "n/a"


def _tile(label: str, value: str) -> html.Div:
    return html.Div([html.Div(label, className="tile-label"), html.Div(value, className="tile-value")],
                    className="tile")


def _theme(dark: Optional[bool]) -> str:
    return "light" if dark is False else "dark"


def _player_dot(slot: int) -> html.Span:
    return html.Span(className=f"dot dot-p{slot + 1}")


# ---------------------------------------------------------------- layout

def _player_picker(slot: int) -> dbc.Col:
    prefix = f"p{slot + 1}"
    default = DEFAULT_P1 if slot == 0 else DEFAULT_P2
    return dbc.Col(
        [
            html.Div([_player_dot(slot), f"Player {slot + 1}"], className="picker-title"),
            dcc.Dropdown(
                id=f"{prefix}-player", options=_player_option(default), value=default,
                placeholder="Type a player name…", clearable=False, className="mb-2",
            ),
            dcc.Dropdown(id=f"{prefix}-season", placeholder="Season", clearable=False),
            # The year this player was last explicitly given (default or a pick
            # from the season list). Surface switches fall back temporarily when
            # that year has no season on the new surface, without forgetting it.
            dcc.Store(id=f"{prefix}-preferred-year", data=DEFAULT_YEARS[prefix]),
        ],
        md=5,
    )


def _about() -> dbc.Accordion:
    card = predictor.model_card()
    meta = db.meta()
    grouped = card.get("metrics", {}).get("grouped_holdout", {})
    temporal_key = next((k for k in card.get("metrics", {}) if k.startswith("temporal")), None)
    temporal = card.get("metrics", {}).get(temporal_key, {}) if temporal_key else {}
    model_name = (card.get("model") or "model").replace("_", " ")
    return dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.P(
                        "Pick two players, a season for each and a surface. The model compares the two "
                        "seasons' stats on that surface (results, ranking, serve and return numbers) "
                        "and estimates who would win."
                    ),
                    html.P(
                        f"It is a {model_name}{' with isotonic calibration' if card.get('calibrated') else ''}"
                        f" trained on {card.get('training_matches', 0):,} real ATP matches. Player strength "
                        "comes from Elo ratings (overall and per surface), which react to results straight "
                        "away and account for who you beat. A season's rating is the average of the "
                        "player's ratings going into each of its matches, the same pre-match ratings "
                        "the model was trained on. ATP ranking is shown but not used: it reflects "
                        "the previous 12 months, so a player returning from injury would look weaker than "
                        "their season was. For training, each player's season stats exclude the whole "
                        "tournament being predicted, so the model never sees the result. Predictions are "
                        "symmetric: swapping the players flips the probability exactly."
                    ),
                    html.Ul(
                        [
                            html.Li(f"Held-out matches: {grouped.get('accuracy', 0):.1%} accuracy, "
                                    f"AUC {grouped.get('auc', 0):.3f}"),
                            html.Li(f"Seasons 2019+ (trained on earlier years only): "
                                    f"{temporal.get('accuracy', 0):.1%} accuracy, AUC {temporal.get('auc', 0):.3f}"),
                            html.Li("Serve and return stats exist mostly from 1991; earlier seasons rely on "
                                    "results and ranking."),
                        ]
                    ),
                    html.P(
                        f"Model v{card.get('model_version', '?')} · data snapshot "
                        f"{meta.get('snapshot_id', '?')} (seasons {meta.get('min_year')}–{meta.get('max_year')})",
                        className="text-muted small mb-0",
                    ),
                ],
                title="How it works",
            ),
        ],
        start_collapsed=True,
        className="mt-4",
    )


def _footer() -> html.Footer:
    return html.Footer(
        [
            html.P(["Created by ", html.A("Utku Gürel", href="https://github.com/utkugurel", target="_blank")]),
            html.P(
                [
                    "Data: ATP results and stats by ",
                    html.A("Jeff Sackmann / Tennis Abstract", href="http://www.tennisabstract.com/",
                           target="_blank"),
                    " (snapshot as of Dec 2024; the original repository is no longer online). "
                    "Built with Plotly Dash, scikit-learn and ONNX.",
                ]
            ),
            html.P(
                [
                    "For educational and research purposes only. Licensed under ",
                    html.A("CC BY-NC-SA 4.0", href="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                           target="_blank"),
                    ".",
                ],
                className="small",
            ),
        ],
        className="site-footer",
    )


def _graph(graph_id: str) -> dcc.Graph:
    return dcc.Graph(id=graph_id, config={"displayModeBar": False})


surface_options = [{"label": s, "value": s} for s in SURFACES]

dash_app.layout = dbc.Container(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.Span("☀", **{"aria-hidden": "true"}),
                        # Remembered per browser by the theme clientside callback;
                        # THEME_BOOT applies the saved choice before first paint.
                        dbc.Switch(
                            id="theme-switch", value=True, label="Dark mode",
                            label_class_name="visually-hidden",
                        ),
                        html.Span("☾", **{"aria-hidden": "true"}),
                    ],
                    className="theme-toggle",
                    title="Toggle light/dark mode",
                ),
                html.H1("Tennis Match Predictor", className="title"),
                html.P(
                    "Compare any two ATP seasons on any surface: win probability, serve and return "
                    "profiles, career arcs and head-to-head history.",
                    className="subtitle",
                ),
            ],
            className="site-header text-center my-4",
        ),
        dcc.Store(id="theme"),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        _player_picker(0),
                        dbc.Col(
                            [
                                html.Div("Surface", className="picker-title justify-content-md-center"),
                                dbc.RadioItems(
                                    id="surface", options=surface_options, value=DEFAULT_SURFACE,
                                    className="btn-group surface-group", inputClassName="btn-check",
                                    labelClassName="btn btn-outline-secondary btn-sm",
                                    labelCheckedClassName="active",
                                ),
                            ],
                            md=2, className="d-flex flex-column align-items-md-center my-3 my-md-0",
                        ),
                        _player_picker(1),
                    ],
                    className="g-3",
                )
            ),
            className="picker-card",
        ),
        dcc.Loading(html.Div(id="hero", className="mt-4"), type="dot", color=charts.P1_COLOR),
        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        dbc.Row(id="season-tiles", className="g-3 mt-1"),
                        dbc.Row(
                            [
                                dbc.Col(_graph("radar"), lg=5),
                                dbc.Col(_graph("comparison"), lg=7),
                            ],
                            className="g-3 mt-1",
                        ),
                    ],
                    label="Season comparison", tab_id="tab-season",
                ),
                dbc.Tab(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="career-metric",
                                        options=[{"label": label, "value": key}
                                                 for key, (label, _) in charts.CAREER_METRICS.items()],
                                        value="win_pct", clearable=False,
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.RadioItems(
                                        id="career-surface",
                                        options=[{"label": "All surfaces", "value": "All"}] + surface_options,
                                        value="All", inline=True,
                                    ),
                                    md=8, className="d-flex align-items-center",
                                ),
                            ],
                            className="g-3 mt-2",
                        ),
                        _graph("career"),
                    ],
                    label="Career trends", tab_id="tab-career",
                ),
                dbc.Tab(
                    [
                        dbc.RadioItems(
                            id="surface-scope",
                            options=[{"label": "Whole career", "value": "career"},
                                     {"label": "Selected seasons", "value": "season"}],
                            value="career", inline=True, className="mt-3",
                        ),
                        _graph("surface-bars"),
                        dbc.Row(
                            [dbc.Col(_graph("heatmap-p1"), lg=6), dbc.Col(_graph("heatmap-p2"), lg=6)],
                            className="g-3",
                        ),
                    ],
                    label="Surfaces", tab_id="tab-surfaces",
                ),
                dbc.Tab(
                    [
                        dbc.Row(id="h2h-tiles", className="g-3 mt-1"),
                        _graph("h2h-timeline"),
                        html.Div(id="h2h-table", className="table-wrap"),
                    ],
                    label="Head-to-head", tab_id="tab-h2h",
                ),
            ],
            id="tabs", active_tab="tab-season", className="mt-4",
        ),
        _about(),
        _footer(),
    ],
    fluid="lg",
    className="pb-4",
)


# ---------------------------------------------------------------- callbacks

def closest_season(years, target: Optional[int]) -> Optional[int]:
    """The season nearest to `target` (ties go to the later year)."""
    if not years:
        return None
    if target is None:
        return max(years)
    return min(years, key=lambda y: (abs(y - target), -y))


def choose_season(trigger: Optional[str], current: Optional[int], preferred: Optional[int],
                  years, player_id: int, surface: str):
    """(season to show, year to remember) after a picker event.

    - "season": the user picked a year, so remember it.
    - "player": a new player starts on their prime season on this surface.
    - "surface" or the initial load: show the remembered year if the player
      has a season on this surface, otherwise the closest one, but keep
      remembering the original year so switching back restores it.
    """
    if trigger == "season" and current in years:
        return current, current
    if trigger == "player":
        prime = db.prime_season(player_id, surface)
        return prime, prime
    if preferred in years:
        return preferred, preferred
    return closest_season(years, preferred), preferred


def _register_player_search(prefix: str):
    @dash_app.callback(
        Output(f"{prefix}-player", "options"),
        Input(f"{prefix}-player", "search_value"),
        State(f"{prefix}-player", "value"),
    )
    def search(text, current):
        if not text:
            raise PreventUpdate
        options = db.search_players(text)
        if current and all(o["value"] != current for o in options):
            options = _player_option(current) + options
        return options

    @dash_app.callback(
        Output(f"{prefix}-season", "options"),
        Output(f"{prefix}-season", "value"),
        Output(f"{prefix}-preferred-year", "data"),
        Input(f"{prefix}-player", "value"),
        Input("surface", "value"),
        Input(f"{prefix}-season", "value"),
        State(f"{prefix}-preferred-year", "data"),
    )
    def seasons(player_id, surface, current, preferred):
        if not player_id:
            return [], None, preferred
        rows = db.seasons_for(int(player_id), surface)
        options = [
            {"label": f"{r['year']}  ·  {r['wins']}–{r['losses']}", "value": r["year"]} for r in rows
        ]
        years = {r["year"] for r in rows}
        trigger = {
            f"{prefix}-player": "player",
            f"{prefix}-season": "season",
            "surface": "surface",
        }.get(ctx.triggered_id)
        value, remembered = choose_season(trigger, current, preferred, years, int(player_id), surface)
        return options, value, remembered


_register_player_search("p1")
_register_player_search("p2")


MEDAL_ICONS = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉"}


def honours_line(player_id: int, year: int) -> html.Div:
    """Grand Slams and Olympic medal won that calendar year (all surfaces)."""
    honours = db.year_honours(player_id, year)
    items = []
    if honours["slams"]:
        items.append(html.Span(["🏆 ", "Grand Slams: ", html.B(", ".join(honours["slams"]))]))
    if honours["olympic_medal"]:
        medal = honours["olympic_medal"]
        items.append(html.Span([f"{MEDAL_ICONS[medal]} ", html.B(f"Olympic {medal.lower()}")]))
    if not items:
        return html.Div(f"No Grand Slam titles or Olympic medals in {year}.", className="honours muted")
    return html.Div(
        [
            html.Div(f"{year} honours (all surfaces)", className="tile-label"),
            html.Div(items, className="honours-items"),
        ],
        className="honours",
    )


def season_tiles(stats: Dict, slot: int, year: int) -> dbc.Col:
    info = db.player(stats["player_id"])
    rank = stats.get("rank")
    achievements = db.season_achievements(stats["player_id"], year, stats["surface"])
    top10 = achievements["top10"]
    tiles = [
        _tile("Record", f"{int(stats['wins'])}–{int(stats['losses'])}"),
        _tile("Win %", _fmt_pct(stats["win_pct"])),
        _tile(f"{stats['surface']} titles", str(achievements["titles"])),
        _tile("Vs top 10", f"{top10[0]}–{top10[1]}" if top10 else "n/a"),
        _tile("Avg rank", f"{rank:.0f}" if rank else "n/a"),
        _tile("Age", f"{stats['age']:.1f}" if stats["age"] == stats["age"] else "n/a"),
        _tile("Hand", {"L": "Left", "R": "Right"}.get(stats.get("hand"), "n/a")),
        _tile("Height", f"{info['ht']:.0f} cm" if info and info.get("ht") else "n/a"),
    ]
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div([_player_dot(slot), html.Span(info["label"] if info else "?"),
                              html.Span(f"\u00a0· {year} · {stats['surface']}", className="text-muted ms-1")],
                             className="tile-heading"),
                    html.Div(tiles, className="tile-grid"),
                    honours_line(stats["player_id"], year),
                ]
            ),
            className=f"player-card p{slot + 1}",
        ),
        md=6,
    )


def hero(p1: Dict, p2: Dict, y1: int, y2: int, probability: float) -> dbc.Card:
    p1_pct, p2_pct = probability * 100, (1 - probability) * 100
    names = [db.player(p1["player_id"])["name"], db.player(p2["player_id"])["name"]]
    notes = []
    for stats, name, year in ((p1, names[0], y1), (p2, names[1], y2)):
        if stats["matches_played"] < SMALL_SAMPLE:
            notes.append(f"{name} played only {int(stats['matches_played'])} matches on "
                         f"{stats['surface'].lower()} in {year}, so this estimate is uncertain.")
        if not stats["has_serve_stats"]:
            notes.append(f"No serve/return stats are recorded for {name} in {year}; the estimate "
                         "relies on results and ranking.")

    def side(slot, name, year, pct):
        return html.Div(
            [
                html.Div([_player_dot(slot), f"{name}"], className="hero-name"),
                html.Div(f"{year} · {p1['surface']}", className="hero-sub"),
                html.Div(f"{pct:.1f}%", className="hero-pct"),
            ],
            className=f"hero-side {'text-end' if slot else ''}",
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [side(0, names[0], y1, p1_pct), html.Div("vs", className="hero-vs"),
                     side(1, names[1], y2, p2_pct)],
                    className="hero-row",
                ),
                dbc.Progress(
                    [
                        dbc.Progress(value=p1_pct, bar=True, className="bar-p1"),
                        dbc.Progress(value=p2_pct, bar=True, className="bar-p2"),
                    ],
                    className="hero-bar",
                ),
                html.Div("Win probability", className="hero-caption"),
                *[dbc.Alert(note, color="warning", className="py-2 px-3 mt-3 mb-0 small") for note in notes],
            ]
        ),
        className="hero-card",
    )


@dash_app.callback(
    Output("hero", "children"),
    Output("season-tiles", "children"),
    Output("radar", "figure"),
    Output("comparison", "figure"),
    Input("p1-player", "value"),
    Input("p1-season", "value"),
    Input("p2-player", "value"),
    Input("p2-season", "value"),
    Input("surface", "value"),
    Input("theme-switch", "value"),
)
def update_prediction(p1_id, y1, p2_id, y2, surface, dark=True):
    theme = _theme(dark)
    p1 = _season_or_none(p1_id, y1, surface)
    p2 = _season_or_none(p2_id, y2, surface)
    if p1 is None or p2 is None:
        missing = [_display_name(pid) for pid, s in ((p1_id, p1), (p2_id, p2)) if pid and s is None]
        message = (
            f"No {surface.lower()}-court seasons found for {', '.join(missing)}."
            if missing else "Choose two players and seasons."
        )
        empty = charts.empty_figure(message, theme=theme)
        return dbc.Alert(message, color="secondary"), [], empty, empty

    probability = predictor.win_probability(p1, p2)
    named = [
        {**p1, "display_name": _display_name(p1_id, y1)},
        {**p2, "display_name": _display_name(p2_id, y2)},
    ]
    percentiles = [
        {stat: db.percentile(stat, s.get(stat), surface) for stat, _ in charts.RADAR_STATS}
        for s in (p1, p2)
    ]
    return (
        hero(p1, p2, y1, y2, probability),
        [season_tiles(p1, 0, y1), season_tiles(p2, 1, y2)],
        charts.radar(named, percentiles, surface, theme=theme),
        charts.comparison_bars(named, theme=theme),
    )


@dash_app.callback(
    Output("career", "figure"),
    Input("p1-player", "value"),
    Input("p2-player", "value"),
    Input("p1-season", "value"),
    Input("p2-season", "value"),
    Input("career-metric", "value"),
    Input("career-surface", "value"),
    Input("theme-switch", "value"),
)
def update_career(p1_id, p2_id, y1, y2, metric, surface, dark=True):
    theme = _theme(dark)
    ids = [pid for pid in (p1_id, p2_id) if pid]
    if not ids:
        return charts.empty_figure("Choose players.", theme=theme)
    surface = None if surface == "All" else surface
    # Keep both slots (colours follow the slot); an empty slot draws nothing.
    careers = [db.career(int(pid), surface) if pid else [] for pid in (p1_id, p2_id)]
    names = [_display_name(pid) if pid else "" for pid in (p1_id, p2_id)]
    return charts.career_lines(careers, names, metric, [y1, y2], theme=theme)


@dash_app.callback(
    Output("surface-bars", "figure"),
    Output("heatmap-p1", "figure"),
    Output("heatmap-p2", "figure"),
    Input("p1-player", "value"),
    Input("p2-player", "value"),
    Input("p1-season", "value"),
    Input("p2-season", "value"),
    Input("surface-scope", "value"),
    Input("theme-switch", "value"),
)
def update_surfaces(p1_id, p2_id, y1, y2, scope, dark=True):
    theme = _theme(dark)
    if not p1_id or not p2_id:
        empty = charts.empty_figure("Choose players.", theme=theme)
        return empty, empty, empty
    years = (y1, y2) if scope == "season" else (None, None)
    splits = [db.surface_splits(int(pid), year) for pid, year in zip((p1_id, p2_id), years)]
    names = [_display_name(pid, year) for pid, year in zip((p1_id, p2_id), years)]
    heatmaps = [
        charts.surface_heatmap(db.year_surface_grid(int(pid)), _display_name(pid), slot, theme=theme)
        for slot, pid in enumerate((p1_id, p2_id))
    ]
    return charts.surface_bars(splits, names, theme=theme), *heatmaps


def h2h_table(matches: List[Dict]) -> html.Div:
    if not matches:
        return html.Div()
    header = html.Thead(html.Tr([html.Th(h) for h in ("Date", "Tournament", "Round", "Surface", "Winner", "Score")]))
    rows = [
        html.Tr(
            [
                html.Td(m["tourney_date"]),
                html.Td(m["tourney_name"]),
                html.Td(m["round"]),
                html.Td(m["surface"]),
                html.Td([_player_dot(m["slot"]), m["winner_name"]]),
                html.Td(m["score"] or ""),
            ]
        )
        for m in reversed(matches)
    ]
    return dbc.Table([header, html.Tbody(rows)], size="sm", hover=True, responsive=True,
                     className="h2h-table mt-2")


@dash_app.callback(
    Output("h2h-tiles", "children"),
    Output("h2h-timeline", "figure"),
    Output("h2h-table", "children"),
    Input("p1-player", "value"),
    Input("p2-player", "value"),
    Input("theme-switch", "value"),
)
def update_h2h(p1_id, p2_id, dark=True):
    theme = _theme(dark)
    if not p1_id or not p2_id:
        return [], charts.empty_figure("Choose players.", theme=theme), None
    if p1_id == p2_id:
        message = "Pick two different players to see their head-to-head."
        return [], charts.empty_figure(message, theme=theme), None
    ids = (int(p1_id), int(p2_id))
    names = [_display_name(pid) for pid in ids]
    matches = [
        {**m, "slot": 0 if m["winner_id"] == ids[0] else 1,
         "winner_name": names[0] if m["winner_id"] == ids[0] else names[1]}
        for m in db.head_to_head(*ids)
    ]
    wins = [sum(1 for m in matches if m["slot"] == slot) for slot in (0, 1)]
    by_surface = []
    for surface in SURFACES:
        on = [m for m in matches if m["surface"] == surface]
        if on:
            w = sum(1 for m in on if m["slot"] == 0)
            by_surface.append(_tile(surface, f"{w}–{len(on) - w}"))
    tiles = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Head-to-head", className="tile-label"),
                        html.Div(
                            [_player_dot(0), f"{names[0]} {wins[0]} – {wins[1]} {names[1]}", _player_dot(1)],
                            className="h2h-score",
                        ),
                    ]
                )
            ),
            md=6,
        ),
        dbc.Col(dbc.Card(dbc.CardBody(html.Div(by_surface or [_tile("Matches", "0")], className="tile-grid"))),
                md=6),
    ]
    return tiles, charts.h2h_timeline(matches, ids, names, theme=theme), h2h_table(matches)


# Theme handling runs in the browser. On page load it restores the saved
# choice into the switch (which re-renders the charts in that theme); on a
# click it applies and saves the new one.
dash_app.clientside_callback(
    """
    function(dark) {
        var triggered = (dash_clientside.callback_context.triggered || [])
            .filter(function(t) { return t.prop_id && t.prop_id !== "."; });
        var mode;
        if (triggered.length === 0) {
            var saved = null;
            try { saved = localStorage.getItem("tmp-theme"); } catch (e) {}
            mode = saved === "light" ? "light" : "dark";
        } else {
            mode = dark === false ? "light" : "dark";
        }
        document.documentElement.setAttribute("data-bs-theme", mode);
        try { localStorage.setItem("tmp-theme", mode); } catch (e) {}
        var switchValue = mode === "dark";
        return [switchValue === dark ? dash_clientside.no_update : switchValue, mode];
    }
    """,
    Output("theme-switch", "value"),
    Output("theme", "data"),
    Input("theme-switch", "value"),
)


# ---------------------------------------------------------------- JSON endpoints

API_PATHS = ("/api/", "/predict")
PLAYERS_DEFAULT_LIMIT = 100
PLAYERS_MAX_LIMIT = 200


@server.after_request
def add_cors_headers(response):
    # Only the JSON API is meant for other origins, not Dash's own routes.
    if request.path.startswith(API_PATHS):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@server.route("/api/players", methods=["GET"])
def api_players():
    """Players as [{player_id, name, label}]. `?q=` filters by name; without it
    the most active players are returned (capped, not the whole table)."""
    query = request.args.get("q", "").strip()
    try:
        limit = min(max(int(request.args.get("limit", PLAYERS_DEFAULT_LIMIT)), 1), PLAYERS_MAX_LIMIT)
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400
    options = db.search_players(query, limit=limit) if query else db.top_players(limit)
    return jsonify(
        [{"player_id": o["value"], "name": db.player(o["value"])["name"], "label": o["label"]}
         for o in options]
    )

@server.route("/health")
def health():
    card = predictor.model_card()
    return jsonify(
        {
            **predictor.model_status(),
            "model_version": card.get("model_version"),
            "data_snapshot": db.meta().get("snapshot_id"),
        }
    )


def _resolve_player(name: str):
    """Return (player_id, None) for an exact name/label match, else
    (None, error response). Never guesses a different player."""
    found = db.players_by_exact_name(name)
    if len(found) == 1:
        return found[0]["player_id"], None
    if found:
        labels = [p["label"] for p in found]
        return None, (jsonify({"error": f"Several players are called '{name}'. Use one of the labels.",
                               "candidates": labels}), 409)
    suggestions = [o["label"] for o in db.search_players(name, limit=5)]
    return None, (jsonify({"error": f"Unknown player '{name}'.", "suggestions": suggestions}), 404)


@server.route("/api/predict", methods=["POST", "OPTIONS"])
@server.route("/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return "", 204
    status = predictor.model_status()
    if not status["model_loaded"]:
        return jsonify({"error": "Model not available.", "details": status["error"]}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    try:
        surface = str(payload["surface"]).strip().title()
        players = []
        for slot in (1, 2):
            name = str(payload[f"player{slot}_name"])
            year = int(payload[f"player{slot}_year"])
            players.append((name, year))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Required: player1_name, player1_year, player2_name, "
                                 "player2_year (integers), surface."}), 400
    if surface not in SURFACES:
        return jsonify({"error": f"surface must be one of {', '.join(SURFACES)}."}), 400

    stats = []
    for name, year in players:
        player_id, error = _resolve_player(name)
        if error:
            return error
        season = db.season(player_id, year, surface)
        if season is None:
            return jsonify({"error": f"No {surface} stats for {name} in {year}."}), 404
        stats.append(season)

    try:
        probability = predictor.win_probability(stats[0], stats[1])
    except Exception as exc:  # onnxruntime failures should still return JSON
        server.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500

    def summary(s, p):
        return {
            "name": db.player(s["player_id"])["name"],
            "year": int(s["year"]),
            "surface": surface,
            "win_probability": round(p, 4),
            "season_summary": {
                "wins": int(s["wins"]), "losses": int(s["losses"]),
                "win_pct": round(s["win_pct"], 4), "matches_played": int(s["matches_played"]),
                "rank": round(s["rank"], 1) if s["rank"] else None,
            },
        }

    return jsonify(
        {
            "surface": surface,
            "player1": summary(stats[0], probability),
            "player2": summary(stats[1], 1 - probability),
            "model_version": predictor.model_card().get("model_version"),
        }
    )


if __name__ == "__main__":
    dash_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=bool(os.environ.get("DEBUG")))
