"""Plotly figure builders. Pure functions: data in, go.Figure out.

Player colours are fixed per slot (colour follows the player, never the
rank) and were checked for colour-blind separation on both the dark and the
light surface: P1 emerald #059669, P2 blue #3b82f6. Every chart also labels
players by name or legend, so identity never relies on colour alone.

Every builder takes `theme` ("dark" or "light"); only surfaces, text, grid
and the heatmap ramps change between themes.
"""

from typing import Dict, List, Optional, Sequence

import plotly.graph_objects as go

from tennis_features import SURFACES

P1_COLOR = "#059669"
P2_COLOR = "#3b82f6"
PLAYER_COLORS = (P1_COLOR, P2_COLOR)
PLAYER_SYMBOLS = ("circle", "diamond")
THEMES = {
    "dark": {
        "surface": "#0f172a",
        "ink": "#e2e8f0",
        "ink_muted": "#94a3b8",
        "grid": "#1e293b",
        # Sequential ramps (near the surface -> strongest hue) for heatmaps.
        "ramps": (
            [[0, "#1e293b"], [0.5, "#065f46"], [1, "#34d399"]],
            [[0, "#1e293b"], [0.5, "#1e40af"], [1, "#93c5fd"]],
        ),
    },
    "light": {
        "surface": "#ffffff",
        "ink": "#0f172a",
        "ink_muted": "#475569",
        "grid": "#e2e8f0",
        "ramps": (
            [[0, "#f1f5f9"], [0.5, "#6ee7b7"], [1, "#047857"]],
            [[0, "#f1f5f9"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
        ),
    },
}
DEFAULT_THEME = "dark"
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"


RADAR_STATS = [
    ("first_serve_pct", "1st serve in"),
    ("first_serve_win_pct", "1st serve won"),
    ("second_serve_win_pct", "2nd serve won"),
    ("ace_rate", "Ace rate"),
    ("bp_save_pct", "BP saved"),
    ("return_points_won_pct", "Return pts won"),
    ("bp_conversion_pct", "BP converted"),
]

COMPARISON_STATS = [
    ("win_pct", "Win %", "pct"),
    ("first_serve_pct", "1st serve in", "pct"),
    ("first_serve_win_pct", "1st serve won", "pct"),
    ("second_serve_win_pct", "2nd serve won", "pct"),
    ("service_points_won_pct", "Serve pts won", "pct"),
    ("bp_save_pct", "BP saved", "pct"),
    ("return_points_won_pct", "Return pts won", "pct"),
    ("bp_conversion_pct", "BP converted", "pct"),
    ("ace_rate", "Ace rate", "pct"),
    ("df_rate", "Double fault rate", "pct"),
]

CAREER_METRICS = {
    "win_pct": ("Win %", "pct"),
    "elo": ("Elo rating (year end)", "count"),
    "surface_elo": ("Surface Elo (pick a surface)", "count"),
    "rank": ("Average ranking", "rank"),
    "matches_played": ("Matches played", "count"),
    "first_serve_pct": ("1st serve in %", "pct"),
    "first_serve_win_pct": ("1st serve points won %", "pct"),
    "second_serve_win_pct": ("2nd serve points won %", "pct"),
    "return_points_won_pct": ("Return points won %", "pct"),
    "bp_save_pct": ("Break points saved %", "pct"),
    "ace_rate": ("Aces per service point %", "pct"),
}


def _valid(value) -> bool:
    return value is not None and value == value


def _theme(theme: Optional[str]) -> Dict:
    return THEMES.get(theme or DEFAULT_THEME, THEMES[DEFAULT_THEME])


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, theme: Optional[str], height: int = 380, **kwargs) -> go.Figure:
    t = _theme(theme)
    fig.update_layout(
        template="plotly_dark" if t is THEMES["dark"] else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=t["ink"], size=13),
        margin=dict(l=48, r=16, t=48, b=40),
        height=height,
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["grid"],
                        font=dict(color=t["ink"], family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=t["ink_muted"])),
        **kwargs,
    )
    axes = dict(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["grid"],
                tickfont=dict(color=t["ink_muted"]))
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    return fig


def empty_figure(message: str, height: int = 320, theme: Optional[str] = None) -> go.Figure:
    t = _theme(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=t["ink_muted"], size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, theme, height=height)


def radar(players: Sequence[Dict], percentiles: Sequence[Dict[str, Optional[float]]],
          surface: str, theme: Optional[str] = None) -> go.Figure:
    """Serve/return profile as percentiles against all ATP seasons on the surface."""
    t = _theme(theme)
    if not any(p.get("has_serve_stats") for p in players):
        return empty_figure("No match stats recorded for these seasons (mostly before 1991).",
                            theme=theme)
    labels = [label for _, label in RADAR_STATS]
    shown = [i for i, p in enumerate(players) if p.get("has_serve_stats")]

    # One tooltip per axis listing every player, so a point hidden under the
    # other player's marker can still be read.
    marks = ("●", "◆")
    hover = []
    for stat, label in RADAR_STATS:
        lines = [f"<b>{label}</b>"]
        for i in shown:
            value, pct = players[i].get(stat), percentiles[i].get(stat)
            value_text = f"{value * 100:.1f}%" if _valid(value) else "n/a"
            pct_text = f" · {pct:.0f}th percentile" if pct is not None else ""
            lines.append(f"{marks[i]} {players[i]['display_name']}: {value_text}{pct_text}")
        hover.append("<br>".join(lines))

    fig = go.Figure()
    # P2 is drawn first with a larger diamond, P1 on top with a
    # smaller circle: when values coincide both shapes stay visible.
    for i in sorted(shown, reverse=True):
        r = [percentiles[i].get(stat) or 0 for stat, _ in RADAR_STATS]
        fig.add_trace(
            go.Scatterpolar(
                r=r + r[:1],
                theta=labels + labels[:1],
                name=players[i]["display_name"],
                line=dict(color=PLAYER_COLORS[i], width=2, dash="solid" if i == 0 else "dot"),
                marker=dict(
                    size=8 if i == 0 else 14,
                    symbol=PLAYER_SYMBOLS[i],
                    color=PLAYER_COLORS[i],
                    line=dict(color=t["surface"] if i == 0 else PLAYER_COLORS[i], width=2),
                ),
                fill="toself",
                fillcolor=_rgba(PLAYER_COLORS[i], 0.15),
                customdata=hover + hover[:1],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], gridcolor=t["grid"],
                            linecolor=t["grid"], tickfont=dict(color=t["ink_muted"], size=10)),
            angularaxis=dict(gridcolor=t["grid"], linecolor=t["grid"],
                             tickfont=dict(color=t["ink"], size=12)),
        ),
    )
    title = f"Percentile vs all ATP {surface.lower()}-court seasons<br>with 10+ matches"
    fig = _layout(fig, theme, height=460,
                  title=dict(text=title, font=dict(size=13, color=t["ink_muted"]), x=0.5, xanchor="center"))
    fig.update_layout(margin=dict(l=72, r=72, t=56, b=56),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                                  traceorder="reversed"))
    return fig


def comparison_bars(players: Sequence[Dict], theme: Optional[str] = None) -> go.Figure:
    """Side-by-side season stats (grouped horizontal bars)."""
    t = _theme(theme)
    stats = [(key, label) for key, label, _ in COMPARISON_STATS
             if any(_valid(p.get(key)) for p in players)]
    labels = [label for _, label in stats][::-1]
    top = max((p.get(key) or 0) * 100 for p in players for key, _ in stats if _valid(p.get(key))) or 100
    fig = go.Figure()
    # Horizontal groups stack the first trace at the bottom, so add P2 first
    # and reverse the legend to keep P1 on top everywhere.
    for i, player in reversed(list(enumerate(players))):
        values = [player.get(key) for key, _ in stats][::-1]
        x = [v * 100 if _valid(v) else None for v in values]
        fig.add_trace(
            go.Bar(
                y=labels, x=x, orientation="h", name=player["display_name"],
                marker=dict(color=PLAYER_COLORS[i], line=dict(color=t["surface"], width=2),
                            cornerradius=4),
                text=[f"{v:.1f}%" if v is not None else "n/a" for v in x],
                # Inside the bar end when the bar is long enough, so labels don't run
                # off narrow screens; short bars keep the label outside.
                textposition=["inside" if v is not None and v >= 0.4 * top else "outside" for v in x],
                insidetextanchor="end", textangle=0,
                insidetextfont=dict(color="#ffffff", size=11),
                outsidetextfont=dict(color=t["ink_muted"], size=11),
                cliponaxis=False,
                hovertemplate="%{x:.1f}%",
            )
        )
    # Hovering a stat shows both players in one tooltip.
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, legend_traceorder="reversed",
                      hovermode="y unified")
    # Extra room past the longest bar for labels, without ticks beyond the data.
    fig.update_xaxes(ticksuffix="%", range=[0, top * 1.22],
                     tickvals=[t for t in (0, 25, 50, 75, 100) if t <= top * 1.05])
    return _layout(fig, theme, height=max(360, 52 * len(stats) + 80))


def career_lines(careers: Sequence[List[Dict]], names: Sequence[str], metric: str,
                 highlight_years: Sequence[Optional[int]], theme: Optional[str] = None) -> go.Figure:
    t = _theme(theme)
    label, kind = CAREER_METRICS[metric]
    fig = go.Figure()
    has_data = False
    for i, (rows, name) in enumerate(zip(careers, names)):
        points = [(r["year"], r.get(metric), r) for r in rows if _valid(r.get(metric))]
        if not points:
            continue
        has_data = True
        years = [p[0] for p in points]
        values = [p[1] * 100 if kind == "pct" else p[1] for p in points]
        records = [f"{int(p[2]['wins'])}–{int(p[2]['losses'])}" for p in points]
        value_format = ".1f" if kind == "pct" else ".0f"
        fig.add_trace(
            go.Scatter(
                x=years, y=values, name=name, mode="lines+markers",
                line=dict(color=PLAYER_COLORS[i], width=2),
                marker=dict(size=8, symbol=PLAYER_SYMBOLS[i], color=PLAYER_COLORS[i],
                            line=dict(color=t["surface"], width=2)),
                customdata=records,
                hovertemplate=(
                    "%{x}: %{y:" + value_format + "}" + ("%" if kind == "pct" else "")
                    + "<br>Record %{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        )
        year = highlight_years[i] if i < len(highlight_years) else None
        if year in years:
            j = years.index(year)
            fig.add_trace(
                go.Scatter(
                    x=[year], y=[values[j]], mode="markers+text", showlegend=False,
                    marker=dict(size=16, symbol="star", color=PLAYER_COLORS[i],
                                line=dict(color=t["ink"], width=1.5)),
                    text=[f"{name.split()[-1]} {year}"], textposition="top center",
                    textfont=dict(color=t["ink"], size=11), hoverinfo="skip",
                )
            )
    if not has_data:
        return empty_figure(f"No {label.lower()} data for these players.", theme=theme)
    if kind == "rank":
        fig.update_yaxes(autorange="reversed", type="log", title=label)
    else:
        fig.update_yaxes(title=label, ticksuffix="%" if kind == "pct" else "")
    fig.update_xaxes(dtick=2 if fig.data and len(fig.data[0].x) < 20 else 5)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, theme, height=420)


def surface_bars(splits: Sequence[List[Dict]], names: Sequence[str],
                 theme: Optional[str] = None) -> go.Figure:
    t = _theme(theme)
    fig = go.Figure()
    for i, (rows, name) in enumerate(zip(splits, names)):
        by_surface = {r["surface"]: r for r in rows}
        x = [s for s in SURFACES if s in by_surface]
        y = [by_surface[s]["win_pct"] * 100 for s in x]
        rec = [f"{int(by_surface[s]['wins'])}–{int(by_surface[s]['losses'])}" for s in x]
        fig.add_trace(
            go.Bar(
                x=x, y=y, name=name,
                marker=dict(color=PLAYER_COLORS[i], line=dict(color=t["surface"], width=2),
                            cornerradius=4),
                text=[f"{v:.0f}%" for v in y], textposition="outside",
                textfont=dict(color=t["ink_muted"], size=12), cliponaxis=False,
                customdata=rec,
                hovertemplate="%{x}: %{y:.1f}% (%{customdata})<extra>%{fullData.name}</extra>",
            )
        )
    if not fig.data:
        return empty_figure("No matches found.", theme=theme)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 105], ticksuffix="%", title="Win %")
    return _layout(fig, theme, height=380)


def surface_heatmap(grid: List[Dict], name: str, slot: int, theme: Optional[str] = None) -> go.Figure:
    """Year x surface win % for one player."""
    t = _theme(theme)
    if not grid:
        return empty_figure("No matches found.", theme=theme)
    years = sorted({r["year"] for r in grid})
    surfaces = [s for s in SURFACES if any(r["surface"] == s for r in grid)]
    cell = {(r["year"], r["surface"]): r for r in grid}
    z, text = [], []
    for surface in surfaces:
        z_row, t_row = [], []
        for year in years:
            r = cell.get((year, surface))
            if r and r["matches_played"]:
                z_row.append(100 * r["wins"] / r["matches_played"])
                t_row.append(f"{r['wins']}–{r['losses']}")
            else:
                z_row.append(None)
                t_row.append("")
        z.append(z_row)
        text.append(t_row)
    fig = go.Figure(
        go.Heatmap(
            x=years, y=surfaces, z=z, customdata=text, zmin=0, zmax=100,
            colorscale=t["ramps"][slot], xgap=2, ygap=2,
            colorbar=dict(title=dict(text="Win %", font=dict(color=t["ink_muted"])), ticksuffix="%",
                          tickfont=dict(color=t["ink_muted"]), thickness=10),
            hovertemplate="%{x} · %{y}<br>Win %{z:.0f}% (%{customdata})<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, dtick=2 if len(years) < 16 else 4)
    return _layout(fig, theme, height=60 * len(surfaces) + 110,
                   title=dict(text=name, font=dict(size=14, color=t["ink_muted"])))


def h2h_timeline(matches: List[Dict], ids: Sequence[int], names: Sequence[str],
                 theme: Optional[str] = None) -> go.Figure:
    t = _theme(theme)
    if not matches:
        return empty_figure("These players never met in a tour-level match.", theme=theme)
    fig = go.Figure()
    for i, (pid, name) in enumerate(zip(ids, names)):
        won = [m for m in matches if m["winner_id"] == pid]
        if not won:
            continue
        fig.add_trace(
            go.Scatter(
                x=[m["tourney_date"] for m in won],
                y=[m["surface"] for m in won],
                mode="markers", name=f"{name} won",
                marker=dict(size=14, symbol=PLAYER_SYMBOLS[i], color=PLAYER_COLORS[i],
                            line=dict(color=t["surface"], width=2)),
                customdata=[[m["tourney_name"], m["round"], m["score"] or ""] for m in won],
                hovertemplate="%{x|%b %Y} · %{customdata[0]} %{customdata[1]}<br>%{customdata[2]}"
                              "<extra>%{fullData.name}</extra>",
            )
        )
    present = [s for s in SURFACES if any(m["surface"] == s for m in matches)]
    fig.update_yaxes(categoryorder="array", categoryarray=present[::-1])
    return _layout(fig, theme, height=120 + 50 * len(present))
