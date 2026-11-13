"""Visual helpers for notebooks 12_GroupBy and 13_Mini_Project.

    from course_viz.grouping import (groupby_steps, compare_three, styled_agg,
        bar_labeled, weekday_bars, month_line, weekday_month_heatmap,
        pivot_heatmap, roadmap, dashboard, chart_makeover, explore_question,
        progress_checklist)
"""

from __future__ import annotations

import html as _html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from course_viz import GROUPS, _show, _table

__all__ = [
    "groupby_steps", "compare_three", "styled_agg", "bar_labeled", "weekday_bars",
    "month_line", "weekday_month_heatmap", "pivot_heatmap", "roadmap", "dashboard",
    "chart_makeover", "explore_question", "progress_checklist", "WEEKDAYS",
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# One consistent palette (Okabe-Ito): base bars, highlight, accent.
BASE, HI, ACC = "#56b4e9", "#d55e00", "#0072b2"


def _nice(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".") if isinstance(x, (float, np.floating)) else f"{x:,}"


def _head(n: int, text: str) -> str:
    return (f'<div class="cv-cap" style="font-size:15px;margin-top:10px">'
            f'<span class="cv-key" style="background:rgba(128,128,128,.3)">STEP {n}</span>{_html.escape(text)}</div>')


# --------------------------------------------------------------------------- #
# Step-by-step group-by with the arithmetic written out
# --------------------------------------------------------------------------- #
def groupby_steps(df: pd.DataFrame, key: str, value: str, agg: str = "sum") -> None:
    """Show a SMALL group-by in four stacked stages, arithmetic included."""
    keys = list(dict.fromkeys(df[key]))
    pal = {g: GROUPS[i % len(GROUPS)] for i, g in enumerate(keys)}
    raw = df[[key, value]].reset_index(drop=True)

    def paint(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for i in frame.index:
            out.loc[i, :] = f"background:{pal[frame.at[i, key]]}"
        return out

    srt = raw.sort_values(key, kind="stable")
    lines, result = [], {}
    for g in keys:
        vals = raw.loc[raw[key] == g, value].tolist()
        s = ", ".join(_nice(v) for v in vals)
        if agg == "sum":
            res = sum(vals); eq = " + ".join(_nice(v) for v in vals) + f" = <b>{_nice(res)}</b>"
        elif agg == "mean":
            res = float(np.mean(vals))
            eq = f"({' + '.join(_nice(v) for v in vals)}) / {len(vals)} = <b>{_nice(res)}</b>"
        elif agg == "max":
            res = max(vals); eq = f"biggest of [{s}] = <b>{_nice(res)}</b>"
        elif agg == "min":
            res = min(vals); eq = f"smallest of [{s}] = <b>{_nice(res)}</b>"
        elif agg == "median":
            res = float(np.median(vals)); eq = f"middle of sorted [{', '.join(_nice(v) for v in sorted(vals))}] = <b>{_nice(res)}</b>"
        else:  # size / count
            res = len(vals); eq = f"{' + '.join('1' for _ in vals)} = <b>{res}</b> rows"
        result[g] = res
        lines.append(f'<div style="margin:3px 0"><span class="cv-key" style="background:{pal[g]}">'
                     f'{_html.escape(str(g))}</span> {eq}</div>')
    res_df = pd.Series(result, name=f"{agg}({value})").rename_axis(key).to_frame()
    res_sty = res_df.style.apply(
        lambda f: pd.DataFrame({c: [f"background:{pal[g]}; font-weight:bold" for g in f.index] for c in f.columns},
                               index=f.index), axis=None)
    _show(
        _head(1, f"Raw rows (as they arrive, mixed up) , {len(raw)} rows")
        + f'<div>{_table(raw.style)}</div>'
        + _head(2, f"SPLIT: sort and colour the rows by '{key}', {len(keys)} groups")
        + f'<div>{_table(srt.style.apply(paint, axis=None))}</div>'
        + _head(3, f"APPLY: do {agg}('{value}') inside each group, one at a time")
        + "".join(lines)
        + _head(4, "COMBINE: one row per group, the same colours")
        + f'<div>{_table(res_sty)}</div>')


# --------------------------------------------------------------------------- #
# groupby vs value_counts vs boolean mean
# --------------------------------------------------------------------------- #
def compare_three(df: pd.DataFrame, key: str, mask: pd.Series, mask_label: str = "flag") -> None:
    """Three panes: counts built two ways, then a share built the 'boolean mean' way."""
    a = df.groupby(key).size().rename("rows")
    b = df[key].value_counts().sort_index().rename("rows")
    c = (mask.groupby(df[key]).mean() * 100).round(1).rename(f"% {mask_label}")
    big = a.to_frame().style.bar(color="rgba(86,180,233,.6)", vmin=0)
    big2 = b.to_frame().style.bar(color="rgba(86,180,233,.6)", vmin=0)
    big3 = c.to_frame().style.bar(color="rgba(213,94,0,.5)", vmin=0)

    def pane(title, code, sty, note):
        return (f'<div class="cv-pane"><div class="cv-cap">{title}</div>'
                f'<code>{_html.escape(code)}</code>{_table(sty)}<div style="max-width:260px;font-size:12px">{note}</div></div>')

    _show(
        pane("A. groupby + size", f"df.groupby('{key}').size()", big.format(precision=0), "Counts rows per group. Order: alphabetical.")
        + pane("B. value_counts", f"df['{key}'].value_counts()", big2.format(precision=0),
               "SAME numbers as A (shown alphabetically here). A one-word shortcut that only counts.")
        + pane("C. mean of True/False", f"mask.groupby(df['{key}']).mean()", big3.format(precision=1),
               "A DIFFERENT question: what SHARE of each group is True? mean = (number of True) / (rows).")
        + '<div style="clear:both"></div>')


# --------------------------------------------------------------------------- #
# Aggregation result: styled table + sorted bar chart
# --------------------------------------------------------------------------- #
def styled_agg(table: pd.DataFrame) -> None:
    """Show an aggregation table with in-cell bars (first col) and heat colours (other numeric cols)."""
    num = list(table.select_dtypes("number").columns)
    sty = table.style.format(precision=1, thousands=",").bar(subset=num[:1], color="rgba(86,180,233,.65)", vmin=0)
    if len(num) > 1:
        sty = sty.background_gradient(subset=num[1:], cmap="YlOrRd", axis=0)
    _show(_table(sty))


def bar_labeled(s: pd.Series, title: str, xlabel: str, *, fmt: str = "{:,.0f}", highlight_max: bool = True):
    """Sorted horizontal bars with the value written at the end of each bar."""
    s = s.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(s) + 1.3))
    colors = [HI if (highlight_max and v == s.max()) else BASE for v in s]
    ax.barh(s.index.astype(str), s.to_numpy(), color=colors)
    for y, v in enumerate(s):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=10)
    ax.set_xlim(0, s.max() * 1.18)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    plt.show()
    return None


# --------------------------------------------------------------------------- #
# Bikes
# --------------------------------------------------------------------------- #
def weekday_bars(series: pd.Series, name: str = "Berri 1", ax=None):
    """Mean per weekday; the busiest bar is highlighted and annotated."""
    m = series.groupby(series.index.day_name()).mean().reindex(WEEKDAYS)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
    top = m.idxmax()
    ax.bar([d[:3] for d in WEEKDAYS], m.to_numpy(), color=[HI if d == top else BASE for d in WEEKDAYS])
    i = WEEKDAYS.index(top)
    ax.annotate(f"Busiest: {top}\n{m[top]:,.0f} riders/day", (i, m[top]), xytext=(i + 1.2, m[top] * 1.04),
                arrowprops={"arrowstyle": "->"}, fontsize=9, va="bottom")
    ax.axvspan(4.5, 6.5, color="gray", alpha=0.12)
    ax.text(5.5, m.max() * 0.08, "weekend", ha="center", fontsize=9)
    ax.set_ylim(0, m.max() * 1.3)
    ax.set_title(f"{name}: average riders per day, by weekday", loc="left", fontweight="bold")
    ax.set_ylabel("Average riders per day")
    ax.spines[["top", "right"]].set_visible(False)
    if own:
        fig.tight_layout()
        plt.show()
    return None if own else ax


def month_line(series: pd.Series, name: str = "Berri 1", ax=None):
    """Monthly total as a line; summer (Jun-Aug) shaded, peak annotated."""
    m = series.groupby(series.index.month).sum()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axvspan(5.5, 8.5, color="#e69f00", alpha=0.2, label="summer (Jun-Aug)")
    ax.plot(m.index, m.to_numpy(), marker="o", color=ACC)
    pk = m.idxmax()
    ax.annotate(f"peak: {MONTHS[pk - 1]}\n{m[pk]:,.0f}", (pk, m[pk]), xytext=(pk + 1.2, m[pk] * 0.9),
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set_xticks(range(1, 13), MONTHS, fontsize=8)
    ax.set_ylim(0, m.max() * 1.15)
    ax.set_title(f"{name}: total riders per month (2012)", loc="left", fontweight="bold")
    ax.set_ylabel("Total riders")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if own:
        fig.tight_layout()
        plt.show()
    return None if own else ax


def _heat(ax, grid: pd.DataFrame, cmap, cbar_label, fmt="{:,.0f}", fig=None):
    vals = grid.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns], fontsize=8)
    ax.set_yticks(range(grid.shape[0]), [str(r) for r in grid.index], fontsize=8)
    vmax = np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if vals[i, j] > vmax * 0.6 else "black")
    cb = (fig or ax.figure).colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    return im


def weekday_month_heatmap(series: pd.Series, name: str = "Berri 1"):
    """Weekday x month grid of average daily riders: two group keys pictured as a heat-map."""
    g = series.groupby([series.index.month, series.index.day_name()]).mean().unstack(0).reindex(WEEKDAYS)
    g.columns = [MONTHS[m - 1] for m in g.columns]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _heat(ax, g, "YlOrRd", "avg riders per day", fig=fig)
    ax.set_title(f"{name}: weekday x month (each cell = one group)", loc="left", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Weekday")
    fig.tight_layout()
    plt.show()
    return g


def pivot_heatmap(df: pd.DataFrame, rows: str, cols: str, *, title: str = "", top_cols: int = 5):
    """Two-key groupby(...).size().unstack() drawn as an annotated heat-map (share of each row, in %)."""
    tab = df.groupby([rows, cols]).size().unstack(fill_value=0)
    tab = tab[tab.sum().sort_values(ascending=False).index[:top_cols]]
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _heat(ax, pct, "Blues", f"% of the {rows.lower()}'s complaints", fmt="{:.0f}%", fig=fig)
    ax.set_xlabel(cols)
    ax.set_ylabel(rows)
    ax.set_title(title or f"groupby(['{rows}', '{cols}']).size().unstack()", loc="left", fontweight="bold")
    fig.tight_layout()
    plt.show()
    return tab


# --------------------------------------------------------------------------- #
# Mini-project helpers
# --------------------------------------------------------------------------- #
def roadmap(current: int | None = None) -> None:
    """Progress strip for the mini project; step numbers <= current are shown as done."""
    steps = ["1 Load & clean", "2 Question", "3 Filter", "4 Group", "5 Visualize", "5b Bikes", "6 Finding"]
    parts = []
    for i, s in enumerate(steps):
        done = current is not None and i < current
        here = current is not None and i == current
        mark = "✓ " if done else ("▶ " if here else "")
        style = ("background:rgba(0,158,115,.40)" if done else "background:rgba(240,228,66,.40);font-weight:bold"
                 if here else "")
        parts.append(f'<span class="cv-box" style="{style}">{mark}{_html.escape(s)}</span>')
    _show('<div class="cv-cap">Your roadmap</div><div>' + '<span class="cv-arrow">➜</span>'.join(parts) + "</div>")


def _panel_heating(ax, df):
    s = ((df["Complaint Type"] == "HEATING").groupby(df["Borough"]).mean() * 100).sort_values()
    ax.barh(s.index.str.title(), s.to_numpy(), color=[HI if v == s.max() else BASE for v in s])
    for y, v in enumerate(s):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, s.max() * 1.2)
    ax.set_title("Heating: share of complaints", loc="left", fontweight="bold", fontsize=10)
    ax.set_xlabel("% of borough's complaints")
    return s


def dashboard(df: pd.DataFrame, berri: pd.Series):
    """A finished 2x2 dashboard built from the project data."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    share = _panel_heating(axes[0, 0], df)
    weekday_bars(berri, ax=axes[0, 1])
    axes[0, 1].set_title("Bikes: average riders by weekday", loc="left", fontweight="bold", fontsize=10)
    month_line(berri, ax=axes[1, 0])
    axes[1, 0].set_title("Bikes: total riders by month", loc="left", fontweight="bold", fontsize=10)
    ax = axes[1, 1]
    ax.axis("off")
    wd = berri.groupby(berri.index.day_name()).sum().idxmax()
    mo = MONTHS[berri.groupby(berri.index.month).sum().idxmax() - 1]
    text = (f"KEY FINDINGS\n\n"
            f"• Heating is {share.max():.1f}% of {share.idxmax().title()}'s\n  complaints but only {share.min():.1f}% of\n  {share.idxmin().title()}'s.\n\n"
            f"• Berri 1 is busiest on {wd}s\n  and in {mo}.\n\n"
            f"• Caution: 311 data covers one month\n  (Oct 2013); bikes cover 2012 only.")
    ax.text(0.02, 0.95, text, va="top", fontsize=11, family="monospace", transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.8", "fc": "#f2f2f2", "ec": "#888888"}, color="#222222")
    for a in axes.flat[:3]:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Neighbourhood problems and bike habits: one-page summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    return fig


def chart_makeover(share: pd.Series | None = None) -> None:
    """Before / after: the same numbers as a bad chart and as a good chart."""
    if share is None:
        share = pd.Series({"BRONX": 20.8, "BROOKLYN": 13.8, "MANHATTAN": 14.4, "QUEENS": 8.5, "STATEN ISLAND": 3.5})
    fig, (bad, good) = plt.subplots(1, 2, figsize=(12, 4.6))
    # BEFORE
    x = np.arange(len(share))
    bad.bar(x, share.to_numpy(), color=["red", "lime", "blue", "magenta", "cyan"][: len(share)],
            edgecolor="black", linewidth=3, hatch="//")
    bad.bar(x + 0.08, share.to_numpy() - 0.6, color="gray", alpha=0.5, zorder=0)  # fake 3-D shadow
    bad.set_ylim(8, 22)
    bad.set_xticks(x, [b[:3] for b in share.index])
    bad.grid(True, linewidth=1.5, color="black", alpha=0.5)
    bad.set_title("BEFORE", color="#b00000", fontweight="bold", loc="left")
    bad.set_xlabel("\n✗ no title or units   ✗ axis starts at 8, not 0   ✗ rainbow + hatch + shadow\n✗ unsorted, cryptic labels   ✗ heavy gridlines",
                   fontsize=8, color="#b00000")
    # AFTER
    s = share.sort_values()
    good.barh(s.index.str.title(), s.to_numpy(), color=[HI if v == s.max() else BASE for v in s])
    for y, v in enumerate(s):
        good.text(v, y, f" {v:.1f}%", va="center", fontsize=10)
    good.set_xlim(0, s.max() * 1.18)
    good.set_title(f"AFTER: heating is {s.max() / s.min():.0f}x more common in {s.idxmax().title()}\nthan in {s.idxmin().title()}",
                   loc="left", fontweight="bold", fontsize=11)
    good.set_xlabel("% of the borough's complaints that are about heating\n✓ title states the finding   ✓ axis from 0   ✓ sorted   ✓ one highlight colour   ✓ values labelled")
    good.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()


def explore_question(df: pd.DataFrame):
    """Dropdown of boroughs + metric toggle: top complaint types for the chosen borough."""
    import ipywidgets as w

    boroughs = ["All"] + sorted(df["Borough"].dropna().unique())
    bd = w.Dropdown(options=boroughs, description="Borough")
    md = w.ToggleButtons(options=["complaints", "% of borough", "median hours"], description="Metric")

    def draw(b, m):
        sub = df if b == "All" else df[df["Borough"] == b]
        top = sub["Complaint Type"].value_counts().head(8).index
        sub = sub[sub["Complaint Type"].isin(top)]
        g = sub.groupby("Complaint Type")
        if m == "complaints":
            s, lab, fmt = g.size(), "number of complaints", "{:,.0f}"
        elif m == "% of borough":
            s, lab, fmt = g.size() / len(df if b == "All" else df[df["Borough"] == b]) * 100, "% of the borough's complaints", "{:.1f}%"
        else:
            s, lab, fmt = g["hours"].median(), "median hours to close", "{:.1f} h"
        bar_labeled(s, f"{b}: top 8 complaint types ({m})", lab, fmt=fmt)

    out = w.interactive_output(draw, {"b": bd, "m": md})
    box = w.VBox([w.HTML("<b>Explore your question:</b> pick a borough and a metric."), bd, md, out])
    display(box)
    return box


def progress_checklist(items: list[str]):
    """Tick-boxes with a live progress bar."""
    import ipywidgets as w

    boxes = [w.Checkbox(value=False, description=t, indent=False, layout=w.Layout(width="auto")) for t in items]
    bar = w.IntProgress(value=0, min=0, max=len(items), description="Done:", bar_style="success")

    def upd(_):
        bar.value = sum(b.value for b in boxes)

    for b in boxes:
        b.observe(upd, "value")
    box = w.VBox([bar, *boxes])
    display(box)
    return box
