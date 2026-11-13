"""Extra visuals for notebooks 10 (filtering) and 11 (dates and missing values).

    from course_viz.filtering import venn, truth_table, explore_two_filters, ...

Same design rules as ``course_viz``: translucent colours (light + dark theme),
Okabe-Ito colour-blind-safe palette, and meaning is never carried by colour
alone (✓ / ✗ marks, hatching and text labels are always present).
"""

from __future__ import annotations

import html as _html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from course_viz import HIT, MISS, _show

__all__ = [
    "venn", "truth_table", "explore_two_filters", "before_after_dates", "timeline",
    "clock", "weekday_hour_heatmap", "decision_cards", "copy_on_write_demo",
]

GREEN, ORANGE, BLUE = "#009e73", "#d55e00", "#0072b2"
OKABE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#999999"]


# --------------------------------------------------------------------------- #
# Venn diagram for two masks
# --------------------------------------------------------------------------- #
def _lens_area(d: float, r1: float, r2: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    tri = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - tri


_REGIONS = {  # name -> (in A, in B)
    "a_only": (True, False), "both": (True, True), "b_only": (False, True), "neither": (False, False),
}
_SHOW = {  # which regions each operation keeps
    "and": ["both"], "or": ["a_only", "both", "b_only"], "not_a": ["b_only", "neither"],
    "a_only": ["a_only"], None: [],
}
_TITLE = {
    "and": "A & B   (both)", "or": "A | B   (either)", "not_a": "~A   (not A)",
    "a_only": "A & ~B   (A but not B)", None: "",
}


def venn(mask_a: pd.Series, mask_b: pd.Series, label_a: str = "A", label_b: str = "B", *,
         op: str | None = None, ax=None) -> None:
    """Two translucent circles sized by row counts, with the count of every region.

    ``op`` highlights what an operator keeps: ``"and"`` (&), ``"or"`` (|),
    ``"not_a"`` (~A) or ``"a_only"`` (A & ~B). Without ``op`` the four regions are just counted.
    """
    a = mask_a.to_numpy(bool)
    b = mask_b.to_numpy(bool)
    n = len(a)
    counts = {k: int(((a == ra) & (b == rb)).sum()) for k, (ra, rb) in _REGIONS.items()}
    na, nb, both = counts["a_only"] + counts["both"], counts["b_only"] + counts["both"], counts["both"]

    big, small = (na, nb) if na >= nb else (nb, na)
    r_big = 1.0
    r_small = np.sqrt(small / big) * r_big if big else 0.1
    r_a, r_b = (r_big, r_small) if na >= nb else (r_small, r_big)
    target = both / max(big, 1) * np.pi * r_big ** 2
    lo, hi = abs(r_a - r_b), r_a + r_b
    for _ in range(60):  # bisect the centre distance so the overlap area matches `both`
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _lens_area(mid, r_a, r_b) > target else (lo, mid)
    d = (lo + hi) / 2
    ca, cb = (0.0, 0.0), (d, 0.0)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5.4))
    x0, x1 = min(ca[0] - r_a, cb[0] - r_b) - 0.35, max(ca[0] + r_a, cb[0] + r_b) + 0.35
    y0, y1 = -max(r_a, r_b) - 0.35, max(r_a, r_b) + 0.35
    gx, gy = np.meshgrid(np.linspace(x0, x1, 500), np.linspace(y0, y1, 500))
    in_a = (gx - ca[0]) ** 2 + (gy - ca[1]) ** 2 <= r_a ** 2
    in_b = (gx - cb[0]) ** 2 + (gy - cb[1]) ** 2 <= r_b ** 2

    img = np.zeros(gx.shape + (4,))
    for name in _SHOW[op]:
        ra, rb = _REGIONS[name]
        img[(in_a == ra) & (in_b == rb)] = (0.0, 0.62, 0.45, 0.55)
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="lower", zorder=1)
    ax.add_patch(plt.Circle(ca, r_a, fc=BLUE, alpha=0.22, ec=BLUE, lw=2, zorder=2))
    ax.add_patch(plt.Circle(cb, r_b, fc=ORANGE, alpha=0.22, ec=ORANGE, lw=2, zorder=2))
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="gray", lw=1.5, ls="--", zorder=3))

    # representative point (centroid of the grid cells) of every region, for the arrows
    def centre(ra, rb):
        sel = (in_a == ra) & (in_b == rb)
        return (gx[sel].mean(), gy[sel].mean()) if sel.any() else None

    label_pos = {"a_only": (x0 + 0.25 * (x1 - x0), y1 + 0.02), "both": (x0 + 0.5 * (x1 - x0), y0 - 0.02),
                 "b_only": (x0 + 0.78 * (x1 - x0), y1 + 0.02)}
    names = {"a_only": f"only {label_a}", "both": "both", "b_only": f"only {label_b}"}
    for key, (ra, rb) in list(_REGIONS.items())[:3]:
        c = centre(ra, rb)
        txt = f"{names[key]}\n{counts[key]:,}"
        va = "bottom" if key != "both" else "top"
        if c is None:
            ax.text(*label_pos[key], txt, ha="center", va=va, fontsize=10, fontweight="bold")
            continue
        ax.annotate(txt, xy=c, xytext=label_pos[key], ha="center", va=va, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2), zorder=5)
        ax.plot(*c, "o", color="black", ms=3, zorder=5)
    ax.text(x1 - 0.05, y0 + 0.05, f"neither: {counts['neither']:,}", ha="right", va="bottom", fontsize=10,
            fontweight="bold")
    ax.text(x0 + 0.05, y0 + 0.05, f"all {n:,} rows", ha="left", va="bottom", fontsize=9, color="gray")
    ax.text(x0 + 0.05, y1 - 0.05, f"blue circle:   {label_a} = {na:,}", ha="left", va="top", fontsize=9, color=BLUE)
    ax.text(x0 + 0.05, y1 - 0.05 - 0.11 * (y1 - y0) / 1.8, f"orange circle: {label_b} = {nb:,}", ha="left",
            va="top", fontsize=9, color=ORANGE)

    if op:
        kept = sum(counts[k] for k in _SHOW[op])
        ax.set_title(f"{_TITLE[op]}  keeps the green region:  ✓ {kept:,} rows", fontsize=12, pad=34)
    else:
        ax.set_title("Circle area is proportional to the number of rows", fontsize=11, pad=34)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    if own:
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def truth_table(mask_a: pd.Series, mask_b: pd.Series, label_a: str = "A", label_b: str = "B") -> None:
    """The four True/False combinations of two masks: how many rows each has and what &, |, ~ say."""
    a = mask_a.to_numpy(bool)
    b = mask_b.to_numpy(bool)

    def cell(v):
        bg = HIT if v else MISS
        return f'<td style="background:{bg};text-align:center">{"✓ True" if v else "✗ False"}</td>'

    rows = ""
    for va in (True, False):
        for vb in (True, False):
            n = int(((a == va) & (b == vb)).sum())
            rows += (f"<tr>{cell(va)}{cell(vb)}{cell(va and vb)}{cell(va or vb)}{cell(not va)}"
                     f'<td style="text-align:right"><b>{n:,}</b></td></tr>')
    esc = _html.escape
    head = "".join(f'<th style="padding:4px 10px">{h}</th>' for h in (
        f"A<br><small>{esc(label_a)}</small>", f"B<br><small>{esc(label_b)}</small>",
        "A &amp; B<br><small>both</small>", "A | B<br><small>either</small>", "~A<br><small>not A</small>",
        "rows"))
    _show(f'<table style="border-collapse:collapse"><caption class="cv-cap">Every row falls in exactly '
          f"one of four cases</caption><tr>{head}</tr>{rows}</table>")


# --------------------------------------------------------------------------- #
# Two linked filters
# --------------------------------------------------------------------------- #
def explore_two_filters(df: pd.DataFrame, col_a: str, col_b: str, *, top: int = 8):
    """Two dropdowns (``col_a`` and ``col_b``); live count, share and a bar chart per ``col_a`` value."""
    import ipywidgets as w

    all_ = "(all)"
    opts_a = [all_] + df[col_a].dropna().value_counts().head(top).index.tolist()
    opts_b = [all_] + df[col_b].dropna().value_counts().head(top).index.tolist()
    dd_a = w.Dropdown(options=opts_a, description=col_a[:14])
    dd_b = w.Dropdown(options=opts_b, description=col_b[:14])

    def draw(a, b):
        mask_b = df[col_b] == b if b != all_ else pd.Series(True, index=df.index)
        mask_a = df[col_a] == a if a != all_ else pd.Series(True, index=df.index)
        n = int((mask_a & mask_b).sum())
        counts = df.loc[mask_b, col_a].value_counts().reindex(opts_a[1:], fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(counts) + 1.2))
        colours = [GREEN if (a in (all_, k)) else "#bbbbbb" for k in counts.index]
        bars = ax.barh(counts.index[::-1], counts.to_numpy()[::-1], color=colours[::-1])
        ax.bar_label(bars, labels=[f"{v:,}" + ("  ✓ kept" if c == GREEN else "") for v, c in
                                   zip(counts.to_numpy()[::-1], colours[::-1])], padding=3, fontsize=9)
        ax.set_xlim(0, max(counts.max(), 1) * 1.25)
        ax.set_xlabel("rows")
        cond_a = f"df['{col_a}'] == {a!r}" if a != all_ else "(no filter on A)"
        cond_b = f"df['{col_b}'] == {b!r}" if b != all_ else "(no filter on B)"
        ax.set_title(f"{cond_a}  &  {cond_b}\n→ {n:,} rows ({n / len(df):.1%} of {len(df):,})", fontsize=10,
                     loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    out = w.interactive_output(draw, {"a": dd_a, "b": dd_b})
    box = w.VBox([w.HTML("<b>Combine two conditions with &amp;:</b> pick a value for each. "
                         f"The bars show <code>{col_a}</code> for the rows matching your second choice; "
                         "green bars (✓ kept) also pass the first one."), dd_a, dd_b, out])
    display(box)
    return box


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def before_after_dates(texts: pd.Series, fmt: str) -> None:
    """The same values as text and as real dates, with a dtype badge and what each lets you do."""
    texts = texts.reset_index(drop=True)
    dates = pd.to_datetime(texts, format=fmt)
    badge = ('<span class="cv-key" style="background:{c}"><b>dtype: {d}</b></span>')
    try:
        _ = texts.iloc[0] - texts.iloc[1]
        text_try = "ok"
    except TypeError as exc:
        text_try = f"✗ TypeError: {_html.escape(str(exc))[:70]}"
    gap = dates.iloc[0] - dates.iloc[1]
    left = "".join(f"<tr><td><code>'{_html.escape(str(t))}'</code></td></tr>" for t in texts)
    right = "".join(
        f"<tr><td><code>{d}</code></td><td>{d.day_name()}</td><td>hour {d.hour}</td></tr>" for d in dates)
    _show(
        f'<div class="cv-pane"><div class="cv-cap">BEFORE: text</div>{badge.format(c=MISS, d=texts.dtype)}'
        f"<table>{left}</table><div>row 0 minus row 1: <b>{text_try}</b></div></div>"
        f'<div class="cv-cap" style="display:inline-block;margin:60px 18px 0 0;font-size:24px">➜</div>'
        f'<div class="cv-pane"><div class="cv-cap">AFTER: real dates</div>'
        f"{badge.format(c=HIT, d=dates.dtype)}"
        f"<table>{right}</table><div>row 0 minus row 1: <b>✓ {gap}</b> (a Timedelta)</div></div>")


def timeline(df: pd.DataFrame, n: int = 15, *, created="Created", closed="Closed", by="Agency",
             pool: int = 3000, seed: int = 4) -> None:
    """Gantt-like bars from created to closed. Hatched red = closed BEFORE created (impossible)."""
    d = df.iloc[:pool]
    dur = d[closed] - d[created]
    good = d[dur >= pd.Timedelta(0)].dropna(subset=[closed]).sample(n - 6, random_state=seed)
    bad_year = d[(dur < pd.Timedelta(0)) & (d[closed].dt.year > 2000)].sample(3, random_state=seed)
    bad_1900 = df[(df[closed].dt.year < 2000)].head(1)
    open_ = d[d[closed].isna()].sample(2, random_state=seed)
    rows = pd.concat([good, bad_year, open_]).sort_values(created)
    agencies = sorted(rows[by].unique())
    colour = {a: OKABE[i % len(OKABE)] for i, a in enumerate(agencies)}

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.8))
    lo = min(rows[created].min(), rows[closed].min())
    hi = max(rows[created].max(), rows[closed].max())
    pad = (hi - lo) * 0.12
    for y, (_, r) in enumerate(rows.iterrows()):
        c, e = r[created], r[closed]
        col = colour[r[by]]
        if pd.isna(e):
            ax.barh(y, (hi - c).total_seconds() / 86400, left=mdates_num(c), color=col, alpha=0.35,
                    hatch="..", ec=col, lw=1)
            ax.text(mdates_num(hi) + 0.02, y, "✗ no Closed Date: still open", va="center", fontsize=8)
        elif e < c:
            ax.barh(y, (c - e).total_seconds() / 86400, left=mdates_num(e), fc="none", ec=ORANGE, hatch="////",
                    lw=1.5)
            ax.text(mdates_num(c) + 0.02, y, "✗ closed BEFORE created", va="center", fontsize=8, color=ORANGE,
                    fontweight="bold")
        else:
            ax.barh(y, (e - c).total_seconds() / 86400, left=mdates_num(c), color=col, ec="black", lw=0.5)
            ax.plot(mdates_num(c), y, "|", color="black")
    ax.set_yticks(range(len(rows)), [f"{r[by]} #{i}" for i, (_, r) in enumerate(rows.iterrows(), 1)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(mdates_num(lo - pad), mdates_num(hi + pad * 3.2))
    import matplotlib.dates as mdates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Each bar runs from Created to Closed (hatched = impossible or still open)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colour[a], ec="black", lw=0.5, label=a) for a in agencies]
    handles += [plt.Rectangle((0, 0), 1, 1, fc="none", ec=ORANGE, hatch="////", label="closed before created")]
    handles += [plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="gray", hatch="..", label="still open (no Closed Date)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title=by)
    if len(bad_1900):
        ax.text(0.01, 0.01, f"+ {len(bad_1900)} not shown: Closed = 1900-01-01 (placeholder date)",
                transform=ax.transAxes, fontsize=8, color=ORANGE, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def mdates_num(ts) -> float:
    import matplotlib.dates as mdates
    return mdates.date2num(pd.Timestamp(ts).to_pydatetime())


def clock(series_a: pd.Series, series_b: pd.Series, title_a: str, title_b: str) -> None:
    """Two 24-hour 'clocks' (midnight at the top, clockwise) from hour -> count Series."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), subplot_kw={"projection": "polar"})
    for ax, s, title, col in zip(axes, (series_a, series_b), (title_a, title_b), (BLUE, ORANGE)):
        s = s.reindex(range(24), fill_value=0)
        theta = np.arange(24) * 2 * np.pi / 24
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(theta, s.to_numpy(), width=2 * np.pi / 24 * 0.9, color=col, alpha=0.8, ec="black", lw=0.5)
        ax.set_xticks(theta[::3], ["0h\nmidnight", "3h", "6h", "9h", "12h\nnoon", "15h", "18h", "21h"])
        ax.set_yticklabels([])
        peak = int(s.idxmax())
        ax.set_xlabel(f"peak: hour {peak} with {int(s.max()):,} complaints", fontsize=10, fontweight="bold")
        ax.set_title(title, pad=18, fontsize=11)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def weekday_hour_heatmap(created: pd.Series, title: str = "Complaints by weekday and hour") -> None:
    """Calendar-style heat-map: weekday (rows) x hour (columns), brighter = more complaints."""
    grid = pd.crosstab(created.dt.dayofweek, created.dt.hour).reindex(index=range(7), columns=range(24),
                                                                      fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h}h" for h in range(0, 24, 2)])
    ax.set_xlabel("hour of the day (0h = midnight)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="number of complaints")
    r, c = np.unravel_index(grid.to_numpy().argmax(), grid.shape)
    ax.annotate(f"busiest cell: {grid.iat[r, c]}", xy=(c, r), xytext=(min(c + 4, 18), r - 2.4 if r >= 3 else r + 2.2),
                color="white", fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", color="white"))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Missing values: decision cards, copy-on-write
# --------------------------------------------------------------------------- #
def decision_cards(cards: list[dict]) -> None:
    """Cards with a verdict, a rule of thumb and a real example. Keys: verdict, colour, when, example, code."""
    html = ""
    for c in cards:
        html += (
            f'<div class="cv-pane" style="width:30%;min-width:210px;border:2px solid currentColor;'
            f'border-radius:8px;padding:8px 12px;background:{c["colour"]}">'
            f'<div class="cv-cap" style="font-size:16px">{_html.escape(c["verdict"])}</div>'
            f'<div><i>Use when:</i> {_html.escape(c["when"])}</div>'
            f'<div style="margin-top:6px"><i>Here:</i> {_html.escape(c["example"])}</div>'
            f'<div style="margin-top:6px"><code>{_html.escape(c["code"])}</code></div></div>')
    _show(html)


def copy_on_write_demo(df: pd.DataFrame, cols=("Unique Key", "Agency", "Borough")) -> None:
    """Run 'slice, then edit the slice' on a tiny frame and show original / slice / original-after."""
    orig = df[list(cols)].head(4).copy()
    snapshot = orig.copy()
    small = orig.head(2)
    small.loc[small.index[0], "Agency"] = "CHANGED"

    def paint(cells):
        def f(frame):
            out = pd.DataFrame("", index=frame.index, columns=frame.columns)
            for i, c in cells:
                out.loc[i, c] = f"background:{ORANGE}66; font-weight:bold; outline:2px solid {ORANGE}"
            return out
        return f

    def pane(frame, title, cells=()):
        st = frame.style.apply(paint(cells), axis=None).set_caption(title)
        return f'<div class="cv-pane">{st.to_html()}</div>'

    same = orig["Agency"].equals(snapshot["Agency"])
    arrow = '<span class="cv-arrow" style="margin-top:50px">➜</span>'
    first = orig.index[0]
    _show(
        pane(snapshot, "① original df") + arrow
        + pane(small, "② small = df.head(2), then edited", [(first, "Agency")]) + arrow
        + pane(orig, "③ original df, afterwards")
        + f'<div class="cv-cap">{"✓ the original is untouched: the edit stayed in the small frame" if same else "✗ the original changed (older pandas behaviour)"}</div>')
