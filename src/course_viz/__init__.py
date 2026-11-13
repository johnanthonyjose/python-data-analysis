"""Visual teaching aids for the pandas notebooks.

Every function *displays* something (styled tables, diagrams, widgets) so a
student can SEE what an operation does, not just read about it.

    from course_viz import (show_selection, compare_slicing, show_mask,
                            show_groupby, show_missing, flow, legend,
                            explore_slicing, explore_filter, explore_groupby)

Design rules: colours work on light AND dark Jupyter themes (translucent
backgrounds, text colour inherited), and meaning is never carried by colour
alone (✓ / ✗ marks and borders are used too).
"""

from __future__ import annotations

import html as _html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

__all__ = [
    "legend", "flow", "show_selection", "compare_slicing", "show_mask",
    "show_groupby", "show_missing", "explore_slicing", "explore_filter",
    "explore_groupby",
]

# Colour-blind-safe (Okabe-Ito) fills, translucent so any theme works.
HIT = "rgba(0, 158, 115, 0.40)"      # selected / True      (bluish green)
MISS = "rgba(213, 94, 0, 0.30)"      # rejected / False     (vermillion)
ROW = "rgba(86, 180, 233, 0.38)"     # selected rows        (sky blue)
COL = "rgba(240, 228, 66, 0.38)"     # selected columns     (yellow)
BOTH = "rgba(0, 158, 115, 0.55)"     # row AND column
DIM = "opacity: 0.38;"
GROUPS = [
    "rgba(86, 180, 233, 0.45)", "rgba(230, 159, 0, 0.45)", "rgba(0, 158, 115, 0.45)",
    "rgba(204, 121, 167, 0.45)", "rgba(240, 228, 66, 0.45)", "rgba(0, 114, 178, 0.45)",
]
_CSS = (
    "<style>.cv-box{display:inline-block;border:2px solid currentColor;border-radius:8px;"
    "padding:6px 12px;margin:2px;font-family:monospace;font-size:13px}"
    ".cv-arrow{display:inline-block;margin:0 4px;font-size:20px;vertical-align:middle}"
    ".cv-key{display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:4px;"
    "border:1px solid rgba(128,128,128,.6);font-size:13px}"
    ".cv-pane{display:inline-block;vertical-align:top;margin:0 18px 12px 0}"
    ".cv-cap{font-weight:600;margin:4px 0}</style>"
)


def _show(html: str) -> None:
    display(HTML(_CSS + html))


def _table(styler) -> str:
    return styler.to_html()


# --------------------------------------------------------------------------- #
# Small diagram helpers
# --------------------------------------------------------------------------- #
def legend(**items: str) -> None:
    """Show a colour key, e.g. legend(selected=ROW, rejected=MISS)."""
    keys = "".join(f'<span class="cv-key" style="background:{c}">{_html.escape(k)}</span>'
                   for k, c in items.items())
    _show(f"<div>{keys}</div>")


def flow(*steps: str, caption: str | None = None) -> None:
    """Draw a left-to-right pipeline: flow("df", "mask", "df[mask]")."""
    parts = '<span class="cv-arrow">➜</span>'.join(
        f'<span class="cv-box">{_html.escape(s)}</span>' for s in steps)
    cap = f'<div class="cv-cap">{_html.escape(caption)}</div>' if caption else ""
    _show(f"{cap}<div>{parts}</div>")


# --------------------------------------------------------------------------- #
# Selection: which rows / columns / cells did I pick?
# --------------------------------------------------------------------------- #
def _selected(index: pd.Index, rows) -> np.ndarray:
    if rows is None:
        return np.zeros(len(index), dtype=bool)
    if isinstance(rows, pd.Series) and rows.dtype == bool:
        return rows.reindex(index, fill_value=False).to_numpy()
    if isinstance(rows, slice):
        return index.isin(pd.Series(index).iloc[rows])
    return index.isin(list(rows))


def _styled_selection(df: pd.DataFrame, rows=None, cols=None, *, row_pos=None, caption=""):
    rmask = np.zeros(len(df), dtype=bool) if row_pos is None else np.isin(np.arange(len(df)), row_pos)
    if row_pos is None:
        rmask = _selected(df.index, rows)
    cmask = np.isin(df.columns, list(cols)) if cols is not None else np.zeros(df.shape[1], dtype=bool)
    any_r, any_c = rmask.any(), cmask.any()

    def paint(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for i in range(len(frame)):
            for j in range(frame.shape[1]):
                r, c = rmask[i], cmask[j]
                if r and c:
                    out.iat[i, j] = f"background:{BOTH}; font-weight:bold; outline:2px solid #009e73"
                elif (r and not any_c) or (c and not any_r):
                    out.iat[i, j] = f"background:{ROW if r else COL}; font-weight:bold"
                elif r:
                    out.iat[i, j] = f"background:{ROW}"
                elif c:
                    out.iat[i, j] = f"background:{COL}"
                else:
                    out.iat[i, j] = DIM
        return out

    s = df.style.apply(paint, axis=None).format(na_rep="NaN")
    if caption:
        s = s.set_caption(caption)
    return s


def show_selection(df: pd.DataFrame, rows=None, cols=None, *, n: int = 14, caption: str = "") -> None:
    """Display ``df`` with the picked rows (blue) / columns (yellow) / cells (green outline).

    ``rows`` may be labels, a boolean Series, or a slice of labels. The picture is
    cropped to ``n`` rows around the selection so it stays readable.
    """
    sel = _selected(df.index, rows)
    first = int(np.argmax(sel)) if sel.any() else 0
    lo = max(0, first - 2)
    view = df.iloc[lo: lo + n]
    _show(_table(_styled_selection(view, rows, cols, caption=caption)))


def compare_slicing(df: pd.DataFrame, start: int, stop: int, col=None) -> None:
    """Side-by-side: ``.loc[start:stop]`` (end INCLUSIVE) vs ``.iloc[start:stop]`` (end EXCLUSIVE)."""
    if not pd.api.types.is_integer_dtype(df.index):
        raise ValueError("compare_slicing needs a default integer index (0, 1, 2, ...)")
    lo = max(0, start - 2)
    hi = stop + 3
    view = df.iloc[lo:hi]
    loc_rows = [x for x in view.index if start <= x <= stop]
    cols = [col] if col is not None else None
    loc_html = _table(_styled_selection(
        view, loc_rows, cols,
        caption=f".loc[{start}:{stop}]  →  {len(loc_rows or [])} rows (includes {stop})"))
    pos = list(range(start - lo, stop - lo))
    iloc_html = _table(_styled_selection(
        view, None, cols, row_pos=pos,
        caption=f".iloc[{start}:{stop}]  →  {stop - start} rows (stops BEFORE {stop})"))
    _show(f'<div class="cv-pane">{loc_html}</div><div class="cv-pane">{iloc_html}</div>')


# --------------------------------------------------------------------------- #
# Boolean masks
# --------------------------------------------------------------------------- #
def show_mask(df: pd.DataFrame, mask: pd.Series, cols=None, *, n: int = 10, label: str = "mask") -> None:
    """Show a boolean mask as a ✓ / ✗ column beside the rows it filters.

    Rows that pass are green, rows that fail are orange and faded, so the
    student sees exactly how ``df[mask]`` decides what to keep.
    """
    mask = mask.reindex(df.index)
    view = df if cols is None else df[list(cols)]
    # a balanced sample: first rows that pass and first rows that fail, in original order
    keep = pd.concat([mask[mask].head(n // 2), mask[~mask].head(n - n // 2)]).sort_index().index
    show = view.loc[keep].copy()
    flag = mask.loc[keep]
    show.insert(0, label, np.where(flag, "✓ True", "✗ False"))

    def paint(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for idx in frame.index:
            ok = bool(flag.loc[idx])
            out.loc[idx, :] = f"background:{HIT}" if ok else f"background:{MISS}; opacity:0.55"
        return out

    kept = int(mask.sum())
    cap = f"{kept:,} of {len(mask):,} rows pass  ({kept / len(mask):.1%})"
    _show(_table(show.style.apply(paint, axis=None).set_caption(cap)))


# --------------------------------------------------------------------------- #
# GroupBy: split -> apply -> combine
# --------------------------------------------------------------------------- #
def show_groupby(df: pd.DataFrame, key: str, value: str, agg: str = "sum", *, per_group: int = 3) -> None:
    """Three panels: SPLIT rows into colour-coded groups, APPLY an aggregation, COMBINE the result."""
    groups = list(df[key].dropna().unique())[: len(GROUPS)]
    palette = {g: GROUPS[i] for i, g in enumerate(groups)}
    sample = pd.concat([df[df[key] == g].head(per_group) for g in groups])[[key, value]]

    def paint(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for idx in frame.index:
            out.loc[idx, :] = f"background:{palette.get(frame.at[idx, key], '')}"
        return out

    split = sample.style.apply(paint, axis=None).set_caption("① SPLIT — rows share a colour per group")
    sub = df[df[key].isin(groups)]
    result = sub.groupby(key)[value].agg(agg).reindex(groups).to_frame(f"{agg}({value})")
    rows = "".join(
        f'<div><span class="cv-key" style="background:{palette[g]}">{_html.escape(str(g))}</span>'
        f'{agg}( {_html.escape(value)} ) = <b>{result.loc[g].iloc[0]:,.4g}</b></div>' for g in groups)
    combined = result.style.apply(
        lambda f: pd.DataFrame({c: [f"background:{palette[g]}" for g in f.index] for c in f.columns}, index=f.index),
        axis=None).set_caption("③ COMBINE — one row per group")
    _show(
        f'<div class="cv-pane">{_table(split)}</div>'
        f'<div class="cv-pane"><div class="cv-cap">② APPLY — {agg} inside each group</div>{rows}</div>'
        f'<div class="cv-pane">{_table(combined)}</div>'
    )


# --------------------------------------------------------------------------- #
# Missing data
# --------------------------------------------------------------------------- #
def show_missing(df: pd.DataFrame, *, max_rows: int = 60, title: str = "Missing values") -> None:
    """Heat-map of NaN positions (dark = missing) plus a bar of % missing per column."""
    cols = [c for c in df.columns if df[c].isna().any()] or list(df.columns[:6])
    sample = df[cols].sample(min(max_rows, len(df)), random_state=0).sort_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    ax1.imshow(sample.isna().to_numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
    ax1.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel(f"{len(sample)} random rows")
    ax1.set_title(f"{title}: each dark cell is a NaN")
    pct = df[cols].isna().mean().mul(100).sort_values()
    ax2.barh(pct.index, pct.to_numpy(), color="#d55e00")
    ax2.set_xlabel("% of rows missing")
    ax2.set_title("How much is missing?")
    ax2.set_xlim(0, 100)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Interactive explorers (ipywidgets)
# --------------------------------------------------------------------------- #
def explore_slicing(df: pd.DataFrame, col=None):
    """Drag two sliders; watch .loc and .iloc pick rows side by side."""
    import ipywidgets as w

    top = min(len(df) - 6, 40)
    start = w.IntSlider(value=3, min=0, max=top, description="start")
    length = w.IntSlider(value=4, min=1, max=10, description="length")
    out = w.interactive_output(
        lambda start, length: compare_slicing(df, start, start + length, col),
        {"start": start, "length": length})
    box = w.VBox([w.HTML("<b>Drag the sliders and compare the two tables:</b>"), start, length, out])
    display(box)
    return box


def explore_filter(df: pd.DataFrame, column: str, *, cols=None):
    """Pick a value of ``column``; see the mask, the kept rows and the count update live."""
    import ipywidgets as w

    values = df[column].dropna().value_counts().head(15).index.tolist()
    dd = w.Dropdown(options=values, description=column[:14])
    out = w.interactive_output(
        lambda v: show_mask(df, df[column] == v, cols or [column], label=f"== {v!r}"), {"v": dd})
    box = w.VBox([w.HTML(f"<b>Choose a value:</b> <code>df[df['{column}'] == value]</code>"), dd, out])
    display(box)
    return box


def explore_groupby(df: pd.DataFrame, keys: list[str], value: str):
    """Pick the key and the aggregation; see split → apply → combine redraw."""
    import ipywidgets as w

    k = w.Dropdown(options=keys, description="group by")
    a = w.ToggleButtons(options=["size", "sum", "mean", "median", "max"], description="apply")

    def draw(k, a):
        if a == "size":
            show_groupby(df.assign(_one=1), k, "_one", "sum")
        else:
            show_groupby(df, k, value, a)

    out = w.interactive_output(draw, {"k": k, "a": a})
    box = w.VBox([k, a, out])
    display(box)
    return box
