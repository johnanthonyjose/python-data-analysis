"""Extra visual helpers for the first pandas notebooks (08 and 09).

    from course_viz.intro import before_after, anatomy, dtype_badges, cheat_table
"""

from __future__ import annotations

import html as _html

import pandas as pd
from IPython.display import HTML, display

from course_viz import BOTH, COL, ROW

__all__ = ["before_after", "anatomy", "dtype_badges", "cheat_table"]

_CSS = (
    "<style>.cvi-pane{display:inline-block;vertical-align:top;margin:0 16px 12px 0;"
    "max-width:48%;overflow-x:auto}"
    ".cvi-title{font-weight:600;margin:2px 0 6px 0}"
    ".cvi-t{border-collapse:collapse;font-size:13px}"
    ".cvi-t th,.cvi-t td{border:1px solid rgba(128,128,128,.45);padding:3px 9px;text-align:right}"
    ".cvi-t td.l,.cvi-t th.l{text-align:left}"
    ".cvi-badge{display:inline-block;padding:1px 9px;border-radius:10px;font-family:monospace;"
    "font-size:12px;border:1px solid currentColor}"
    ".cvi-key{display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:4px;"
    "border:1px solid rgba(128,128,128,.6);font-size:13px}"
    ".cvi-note{font-size:12px;opacity:.8;margin-top:3px}</style>"
)
_DTYPE_COLOURS = {
    "int": "rgba(86, 180, 233, 0.45)", "float": "rgba(230, 159, 0, 0.45)",
    "str": "rgba(204, 121, 167, 0.45)", "datetime": "rgba(0, 158, 115, 0.45)",
    "bool": "rgba(240, 228, 66, 0.45)",
}


def _kind(dtype) -> str:
    s = str(dtype)
    for k in _DTYPE_COLOURS:
        if s.startswith(k):
            return k
    return "str"


def _badge(dtype) -> str:
    return (f'<span class="cvi-badge" style="background:{_DTYPE_COLOURS[_kind(dtype)]}">'
            f'{_html.escape(str(dtype))}</span>')


def _fmt(v) -> str:
    if pd.isna(v):
        return "NaN"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return _html.escape(str(v))


def before_after(left, right, left_title: str = "Without", right_title: str = "With", *, n: int = 3) -> None:
    """Two panes side by side. Each side is a DataFrame/Series (first ``n`` rows shown
    with a dtype badge under every column name) or a ready-made HTML string."""

    def pane(obj, title):
        if isinstance(obj, str):
            body = obj
        else:
            df = obj.to_frame() if isinstance(obj, pd.Series) else obj
            df = df.head(n)
            head = "".join(f'<th class="l">{_html.escape(str(c)[:28])}<br>{_badge(df[c].dtype)}</th>'
                           for c in df.columns)
            idx_name = _html.escape(str(df.index.name or ""))
            rows = "".join(
                f'<tr><th>{_fmt(i)}</th>' + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>"
                for i, r in zip(df.index, df.itertuples(index=False)))
            body = (f'<table class="cvi-t"><tr><th>{idx_name}</th>{head}</tr>{rows}</table>'
                    f'<div class="cvi-note">shape {obj.shape} &middot; index: {df.index.dtype}</div>')
        return f'<div class="cvi-pane"><div class="cvi-title">{_html.escape(title)}</div>{body}</div>'

    display(HTML(_CSS + pane(left, "✗ " + left_title) + pane(right, "✓ " + right_title)))


def anatomy(df: pd.DataFrame, column: str, row_pos: int = 1, *, n: int = 4) -> None:
    """Annotated table: index (blue), column names (yellow), one column = a Series (green),
    one row (bold + outline), values (plain)."""
    df = df.head(n)
    cols = list(df.columns)
    head = "".join(
        f'<th class="l" style="background:{BOTH if c == column else COL}">{_html.escape(str(c)[:22])}</th>'
        for c in cols)
    body = ""
    for i, (idx, r) in enumerate(zip(df.index, df.itertuples(index=False))):
        cells = ""
        for c, v in zip(cols, r):
            st = f"background:{BOTH};" if c == column else ""
            if i == row_pos:
                st += "outline:2px solid #d55e00;outline-offset:-2px;font-weight:bold;"
            cells += f'<td style="{st}">{_fmt(v)}</td>'
        body += f'<tr><th style="background:{ROW}">{_fmt(idx)}</th>{cells}</tr>'
    corner = _html.escape(str(df.index.name or ""))
    table = f'<table class="cvi-t"><tr><th style="background:{ROW}">{corner}</th>{head}</tr>{body}</table>'

    def key(c, t, extra=""):
        return f'<span class="cvi-key" style="background:{c};{extra}">{t}</span>'

    keys = (key(ROW, "index = row labels") + key(COL, "column names") +
            key(BOTH, f"one column = a Series ({_html.escape(column)})") +
            key("transparent", "one row (bold, outlined)", "outline:2px solid #d55e00") +
            key("transparent", "plain cells = values"))
    display(HTML(_CSS + f"<div>{keys}</div><div style='margin-top:6px'>{table}</div>"))


def dtype_badges(df: pd.DataFrame, *, max_cols: int = 12) -> None:
    """Table: column | dtype badge | first value | what it is."""
    meaning = {"int": "whole numbers", "float": "decimals (also: whole numbers + NaN)",
               "str": "text", "datetime": "real dates", "bool": "True / False"}
    rows = ""
    for c in df.columns[:max_cols]:
        s = df[c]
        first = s.dropna().iloc[0] if s.notna().any() else float("nan")
        rows += (f'<tr><td class="l">{_html.escape(str(c)[:30])}</td><td class="l">{_badge(s.dtype)}</td>'
                 f'<td class="l">{_fmt(first)}</td><td class="l">{meaning[_kind(s.dtype)]}</td></tr>')
    display(HTML(_CSS + '<table class="cvi-t"><tr><th class="l">column</th><th class="l">dtype</th>'
                 f'<th class="l">first value</th><th class="l">meaning</th></tr>{rows}</table>'))


def cheat_table(headers: list[str], rows: list[list[str]]) -> None:
    """A plain, theme-friendly HTML table (cells may contain simple HTML such as <code>)."""
    head = "".join(f'<th class="l">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for c in r) + "</tr>" for r in rows)
    display(HTML(_CSS + f'<table class="cvi-t"><tr>{head}</tr>{body}</table>'))
