"""Verify the course notebooks.

Usage (from the repository root):

    uv run python tools/verify_notebooks.py                 # every notebook
    uv run python tools/verify_notebooks.py Module1_Basics/01_Variables_and_Expressions.ipynb

Two passes per notebook:

* student  - run exactly as shipped. No traceback may appear (unfinished
             exercises must print a friendly "❌", not crash).
* solution - every code cell tagged ``exercise`` is replaced by the first
             ```python block of the ``<details>`` solution cell that follows
             it. Then there may be no traceback AND no "❌" in any output.

Cell tags understood:
    exercise          code cell the student must complete (needs a <details> solution)
    skip-execution    never run (input(), deliberate infinite loops, ...)
    raises-exception  an error is expected and teaches something
"""

import copy
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```python\n(.*?)```", re.S)


def tags(cell) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def build_solution_notebook(nb):
    """Return (notebook with solutions spliced in, list of problems)."""
    nb = copy.deepcopy(nb)
    problems = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or "exercise" not in tags(cell):
            continue
        sol = None
        for nxt in nb.cells[i + 1 : i + 4]:
            if nxt.cell_type == "markdown" and "<details" in nxt.source:
                m = FENCE.search(nxt.source)
                sol = m.group(1) if m else None
                break
        if sol is None:
            problems.append(f"cell {i}: exercise has no <details> solution with a ```python block")
        else:
            cell.source = sol
    return nb, problems


def execute(nb, workdir: Path):
    for cell in nb.cells:
        if cell.cell_type == "code" and "skip-execution" in tags(cell):
            cell.source = "pass  # skipped by verify_notebooks.py"
    client = NotebookClient(
        nb,
        timeout=300,
        allow_errors=True,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    )
    client.execute()
    return nb


def findings(nb, *, forbid_cross: bool):
    out = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            if o.output_type == "error" and "raises-exception" not in tags(cell):
                out.append(f"cell {i}: {o.ename}: {o.evalue[:120]}")
            text = o.get("text", "") if o.output_type == "stream" else ""
            if forbid_cross and "❌" in text:
                out.append(f"cell {i}: check still failing with the solution: {text.strip()[:120]!r}")
    return out


def verify(path: Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    bad = []

    ran = execute(copy.deepcopy(nb), path.parent)
    bad += [f"[student] {f}" for f in findings(ran, forbid_cross=False)]

    sol_nb, problems = build_solution_notebook(nb)
    bad += [f"[solution] {p}" for p in problems]
    ran = execute(sol_nb, path.parent)
    bad += [f"[solution] {f}" for f in findings(ran, forbid_cross=True)]
    return bad


def main(argv):
    paths = [Path(a).resolve() for a in argv] or sorted(ROOT.glob("Module*/*.ipynb"))
    failed = 0
    for p in paths:
        bad = verify(p)
        label = p.relative_to(ROOT)
        if bad:
            failed += 1
            print(f"FAIL {label}")
            for b in bad:
                print("   ", b)
        else:
            print(f"ok   {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
