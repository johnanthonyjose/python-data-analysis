# Python Data Analysis

A hands-on course: Python basics, data structures, then data analysis with pandas.

| Module | Notebooks |
|---|---|
| 1 Basics | Variables, conditionals, loops, functions |
| 2 Data structures | Strings, lists & tuples, dictionaries |
| 3 Pandas | Reading CSVs, selecting, filtering, dates & missing values, group-by, mini project |

## Getting started

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/) — it installs
the right Python and every library for you.

1. Clone this repository. The two datasets are stored with
   [Git LFS](https://git-lfs.com), so run `git lfs install` once before cloning
   (or `git lfs pull` afterwards).
2. Install everything (Python 3.12+, pandas, numpy, matplotlib, JupyterLab):
   ```bash
   uv sync
   ```
3. Start JupyterLab:
   ```bash
   uv run jupyter lab
   ```
4. Open `Module1_Basics/01_Variables_and_Expressions.ipynb` and start learning.

Dependencies are pinned in `uv.lock`, so everyone gets identical versions.

## How the exercises work

Cells marked **Your turn** contain `...` where you write code. Run the cell
below it: it prints ✅ when you are right, or ❌ with a hint. It never crashes
the notebook, so you can always "Run All". Each exercise has a collapsed
**Show solution** block — try first, peek second.

## For maintainers

```bash
uv run python tools/verify_notebooks.py    # run every notebook as a student and with solutions
uv run nbstripout --install                # optional: strip outputs on commit
```

Add a library with `uv add <package>`, then commit `pyproject.toml` and `uv.lock`.
