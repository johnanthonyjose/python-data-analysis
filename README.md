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

Dependencies are pinned in `uv.lock`, so everyone gets identical versions. (Python 3.13 is
downloaded for you by uv if you don't have it.)

## How the exercises work

Cells marked **Your turn** contain `...` where you write code. Run the cell
below it: it prints ✅ when you are right, or ❌ with a hint. It never crashes
the notebook, so you can always "Run All". Each exercise has a collapsed
**Show solution** block — try first, peek second.

## Visual and interactive explanations

The pandas notebooks (08–13) are built to be *seen*, not just read: highlighted tables
show which rows and columns a selection picks, boolean masks appear as ✓/✗ columns,
group-by is drawn as split → apply → combine, and sliders and dropdowns let you
experiment live. The helpers live in `src/course_viz/` (installed by `uv sync`).
Notebooks ship without saved outputs, so run the cells to see the visuals —
they are designed for both light and dark JupyterLab themes.

## For maintainers

```bash
uv run python tools/verify_notebooks.py    # run every notebook as a student and with solutions
uv run nbstripout --install                # optional: strip outputs on commit
```

Add a library with `uv add <package>`, then commit `pyproject.toml` and `uv.lock`.

### How this course reaches students (JupyterHub)

The hub's student image is **built from this repository**: it installs the runtime
`dependencies` from `uv.lock` plus the `course_checks` / `course_viz` package from `src/`
(the `lab` and `dev` groups are authoring tools and stay out of it). Publishing builds that
image tagged with this repo's commit, runs `tools/verify_notebooks.py` **inside it** with no
network, and only then releases the notebooks — so a notebook can never reach students
without the exact environment it was tested in.

```bash
# on the hub host (jupyter-spark repo):
scripts/add_course.sh git@github.com:<owner>/python-data-analysis.git --flat   # pull latest into staging
make publish-preview      # build the image, verify inside it, show what would change; publishes nothing
make publish              # build + verify + release (image tag course-<commit>)
make publish-status       # is what students have the latest commit?
```

Only folders containing notebooks are published (`Module*/`, with their data files). Keep
shared code in `src/` and never import from a path outside the notebook folders.
