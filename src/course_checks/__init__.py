"""Tiny self-check helpers for the course notebooks.

The helpers *never raise*, so "Run All" keeps going even when an exercise is not
finished yet. They print a friendly ✅ / ❌ message instead.

    from course_checks import check, check_equal

    check(lambda: remainder == 1.0, hint="Use the % operator: number % 2")
    check_equal(lambda: top5.index.tolist(), ["a", "b"], hint="Try value_counts().head(5)")
"""

import math

__all__ = ["check", "check_equal"]


def _report(ok: bool, hint: str, error: Exception | None = None) -> bool:
    if ok:
        print("✅ Correct!")
    else:
        msg = "❌ Not quite yet."
        if error is not None:
            msg += f" ({type(error).__name__}: {error})"
        if hint:
            msg += f"\n   Hint: {hint}"
        print(msg)
    return ok


def check(condition, hint: str = "") -> bool:
    """Print ✅ if ``condition`` is truthy, otherwise ❌ plus ``hint``.

    ``condition`` may be a bool or, preferably, a zero-argument callable
    (``lambda: ...``). A callable is evaluated safely: if the student's
    unfinished code (e.g. a ``...`` placeholder) makes it fail, you get a
    hint instead of a traceback.
    """
    try:
        value = condition() if callable(condition) else condition
        return _report(bool(value), hint)
    except Exception as exc:  # noqa: BLE001 - students' code can fail in any way
        return _report(False, hint, exc)


def _same(actual, expected) -> bool:
    if actual is Ellipsis:
        return False
    if hasattr(actual, "equals") and hasattr(expected, "equals"):  # pandas objects
        return bool(actual.equals(expected))
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return bool(actual == expected)


def check_equal(actual, expected, hint: str = "") -> bool:
    """Compare ``actual`` with ``expected`` (works for numbers, lists, pandas objects).

    ``actual`` may be a value or a zero-argument callable.
    """
    try:
        value = actual() if callable(actual) else actual
        return _report(_same(value, expected), hint)
    except Exception as exc:  # noqa: BLE001
        return _report(False, hint, exc)
