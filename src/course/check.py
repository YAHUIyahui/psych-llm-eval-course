"""Self-check helpers for exercises. They print a verdict and never raise,
so a notebook with unfinished exercises still runs to the end."""
from __future__ import annotations

import numpy as np

_DONE = {"n": 0, "ok": 0}


def check(label: str, got, expected, tol: float | None = None) -> None:
    """Compare an exercise answer with the expected value.

    `got is None` (or Ellipsis) means "not answered yet".
    For numbers, `tol` is an absolute tolerance. For sequences, element-wise.
    """
    _DONE["n"] += 1
    if got is None or got is Ellipsis:
        print(f"[ 待完成 ] {label}")
        return None
    try:
        if tol is not None:
            ok = bool(np.allclose(np.asarray(got, dtype=float),
                                  np.asarray(expected, dtype=float), atol=tol))
        else:
            ok = bool(np.all(np.asarray(got) == np.asarray(expected)))
    except Exception:
        ok = got == expected
    if ok:
        _DONE["ok"] += 1
        print(f"[ 正确 ] {label}")
    else:
        print(f"[ 再想想 ] {label}: 你的答案 = {got!r}")


def check_true(label: str, condition) -> None:
    """Pass when `condition` is truthy; None means not answered."""
    _DONE["n"] += 1
    if condition is None:
        print(f"[ 待完成 ] {label}")
        return None
    if condition:
        _DONE["ok"] += 1
    print(("[ 正确 ] " if condition else "[ 再想想 ] ") + label)


def summary() -> None:
    print(f"本 notebook 自动批改:{_DONE['ok']} / {_DONE['n']} 通过")
