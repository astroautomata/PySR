from __future__ import annotations

import difflib
import inspect
import os
import re
from pathlib import Path
from typing import Any, TypeVar, Union

from numpy import ndarray
from sklearn.utils.validation import _check_feature_names_in  # type: ignore

try:
    from typing import List
except ImportError:
    from typing_extensions import List

T = TypeVar("T", bound=Any)

ArrayLike = Union[ndarray, List[T]]
PathLike = Union[str, Path]


_regexp_im = re.compile(r"\b(\d+\.\d+)im\b")
_regexp_im_sci = re.compile(r"\b(\d+\.\d+)[eEfF]([+-]?\d+)im\b")
_regexp_sci = re.compile(r"\b(\d+\.\d+)[eEfF]([+-]?\d+)\b")


def _apply_regexp_im(x: str):
    return _regexp_im.sub(r"\1j", x)


def _apply_regexp_im_sci(x: str):
    return _regexp_im_sci.sub(r"\1e\2j", x)


def _apply_regexp_sci(x: str):
    return _regexp_sci.sub(r"\1e\2", x)


def _preprocess_julia_floats(s: str) -> str:
    if isinstance(s, str):
        s = _apply_regexp_im(s)
        s = _apply_regexp_im_sci(s)
        s = _apply_regexp_sci(s)
    return s


def _safe_check_feature_names_in(self, variable_names, generate_names=True):
    """_check_feature_names_in with compat for old versions."""
    try:
        return _check_feature_names_in(
            self, variable_names, generate_names=generate_names
        )
    except TypeError:
        return _check_feature_names_in(self, variable_names)


def _available_cpu_count() -> int:
    """The number of CPUs this process may run on.

    Unlike `os.cpu_count()`, this respects the CPU affinity mask, e.g. of a
    Slurm or container allocation that only grants some of a node's cores.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return count or 1


def _subscriptify(i: int) -> str:
    """Converts integer to subscript text form.

    For example, 123 -> "₁₂₃".
    """
    return "".join([chr(0x2080 + int(c)) for c in str(i)])


def _suggest_keywords(cls, k: str) -> list[str]:
    valid_keywords = [
        param
        for param in inspect.signature(cls.__init__).parameters
        if param not in ["self", "kwargs"]
    ]
    suggestions = difflib.get_close_matches(k, valid_keywords, n=3)
    return suggestions
