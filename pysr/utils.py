from __future__ import annotations

import difflib
import inspect
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
