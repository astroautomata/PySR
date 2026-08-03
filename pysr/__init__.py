from __future__ import annotations

import logging
import os

pysr_logger = logging.getLogger("pysr")
pysr_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
pysr_logger.addHandler(handler)

if os.environ.get("PYSR_USE_BEARTYPE", "0") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

# Get the version using importlib.metadata (Python >= 3.8 is required):
from importlib.metadata import PackageNotFoundError, version

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    ParametricExpressionSpec,
    TemplateExpressionSpec,
)
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec
from .sr import PySRRegressor

try:
    __version__ = version("pysr")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"


def __getattr__(name: str):
    if name in {"jl", "SymbolicRegression"}:
        # Kept lazy so importing PySRRegressor can remain Julia-free.
        from .julia_import import SymbolicRegression, jl

        value = {"jl": jl, "SymbolicRegression": SymbolicRegression}[name]
        globals()[name] = value
        return value
    if name == "load_all_packages":
        from .julia_extensions import load_all_packages

        globals()[name] = load_all_packages
        return load_all_packages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "jl",
    "SymbolicRegression",
    "sklearn_monkeypatch",
    "sympy2jax",
    "sympy2torch",
    "install",
    "load_all_packages",
    "PySRRegressor",
    "AbstractExpressionSpec",
    "ExpressionSpec",
    "TemplateExpressionSpec",
    "ParametricExpressionSpec",
    "AbstractLoggerSpec",
    "TensorBoardLoggerSpec",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]
