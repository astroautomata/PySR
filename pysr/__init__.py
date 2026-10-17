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

from ._backend import JULIA_DISABLED as _JULIA_DISABLED
from ._backend import require_julia as _require_julia

# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
if not _JULIA_DISABLED:
    from .julia_import import jl, SymbolicRegression  # isort:skip

# Get the version using importlib.metadata (Python >= 3.8 is required):
from importlib.metadata import PackageNotFoundError, version

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    TemplateExpressionSpec,
)
from .julia_extensions import load_all_packages
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec
from .mutations import (
    AbstractMutation,
    AddNodeMutation,
    BacksolveMutation,
    ConstantMutation,
    DeleteNodeMutation,
    DoNothingMutation,
    FeatureMutation,
    InsertNodeMutation,
    OperatorMutation,
    OptimizeMutation,
    RandomizeMutation,
    RotateTreeMutation,
    SimplifyMutation,
    SwapOperandsMutation,
)
from .plugins import (
    AbstractPlugin,
    AdaptiveMutationWeightsPlugin,
    AdaptiveParsimonyPlugin,
    MutationBurstPlugin,
    SimulatedAnnealingPlugin,
)
from .sr import PySRRegressor
from .type_specs import TypeSpec

try:
    __version__ = version("pysr")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"


def __getattr__(name: str):
    if _JULIA_DISABLED and name in {"jl", "SymbolicRegression"}:
        _require_julia()
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
    "AbstractLoggerSpec",
    "TensorBoardLoggerSpec",
    "TypeSpec",
    "AbstractMutation",
    "ConstantMutation",
    "OperatorMutation",
    "FeatureMutation",
    "SwapOperandsMutation",
    "AddNodeMutation",
    "InsertNodeMutation",
    "DeleteNodeMutation",
    "RotateTreeMutation",
    "BacksolveMutation",
    "SimplifyMutation",
    "RandomizeMutation",
    "OptimizeMutation",
    "DoNothingMutation",
    "AbstractPlugin",
    "AdaptiveMutationWeightsPlugin",
    "AdaptiveParsimonyPlugin",
    "MutationBurstPlugin",
    "SimulatedAnnealingPlugin",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]

if _JULIA_DISABLED:
    __all__ = [name for name in __all__ if name not in {"jl", "SymbolicRegression"}]
