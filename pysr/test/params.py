import inspect
import os
import unittest

import numpy as np

from pysr import PySRRegressor

DEFAULT_PARAMS = inspect.signature(PySRRegressor.__init__).parameters
DEFAULT_NITERATIONS = DEFAULT_PARAMS["niterations"].default
DEFAULT_POPULATIONS = DEFAULT_PARAMS["populations"].default
DEFAULT_NCYCLES = DEFAULT_PARAMS["ncycles_per_iteration"].default

skip_if_beartype = unittest.skipIf(
    os.environ.get("PYSR_USE_BEARTYPE", "0") == "1",
    "Skipping because beartype would fail test",
)

# Operators whose SymPy forms use `Piecewise`, relations, or n-ary extrema:
PIECEWISE_AND_EXTREMUM_EQUATIONS = [
    "greater(x0, x1)",
    "less(x0, x1)",
    "greater_equal(x0, x1)",
    "less_equal(x0, x1)",
    "cond(x0, x1)",
    "logical_or(x0, x1)",
    "logical_and(x0, x1)",
    "relu(x0) + relu(x1)",
    "max(max(x0, x1), x2)",
    "min(x0, 0.5)",
    "clamp(x0, -0.5, 0.5)",
]


def piecewise_test_data():
    X = np.random.RandomState(0).randn(40, 3)
    X[:10, 1] = X[:10, 0]  # Ties distinguish `<` from `<=`
    return X
