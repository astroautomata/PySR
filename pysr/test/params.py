import inspect
import os
import unittest

from pysr import PySRRegressor

DEFAULT_PARAMS = inspect.signature(PySRRegressor.__init__).parameters
DEFAULT_NITERATIONS = DEFAULT_PARAMS["niterations"].default
DEFAULT_POPULATIONS = DEFAULT_PARAMS["populations"].default
DEFAULT_NCYCLES = DEFAULT_PARAMS["ncycles_per_iteration"].default

skip_if_beartype = unittest.skipIf(
    os.environ.get("PYSR_USE_BEARTYPE", "0") == "1",
    "Skipping because beartype would fail test",
)
