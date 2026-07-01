"""Tests for autodiff backend functionality."""

import unittest
from typing import Literal, cast

import numpy as np
import pandas as pd

from pysr import PySRRegressor, jl

from .params import DEFAULT_NITERATIONS


class TestAutodiff(unittest.TestCase):
    def setUp(self):
        self.default_test_kwargs = dict(
            progress=False,
            model_selection="accuracy",
            niterations=DEFAULT_NITERATIONS * 2,
            populations=8,
            temp_equation_file=True,
        )
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(100, 5)

    def _run_autodiff_backend(
        self, backend: Literal["Zygote", "Mooncake", "Enzyme"]
    ) -> str:
        y = 2.5 * self.X[:, 0] + 1.3
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity <= 5",
            autodiff_backend=backend,
        )

        model.fit(self.X, y)

        best = cast(pd.Series, model.get_best())
        self.assertLessEqual(best["loss"], 1e-4)
        backend_type = cast(
            str,
            jl.seval("x -> string(typeof(x))")(model.julia_options_.autodiff_backend),
        )
        return backend_type

    def test_zygote_autodiff_backend_full_run(self):
        self.assertTrue(
            self._run_autodiff_backend("Zygote").startswith("ADTypes.AutoZygote")
        )

    # Broken until https://github.com/chalk-lab/Mooncake.jl/issues/800 is fixed
    # def test_mooncake_autodiff_backend_full_run(self):
    #     self.assertTrue(
    #         self._run_autodiff_backend("Mooncake").startswith("ADTypes.AutoMooncake")
    #     )


def runtests(just_tests=False):
    """Run all tests in test_autodiff.py."""
    tests = [TestAutodiff]
    if just_tests:
        return tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
