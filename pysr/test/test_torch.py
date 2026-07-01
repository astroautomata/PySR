import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sympy  # type: ignore

import pysr
from pysr import PySRRegressor, sympy2torch


class TestTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        # Need to import after juliacall:
        import torch

        self.torch = torch

    def test_sympy2torch(self):
        x, y, z = sympy.symbols("x y z")
        cosx = 1.0 * sympy.cos(x) + y

        X = self.torch.tensor(np.random.randn(1000, 3))
        true = 1.0 * self.torch.cos(X[:, 0]) + X[:, 1]
        torch_module = sympy2torch(cosx, [x, y, z])
        self.assertTrue(
            np.all(np.isclose(torch_module(X).detach().numpy(), true.detach().numpy()))
        )

    def test_pipeline_pandas(self):
        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            extra_sympy_mappings={},
            output_torch_format=True,
        )
        model.fit(X, y)

        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x1)", "square(cos(x1))"],
                "Loss": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        for fname in ["hall_of_fame.csv.bak", "hall_of_fame.csv"]:
            equations["Complexity Loss Equation".split(" ")].to_csv(
                Path(model.output_directory_) / model.run_id_ / fname
            )

        model.refresh(run_directory=str(Path(model.output_directory_) / model.run_id_))
        tformat = model.pytorch()
        self.assertEqual(str(tformat), "_SingleSymPyModule(expression=cos(x1)**2)")

        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X.values)).detach().numpy(),
            np.square(np.cos(X.values[:, 1])),  # Selection 1st feature
            decimal=3,
        )

    def test_pipeline(self):
        X = np.random.randn(100, 10)
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            output_torch_format=True,
        )
        model.fit(X, y)

        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x1)", "square(cos(x1))"],
                "Loss": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        for fname in ["hall_of_fame.csv.bak", "hall_of_fame.csv"]:
            equations["Complexity Loss Equation".split(" ")].to_csv(
                Path(model.output_directory_) / model.run_id_ / fname
            )

        model.refresh(run_directory=str(Path(model.output_directory_) / model.run_id_))

        tformat = model.pytorch()
        self.assertEqual(str(tformat), "_SingleSymPyModule(expression=cos(x1)**2)")

        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X)).detach().numpy(),
            np.square(np.cos(X[:, 1])),  # 2nd feature
            decimal=3,
        )

    def test_mod_mapping(self):
        x, y, z = sympy.symbols("x y z")
        expression = x**2 + sympy.atanh(sympy.Mod(y + 1, 2) - 1) * 3.2 * z

        module = sympy2torch(expression, [x, y, z])

        X = self.torch.rand(100, 3).float() * 10

        true_out = (
            X[:, 0] ** 2
            + self.torch.atanh(self.torch.fmod(X[:, 1] + 1, 2) - 1) * 3.2 * X[:, 2]
        )
        torch_out = module(X)

        np.testing.assert_array_almost_equal(
            true_out.detach(), torch_out.detach(), decimal=3
        )

    def test_custom_operator(self):
        X = np.random.randn(100, 3)
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            output_torch_format=True,
        )
        model.fit(X, y)

        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "mycustomoperator(x1)"],
                "Loss": [1.0, 0.1],
                "Complexity": [1, 2],
            }
        )

        for fname in ["hall_of_fame.csv.bak", "hall_of_fame.csv"]:
            equations["Complexity Loss Equation".split(" ")].to_csv(
                Path(model.output_directory_) / model.run_id_ / fname
            )

        MyCustomOperator = sympy.Function("mycustomoperator")

        model.set_params(
            extra_sympy_mappings={"mycustomoperator": MyCustomOperator},
            extra_torch_mappings={MyCustomOperator: self.torch.sin},
        )
        # TODO: We shouldn't need to specify the run directory here.
        model.refresh(run_directory=str(Path(model.output_directory_) / model.run_id_))
        # self.assertEqual(str(model.sympy()), "sin(x1)")
        # Will automatically use the set global state from get_hof.

        tformat = model.pytorch()
        self.assertEqual(
            str(tformat), "_SingleSymPyModule(expression=mycustomoperator(x1))"
        )
        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X)).detach().numpy(),
            np.sin(X[:, 1]),
            decimal=3,
        )

    def test_avoid_simplification(self):
        # SymPy should not simplify without permission
        torch = self.torch
        ex = pysr.export_sympy.pysr2sympy(
            "square(exp(sign(0.44796443))) + 1.5 * x1",
            # ^ Normally this would become exp1 and require
            #   its own mapping
            feature_names_in=["x1"],
            extra_sympy_mappings={"square": lambda x: x**2},
        )
        m = pysr.export_torch.sympy2torch(ex, ["x1"])
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        np.testing.assert_almost_equal(
            m(torch.tensor(X)).detach().numpy(),
            np.square(np.exp(np.sign(0.44796443))) + 1.5 * X[:, 0],
            decimal=3,
        )

    def test_issue_656(self):
        # Should correctly map numeric symbols to floats
        E_plus_x1 = sympy.exp(1) + sympy.symbols("x1")
        m = pysr.export_torch.sympy2torch(E_plus_x1, ["x1"])
        X = np.random.randn(10, 1)
        np.testing.assert_almost_equal(
            m(self.torch.tensor(X)).detach().numpy(),
            np.exp(1) + X[:, 0],
            decimal=3,
        )

    def test_constant_arguments(self):
        # Test that functions with constant arguments work correctly
        # Regression test for https://github.com/MilesCranmer/PySR/issues/656
        test_cases = [
            (pysr.export_sympy.pysr2sympy("sqrt(2)"), np.sqrt(2)),
            (sympy.exp(2), np.exp(2)),
            (sympy.log(4), np.log(4)),
            (sympy.sin(1), np.sin(1)),
        ]

        for expr, expected in test_cases:
            m = pysr.export_torch.sympy2torch(expr, [])
            result = m(self.torch.randn(10, 1))
            np.testing.assert_almost_equal(result.item(), expected, decimal=3)

        # Test with variables: sqrt(2) * x
        x = sympy.symbols("x")
        expr = sympy.sqrt(2) * x
        m = pysr.export_torch.sympy2torch(expr, [x])
        X = np.random.randn(10, 1)
        np.testing.assert_almost_equal(
            m(self.torch.tensor(X)).detach().numpy().flatten(),
            np.sqrt(2) * X[:, 0],
            decimal=3,
        )

    def test_feature_selection_custom_operators(self):
        rstate = np.random.RandomState(0)
        X = pd.DataFrame({f"k{i}": rstate.randn(2000) for i in range(10, 21)})

        def cos_approx(x):
            return 1 - (x**2) / 2 + (x**4) / 24 + (x**6) / 720

        y = X["k15"] ** 2 + 2 * cos_approx(X["k20"])

        model = PySRRegressor(
            progress=False,
            unary_operators=["cos_approx(x) = 1 - x^2 / 2 + x^4 / 24 + x^6 / 720"],
            select_k_features=3,
            maxsize=10,
            early_stop_condition=1e-5,
            extra_sympy_mappings={"cos_approx": cos_approx},
            random_state=0,
            deterministic=True,
            parallelism="serial",
        )
        np.random.seed(0)
        model.fit(X.values, y.values)
        torch_module = model.pytorch()

        np_output = model.predict(X.values)

        torch_output = torch_module(self.torch.tensor(X.values)).detach().numpy()

        np.testing.assert_almost_equal(y.values, np_output, decimal=3)
        np.testing.assert_almost_equal(y.values, torch_output, decimal=3)


def runtests(just_tests=False):
    """Run all tests in test_torch.py."""
    tests = [TestTorch]
    if just_tests:
        return tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
