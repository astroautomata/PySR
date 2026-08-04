import importlib.util
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pysr import PySRRegressor
from pysr.expression_specs import ParametricExpressionSpec, TemplateExpressionSpec
from pysr.logger_specs import TensorBoardLoggerSpec


class TestRustBackend(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("symbolic_regression_rs", None)

    def test_importing_regressor_does_not_import_juliacall(self):
        code = (
            "import sys; "
            "from pysr import PySRRegressor; "
            "assert 'juliacall' not in sys.modules; "
            "assert PySRRegressor().backend == 'auto'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_auto_backend_resolves_to_rust_when_module_is_available(self):
        self._install_fake_rust_module()

        X = np.linspace(-1, 1, 16, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            binary_operators=["+"],
            niterations=1,
            populations=1,
            population_size=16,
            deterministic=True,
            random_state=0,
            progress=False,
            temp_equation_file=True,
            model_selection="accuracy",
        )

        model.fit(X, y)

        self.assertEqual(model.backend, "auto")
        self.assertEqual(model.backend_, "rust")
        calls = sys.modules["symbolic_regression_rs"].calls
        self.assertEqual(len(calls), 1)

    def test_auto_backend_resolves_to_julia_when_module_is_unavailable(self):
        sys.modules.pop("symbolic_regression_rs", None)
        model = PySRRegressor()

        with patch("pysr.sr.importlib.util.find_spec", return_value=None):
            self.assertEqual(model._resolve_backend(), "julia")

    def test_rust_backend_uses_optional_search_module(self):
        self._install_fake_rust_module()

        X = np.linspace(-1, 1, 16, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            binary_operators=["+", "-"],
            niterations=1,
            populations=1,
            population_size=16,
            deterministic=True,
            random_state=0,
            parallelism="serial",
            progress=False,
            temp_equation_file=True,
            model_selection="accuracy",
            complexity_of_constants=2,
        )

        model.fit(X, y, complexity_of_variables=3)

        calls = sys.modules["symbolic_regression_rs"].calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["operators"], {2: ["+", "sub"]})
        self.assertEqual(calls[0]["variable_names"], ["x0"])
        self.assertEqual(calls[0]["options"]["niterations"], 1)
        self.assertEqual(calls[0]["options"]["batch_size"], 16)
        self.assertEqual(calls[0]["options"]["parallelism"], "serial")
        self.assertEqual(calls[0]["options"]["complexity_of_constants"], 2)
        self.assertEqual(calls[0]["options"]["complexity_of_variables"], 3)
        self.assertEqual(model.rust_backend_version_, "test")
        self.assertLess(model.get_best()["loss"], 1e-8)
        self.assertEqual(str(model.sympy()), "x0")
        self.assertIsInstance(model.latex(), str)
        np.testing.assert_allclose(model.predict(X), y)
        self.assertIn("lambda_format", model.equations_.columns)

    def test_rust_backend_checkpoint_and_csv_loading(self):
        self._install_fake_rust_module()

        X = np.linspace(-1, 1, 16, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PySRRegressor(
                backend="rust",
                binary_operators=["+", "-"],
                niterations=1,
                populations=1,
                population_size=16,
                deterministic=True,
                random_state=0,
                progress=False,
                output_directory=tmpdir,
                run_id="rust-run",
                model_selection="accuracy",
            )
            model.fit(X, y)

            run_directory = Path(tmpdir) / "rust-run"
            equation_file = run_directory / "hall_of_fame.csv"
            checkpoint_file = run_directory / "checkpoint.pkl"
            self.assertTrue(equation_file.exists())
            self.assertTrue(checkpoint_file.exists())

            loaded = PySRRegressor.from_file(run_directory=run_directory)
            np.testing.assert_allclose(loaded.predict(X), y)
            self.assertEqual(loaded.backend, "rust")

            checkpoint_file.unlink()
            loaded_from_csv = PySRRegressor.from_file(
                run_directory=run_directory,
                backend="rust",
                binary_operators=["+", "-"],
                n_features_in=1,
            )
            np.testing.assert_allclose(loaded_from_csv.predict(X), y)

    def test_rust_backend_pickle_roundtrip(self):
        self._install_fake_rust_module()

        X = np.linspace(-1, 1, 16, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            binary_operators=["+", "-"],
            niterations=1,
            populations=1,
            population_size=16,
            deterministic=True,
            random_state=0,
            progress=False,
            temp_equation_file=True,
            model_selection="accuracy",
        )
        model.fit(X, y)

        loaded = pickle.loads(pickle.dumps(model))

        self.assertEqual(loaded.backend, "rust")
        np.testing.assert_allclose(loaded.predict(X), y)

    def _install_fake_rust_module(self):
        calls = []

        def search(X, y, *, options, operators, variable_names):
            calls.append(
                {
                    "X": X,
                    "y": y,
                    "options": options,
                    "operators": operators,
                    "variable_names": variable_names,
                }
            )
            return {
                "backend_version": "test",
                "hall_of_fame": [
                    {"complexity": 1, "loss": 0.0, "equation": variable_names[0]},
                ],
            }

        fake_module = types.SimpleNamespace(search=search, __version__="test")
        fake_module.calls = calls
        sys.modules["symbolic_regression_rs"] = fake_module

    def _assert_unsupported_before_import(
        self,
        unsupported_name,
        *,
        model_kwargs=None,
        fit_kwargs=None,
        X=None,
        y=None,
    ):
        model_kwargs = model_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        if X is None:
            X = np.ones((4, 1), dtype=np.float32)
        if y is None:
            y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
            **model_kwargs,
        )

        sys.modules.pop("symbolic_regression_rs", None)
        with patch(
            "pysr.backends.rust.import_module",
            side_effect=AssertionError("Rust module should not be imported"),
        ):
            with self.assertRaisesRegex(NotImplementedError, unsupported_name):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model.fit(X, y, **fit_kwargs)

    def test_rust_backend_missing_optional_dependency_errors(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
        )

        with patch("pysr.backends.rust.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ImportError, "pysr\\[rust\\]"):
                model.fit(X, y)

    def test_rust_backend_maps_square_to_abs2(self):
        self._install_fake_rust_module()

        X = np.ones((4, 1), dtype=np.float32)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            unary_operators=["square"],
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
        )

        model.fit(X, y)

        calls = sys.modules["symbolic_regression_rs"].calls
        self.assertEqual(calls[0]["operators"][1], ["abs2"])

    def test_rust_backend_uses_pysr_batch_size_defaults(self):
        self._install_fake_rust_module()

        X = np.linspace(-1, 1, 1200, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            batching=True,
            binary_operators=["+"],
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
        )

        model.fit(X, y)

        calls = sys.modules["symbolic_regression_rs"].calls
        self.assertTrue(calls[0]["options"]["batching"])
        self.assertEqual(calls[0]["options"]["batch_size"], 128)

    def test_rust_backend_rejects_unsupported_builtin_operator(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            unary_operators=["cube"],
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
        )

        with self.assertRaisesRegex(ValueError, "cube"):
            model.fit(X, y)

    def test_rust_backend_rejects_inline_custom_operator_before_import(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            unary_operators=["inv(x) = 1 / x"],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            niterations=1,
            populations=1,
            progress=False,
            temp_equation_file=True,
        )

        with self.assertRaisesRegex(NotImplementedError, "inline custom operators"):
            model.fit(X, y)

    def test_rust_backend_rejects_custom_loss_before_import(self):
        unsupported_cases = [
            (
                "loss_function",
                {"loss_function": "loss(tree, dataset, options) = 0.0"},
                {},
            ),
            (
                "loss_function_expression",
                {"loss_function_expression": "loss(expr, dataset, options) = 0.0"},
                {},
            ),
            ("elementwise_loss", {"elementwise_loss": "L1DistLoss()"}, {}),
            ("constraints", {"constraints": {"*": (2, 2)}}, {}),
            ("complexity_mapping", {"complexity_mapping": "x -> 1"}, {}),
            ("complexity_of_operators", {"complexity_of_operators": {"*": 2}}, {}),
            ("complexity_of_constants", {"complexity_of_constants": 1.5}, {}),
            (
                "complexity_of_variables",
                {},
                {"complexity_of_variables": [1]},
            ),
            ("nested_constraints", {"nested_constraints": {"sin": {"sin": 0}}}, {}),
            (
                "dimensional_constraint_penalty",
                {"dimensional_constraint_penalty": 1000.0},
                {},
            ),
            (
                "dimensionless_constants_only",
                {"dimensionless_constants_only": True},
                {},
            ),
            (
                "expression_spec",
                {
                    "expression_spec": TemplateExpressionSpec(
                        "f(x0)", expressions=["f"], variable_names=["x0"]
                    )
                },
                {},
            ),
            (
                "expression_spec",
                {"expression_spec": ParametricExpressionSpec(max_parameters=1)},
                {"category": np.zeros(4, dtype=np.int64)},
            ),
            ("fast_cycle", {"fast_cycle": True}, {}),
            ("turbo", {"turbo": True}, {}),
            ("bumper", {"bumper": True}, {}),
            ("autodiff_backend", {"autodiff_backend": "Zygote"}, {}),
            ("cluster_manager", {"cluster_manager": "multiprocessing"}, {}),
            ("worker_imports", {"worker_imports": ["SomePackage"]}, {}),
            ("logger_spec", {"logger_spec": TensorBoardLoggerSpec()}, {}),
            ("warm_start", {"warm_start": True}, {}),
            ("guesses", {"guesses": ["x0"]}, {}),
            ("optimizer_algorithm", {"optimizer_algorithm": "NelderMead"}, {}),
            ("precision=16", {"precision": 16}, {}),
            (
                "string early_stop_condition",
                {"early_stop_condition": "stop_if(loss, complexity) = loss < 1e-6"},
                {},
            ),
            ("parallelism", {"parallelism": "multiprocessing"}, {}),
            ("procs", {"procs": 2}, {}),
            ("heap_size_hint_in_bytes", {"heap_size_hint_in_bytes": 1_000_000}, {}),
            ("worker_timeout", {"worker_timeout": 10.0}, {}),
            (
                "units",
                {},
                {"X_units": ["m"], "y_units": "m"},
            ),
            (
                "category",
                {},
                {"category": np.zeros(4, dtype=np.int64)},
            ),
            (
                "weights",
                {},
                {"weights": np.ones(4, dtype=np.float32)},
            ),
        ]

        for unsupported_name, model_kwargs, fit_kwargs in unsupported_cases:
            with self.subTest(unsupported_name=unsupported_name):
                self._assert_unsupported_before_import(
                    unsupported_name,
                    model_kwargs=model_kwargs,
                    fit_kwargs=fit_kwargs,
                )

    def test_rust_backend_rejects_multi_output_before_import(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones((4, 2), dtype=np.float32)

        self._assert_unsupported_before_import(
            "multi-output regression",
            X=X,
            y=y,
        )

    def test_rust_backend_real_wrapper_smoke_when_installed(self):
        if importlib.util.find_spec("symbolic_regression_rs") is None:
            self.skipTest("symbolic_regression_rs is not installed")

        pre_modules = set(sys.modules)
        X = np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            backend="rust",
            binary_operators=["+", "-", "*"],
            niterations=1,
            populations=1,
            population_size=16,
            ncycles_per_iteration=20,
            maxsize=10,
            maxdepth=10,
            deterministic=True,
            random_state=0,
            progress=False,
            temp_equation_file=True,
            model_selection="accuracy",
        )

        model.fit(X, y)

        self.assertLess(model.get_best()["loss"], 1e-3)
        np.testing.assert_allclose(model.predict(X), y, atol=0.05)
        self.assertIsInstance(model.latex(), str)
        self.assertNotIn("juliacall", set(sys.modules) - pre_modules)


def runtests(just_tests=False):
    tests = [TestRustBackend]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
