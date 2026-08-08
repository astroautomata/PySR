import functools
import importlib
import os
import pickle as pkl
import platform
import tempfile
import traceback
import unittest
import warnings
from pathlib import Path
from textwrap import dedent
from unittest import mock

import numpy as np
import pandas as pd
import sympy  # type: ignore

try:
    from sklearn.utils.estimator_checks import estimator_checks_generator
except ImportError:
    from sklearn.utils.estimator_checks import check_estimator

    estimator_checks_generator = functools.partial(check_estimator, generate_only=True)

from pysr import (
    PySRRegressor,
    TemplateExpressionSpec,
    TensorBoardLoggerSpec,
    install,
    jl,
    load_all_packages,
)
from pysr.export_latex import sympy2latex
from pysr.export_sympy import pysr2sympy
from pysr.feature_selection import _handle_feature_selection, run_feature_selection
from pysr.julia_helpers import _load_cluster_manager, init_julia
from pysr.sr import (
    _check_assertions,
    _process_constraints,
    _suggest_keywords,
    _validate_elementwise_loss,
    _validate_export_mappings,
    idx_model_selection,
)

from .params import (
    DEFAULT_NCYCLES,
    DEFAULT_NITERATIONS,
    DEFAULT_PARAMS,
    DEFAULT_POPULATIONS,
    skip_if_beartype,
)
from .test_type_specs import TestTypeSpecs

# Disables local saving:
os.environ["SYMBOLIC_REGRESSION_IS_TESTING"] = os.environ.get(
    "SYMBOLIC_REGRESSION_IS_TESTING", "true"
)

# Import from juliacall at end:
from juliacall import JuliaError  # type: ignore


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Using inspect,
        # get default niterations from PySRRegressor, and double them:
        self.default_test_kwargs = dict(
            progress=False,
            model_selection="accuracy",
            niterations=DEFAULT_NITERATIONS * 2,
            populations=DEFAULT_POPULATIONS * 2,
            temp_equation_file=True,
        )
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(100, 5)

    def test_temp_equation_file_respects_tempdir(self):
        with tempfile.TemporaryDirectory() as d:
            tempdir = Path(d) / "pysr-temp"
            model = PySRRegressor(
                temp_equation_file=True, tempdir=str(tempdir), run_id="t"
            )
            model._setup_equation_file()
            self.assertEqual(Path(model.output_directory_).parent, tempdir)

    def test_linear_relation(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y)
        print(model.equations_)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_linear_relation_named(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y, variable_names=["c1", "c2", "c3", "c4", "c5"])
        self.assertIn("c1", model.equations_.iloc[-1]["equation"])

    def test_linear_relation_weighted_bumper(self):
        y = self.X[:, 0]
        weights = np.ones_like(y)
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
            bumper=True,
        )
        model.fit(self.X, y, weights=weights)
        print(model.equations_)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)
        self.assertEqual(
            jl.seval("((::Val{x}) where x) -> x")(model.julia_options_.bumper), True
        )

    def test_multiprocessing_turbo_custom_objective(self):
        for loss_key in ["loss_function", "loss_function_expression"]:
            with self.subTest(loss_key=loss_key):
                self._multiprocessing_turbo_custom_objective(loss_key)

    def _multiprocessing_turbo_custom_objective(self, loss_key):
        rstate = np.random.RandomState(0)
        y = self.X[:, 0]
        y += rstate.randn(*y.shape) * 1e-4

        node_type = "Expression" if loss_key == "loss_function_expression" else "Node"
        model = PySRRegressor(
            **self.default_test_kwargs,
            # Turbo needs to work with unsafe operators:
            unary_operators=["sqrt"],
            procs=2,
            parallelism="multiprocessing",
            turbo=True,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-10 && complexity == 1",
            **{loss_key: f"""
            function my_objective(tree::{node_type}{{T}}, dataset::Dataset{{T}}, options::Options) where T
                prediction, flag = eval_tree_array(tree, dataset.X, options)
                !flag && return T(Inf)
                abs3(x) = abs(x) ^ 3
                return sum(abs3, prediction .- dataset.y) / length(prediction)
            end
            """},
        )
        model.fit(self.X, y)
        print(model.equations_)
        best_loss = model.equations_.iloc[-1]["loss"]
        self.assertLessEqual(best_loss, 1e-10)
        self.assertGreaterEqual(best_loss, 0.0)

        # Test options stored:
        self.assertEqual(
            jl.seval("((::Val{x}) where x) -> x")(model.julia_options_.turbo), True
        )

    def test_multiline_seval(self):
        # The user should be able to run multiple things in a single seval call:
        num = jl.seval("""
            function my_new_objective(x)
                x^2
            end
            1.5
        """)
        self.assertEqual(num, 1.5)

    def test_high_precision_search_custom_loss(self):
        y = 1.23456789 * self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
            elementwise_loss="my_loss(prediction, target) = (prediction - target)^2",
            precision=64,
            parsimony=0.01,
            warm_start=True,
        )
        model.fit(self.X, y)

        # We should have that the model state is now a Float64 hof:
        test_state = model.raw_julia_state_
        self.assertTrue(jl.typeof(test_state[1]).parameters[1] == jl.Float64)

        # Test options stored:
        self.assertEqual(
            jl.seval("((::Val{x}) where x) -> x")(model.julia_options_.turbo), False
        )

    def test_loss_function_with_elementwise_signature_errors_early(self):
        """Issue #982: elementwise (prediction, target) loss passed via loss_function errors."""
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function="bad_full_objective(prediction, target) = (prediction - target)^2",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("elementwise_loss", str(cm.exception))

    def test_loss_function_noncallable_errors_early(self):
        """Issue #982: non-callable loss_function (e.g. '1.0') errors early."""
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function="1.0",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("callable", str(cm.exception))

    def test_loss_function_valid_full_objective_runs(self):
        """Issue #982: a valid (tree, dataset, options) objective is accepted."""
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function="""
            begin
                goodloss(tree, dataset, options) = zero(eltype(dataset.y))
                goodloss
            end
            """,
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        model.fit(X, y)

    def test_loss_function_varargs_objective_runs(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function="""
            begin
                goodvarloss(tree, dataset, options...) = zero(eltype(dataset.y))
                goodvarloss
            end
            """,
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        model.fit(X, y)

    def test_elementwise_loss_wrong_signature_errors_early(self):
        """Validate `elementwise_loss` signature (prediction, target[, weights])."""
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            elementwise_loss="myloss_bad_arity(a) = a",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("elementwise_loss", str(cm.exception))

    def test_elementwise_loss_with_weights_requires_three_args(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            elementwise_loss="myloss2(prediction, target) = (prediction - target)^2",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        weights = np.array([1.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, weights=weights)
        self.assertIn("elementwise_loss", str(cm.exception))
        self.assertIn("weights", str(cm.exception))

    def test_elementwise_loss_with_weights_accepts_three_args(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            elementwise_loss=(
                "myloss3(prediction, target, weights) = weights * (prediction - target)^2"
            ),
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        weights = np.array([1.0, 1.0])
        model.fit(X, y, weights=weights)

    def test_elementwise_loss_float32_probe_accepts_strictly_typed_loss(self):
        custom_loss = jl.seval(
            "(prediction::Float32, target::Float32) -> (prediction - target)^2"
        )
        _validate_elementwise_loss(
            custom_loss,
            has_weights=False,
            probe_value=np.float32(1.0),
        )

    def test_elementwise_loss_float32_fit_accepts_strictly_typed_loss(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            precision=32,
            temp_equation_file=True,
            binary_operators=["+"],
            elementwise_loss=(
                "(prediction::Float32, target::Float32) -> (prediction - target)^2"
            ),
        )
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        model.fit(X, y)

    def test_validation_helpers_skip_nonfunction(self):
        _validate_elementwise_loss(jl.seval("1.0"), has_weights=False)

    def test_validate_export_mappings_typechecks(self):
        with self.assertRaises(ValueError):
            _validate_export_mappings({"a": 1}, None)

    def test_loss_function_expression_elementwise_signature_errors_early(self):
        """Validate `loss_function_expression` signature (expression, dataset, options)."""
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function_expression="bad_expr_objective(prediction, target) = (prediction - target)^2",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("loss_function_expression", str(cm.exception))
        self.assertIn("elementwise_loss", str(cm.exception))

    def test_loss_function_wrong_signature_errors_early(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function="badloss(tree) = 0.0",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("loss_function", str(cm.exception))
        self.assertIn("(tree, dataset, options)", str(cm.exception))

    def test_loss_function_expression_wrong_signature_errors_early(self):
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            binary_operators=["+"],
            loss_function_expression="badexprloss(expression) = 0.0",
        )
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn("loss_function_expression", str(cm.exception))
        self.assertIn("(expression, dataset, options)", str(cm.exception))

    def test_operator_conflict_error(self):
        regressor = PySRRegressor(
            operators={1: ["sin"]},
            unary_operators=["sin"],
            progress=False,
            niterations=0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Cannot use `operators` with `binary_operators` or `unary_operators`",
        ):
            regressor._validate_and_modify_params()

    def test_multioutput_custom_operator_quiet_custom_complexity(self):
        y = self.X[:, [0, 1]] ** 2
        model = PySRRegressor(
            unary_operators=["square_op(x) = x^2"],
            extra_sympy_mappings={"square_op": lambda x: x**2},
            complexity_of_operators={"square_op": 2, "plus": 1},
            binary_operators=["plus"],
            verbosity=0,
            **self.default_test_kwargs,
            procs=0,
            # Test custom operators with turbo:
            turbo=True,
            # Test custom operators with constraints:
            nested_constraints={"square_op": {"square_op": 3}},
            constraints={"square_op": 10},
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
        )
        model.fit(self.X, y)
        equations = model.equations_
        print(equations)
        self.assertIn("square_op", model.equations_[0].iloc[-1]["equation"])
        self.assertLessEqual(equations[0].iloc[-1]["loss"], 1e-4)
        self.assertLessEqual(equations[1].iloc[-1]["loss"], 1e-4)

        test_y1 = model.predict(self.X)
        test_y2 = model.predict(self.X, index=[-1, -1])

        mse1 = np.average((test_y1 - y) ** 2)
        mse2 = np.average((test_y2 - y) ** 2)

        self.assertLessEqual(mse1, 1e-4)
        self.assertLessEqual(mse2, 1e-4)

    def test_custom_variable_complexity(self):
        for outer in (True, False):
            for case in (1, 2):
                y = self.X[:, [0, 1]]
                if case == 1:
                    kwargs = dict(complexity_of_variables=[2, 3])
                elif case == 2:
                    kwargs = dict(complexity_of_variables=2)

                if outer:
                    outer_kwargs = kwargs
                    inner_kwargs = dict()
                else:
                    outer_kwargs = dict()
                    inner_kwargs = kwargs

                model = PySRRegressor(
                    binary_operators=["+"],
                    verbosity=0,
                    **self.default_test_kwargs,
                    early_stop_condition=(
                        f"stop_if_{case}(l, c) = l < 1e-8 && c <= {3 if case == 1 else 2}"
                    ),
                    **outer_kwargs,
                )
                model.fit(self.X[:, [0, 1]], y, **inner_kwargs)
                self.assertLessEqual(model.get_best()[0]["loss"], 1e-8)
                self.assertLessEqual(model.get_best()[1]["loss"], 1e-8)

                self.assertEqual(model.get_best()[0]["complexity"], 2)
                self.assertEqual(
                    model.get_best()[1]["complexity"], 3 if case == 1 else 2
                )

    def test_error_message_custom_variable_complexity(self):
        X = np.ones((10, 2))
        y = np.ones((10,))
        model = PySRRegressor()
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, complexity_of_variables=[1, 2, 3])

        self.assertIn(
            "number of elements in `complexity_of_variables`", str(cm.exception)
        )

    def test_error_message_both_variable_complexity(self):
        X = np.ones((10, 2))
        y = np.ones((10,))
        model = PySRRegressor(complexity_of_variables=[1, 2])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, complexity_of_variables=[1, 2, 3])

        self.assertIn(
            "You cannot set `complexity_of_variables` at both `fit` and `__init__`.",
            str(cm.exception),
        )

    def test_multioutput_weighted_with_callable_temp_equation(self):
        X = self.X.copy()
        y = X[:, [0, 1]] ** 2
        w = self.rstate.rand(*y.shape)
        w[w < 0.5] = 0.0
        w[w >= 0.5] = 1.0

        # Double equation when weights are 0:
        y = (2 - w) * y
        # Thus, pysr needs to use the weights to find the right equation!

        model = PySRRegressor(
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x**2},
            **self.default_test_kwargs,
            procs=0,
            delete_tempfiles=False,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 2",
        )
        model.fit(X.copy(), y, weights=w)

        # These tests are flaky, so don't fail test:
        try:
            np.testing.assert_almost_equal(
                model.predict(X.copy())[:, 0], X[:, 0] ** 2, decimal=3
            )
        except AssertionError:
            print("Error in test_multioutput_weighted_with_callable_temp_equation")
            print("Model equations: ", model.sympy()[0])
            print("True equation: x0^2")

        try:
            np.testing.assert_almost_equal(
                model.predict(X.copy())[:, 1], X[:, 1] ** 2, decimal=3
            )
        except AssertionError:
            print("Error in test_multioutput_weighted_with_callable_temp_equation")
            print("Model equations: ", model.sympy()[1])
            print("True equation: x1^2")

    def test_complex_equations_anonymous_stop(self):
        X = self.rstate.randn(100, 3) + 1j * self.rstate.randn(100, 3)
        y = (2 + 1j) * np.cos(X[:, 0] * (0.5 - 0.3j))
        model = PySRRegressor(
            binary_operators=["+", "-", "*"],
            unary_operators=["cos"],
            **self.default_test_kwargs,
            early_stop_condition="(loss, complexity) -> loss <= 1e-4 && complexity <= 6",
        )
        model.niterations = DEFAULT_NITERATIONS * 10
        model.fit(X, y)
        test_y = model.predict(X)
        self.assertTrue(np.issubdtype(test_y.dtype, np.complexfloating))
        self.assertLessEqual(np.average(np.abs(test_y - y) ** 2), 1e-4)

    def test_empty_operators_single_input_warm_start(self):
        X = self.rstate.randn(100, 1)
        y = X[:, 0] + 3.0
        regressor = PySRRegressor(
            unary_operators=[],
            binary_operators=["plus"],
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
        )
        self.assertTrue("None" in regressor.__repr__())
        regressor.fit(X, y)
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

        self.assertLessEqual(regressor.equations_.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Test if repeated fit works:
        regressor.set_params(
            niterations=1,
            ncycles_per_iteration=2,
            warm_start=True,
            early_stop_condition=None,
        )

        # We should have that the model state is now a Float32 hof:
        test_state = regressor.julia_state_
        self.assertTrue(
            jl.first(jl.typeof(jl.last(test_state)).parameters) == jl.Float32
        )

        # This should exit almost immediately, and use the old equations
        regressor.fit(X, y)

        self.assertLessEqual(regressor.equations_.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Tweak model selection:
        regressor.set_params(model_selection="best")
        self.assertEqual(regressor.get_params()["model_selection"], "best")
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

    def test_warm_start_set_at_init(self):
        # Smoke test for bug where warm_start=True is set at init
        y = self.X[:, 0]
        regressor = PySRRegressor(warm_start=True, max_evals=10)
        regressor.fit(self.X, y)

    def test_noisy_builtin_variable_names(self):
        y = self.X[:, [0, 1]] ** 2 + self.rstate.randn(self.X.shape[0], 1) * 0.05
        model = PySRRegressor(
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x**2},
            **self.default_test_kwargs,
            procs=0,
            denoise=True,
            early_stop_condition="stop_if(loss, complexity) = loss < 0.05 && complexity == 2",
        )
        # We expect in this case that the "best"
        # equation should be the right one:
        model.set_params(model_selection="best")
        # Also try without a temp equation file:
        model.set_params(temp_equation_file=False)
        # We also test builtin variable names
        model.fit(self.X, y, variable_names=["exec", "hash", "x3", "x4", "x5"])
        self.assertLessEqual(model.get_best()[1]["loss"], 1e-2)
        self.assertLessEqual(model.get_best()[1]["loss"], 1e-2)
        self.assertIn("exec", model.latex()[0])
        self.assertIn("hash", model.latex()[1])

    def test_pandas_resample_with_nested_constraints(self):
        X = pd.DataFrame(
            {
                "T": self.rstate.randn(500),
                "x": self.rstate.randn(500),
                "unused_feature": self.rstate.randn(500),
            }
        )

        def true_fn(x):
            return np.array(x["T"] + x["x"] ** 2 + 1.323837)

        y = true_fn(X)
        noise = self.rstate.randn(500) * 0.01
        y = y + noise
        # We also test y as a pandas array:
        y = pd.Series(y)
        # Resampled array is a different order of features:
        Xresampled = pd.DataFrame(
            {
                "unused_feature": self.rstate.randn(100),
                "x": self.rstate.randn(100),
                "T": self.rstate.randn(100),
            }
        )
        model = PySRRegressor(
            unary_operators=[],
            binary_operators=["+", "*", "/", "-"],
            **self.default_test_kwargs,
            denoise=True,
            nested_constraints={"/": {"+": 1, "-": 1}, "+": {"*": 4}},
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-3 && complexity == 7",
        )
        model.fit(X, y, Xresampled=Xresampled)
        self.assertNotIn("unused_feature", model.latex())
        self.assertIn("T", model.latex())
        self.assertIn("x", model.latex())
        self.assertLessEqual(model.get_best()["loss"], 1e-1)
        fn = model.get_best()["lambda_format"]
        X2 = pd.DataFrame(
            {
                "T": self.rstate.randn(100),
                "unused_feature": self.rstate.randn(100),
                "x": self.rstate.randn(100),
            }
        )
        self.assertLess(np.average((fn(X2) - true_fn(X2)) ** 2), 1e-1)
        self.assertLess(np.average((model.predict(X2) - true_fn(X2)) ** 2), 1e-1)

    def test_high_dim_selection_early_stop(self):
        X = pd.DataFrame({f"k{i}": self.rstate.randn(10000) for i in range(10)})
        Xresampled = pd.DataFrame({f"k{i}": self.rstate.randn(100) for i in range(10)})
        y = X["k7"] ** 2 + np.cos(X["k9"]) * 3

        model = PySRRegressor(
            unary_operators=["cos"],
            select_k_features=3,
            early_stop_condition=1e-4,  # Stop once most accurate equation is <1e-4 MSE
            maxsize=12,
            **self.default_test_kwargs,
        )
        model.set_params(model_selection="accuracy")
        model.fit(X, y, Xresampled=Xresampled)
        self.assertLess(np.average((model.predict(X) - y) ** 2), 1e-4)
        # Again, but with numpy arrays:
        model.fit(X.values, y.values, Xresampled=Xresampled.values)
        self.assertLess(np.average((model.predict(X.values) - y.values) ** 2), 1e-4)

    def test_load_model(self):
        """See if we can load a ran model from the equation file."""
        csv_file_data = """Complexity,Loss,Equation
        1,0.19951081,"1.9762075"
        3,0.12717344,"(f0 + 1.4724599)"
        4,0.104823045,"pow_abs(2.2683423, cos(f3))\""""
        # Strip the indents:
        csv_file_data = "\n".join([line.strip() for line in csv_file_data.split("\n")])

        for from_backup in [False, True]:
            output_directory = Path(tempfile.mkdtemp())
            equation_filename = str(output_directory / "hall_of_fame.csv")
            with open(equation_filename + (".bak" if from_backup else ""), "w") as f:
                f.write(csv_file_data)
            model = PySRRegressor.from_file(
                run_directory=output_directory,
                n_features_in=5,
                feature_names_in=["f0", "f1", "f2", "f3", "f4"],
                binary_operators=["+", "*", "/", "-", "^"],
                unary_operators=["cos"],
                precision=64,
            )
            X = self.rstate.rand(100, 5)
            y_truth = 2.2683423 ** np.cos(X[:, 3])
            y_test = model.predict(X, 2)

            np.testing.assert_allclose(y_truth, y_test)

    def test_load_model_with_operators_dict(self):
        csv_file_data = """Complexity,Loss,Equation
        1,0.19951081,"1.9762075"
        3,0.12717344,"(f0 + 1.4724599)"
        4,0.104823045,"pow_abs(2.2683423, cos(f3))\""""
        csv_file_data = "\n".join([line.strip() for line in csv_file_data.split("\n")])

        operators = {
            1: ["cos"],
            2: ["+", "*", "/", "-", "^", "pow_abs"],
        }

        for from_backup in [False, True]:
            output_directory = Path(tempfile.mkdtemp())
            equation_filename = output_directory / "hall_of_fame.csv"
            with open(
                equation_filename.with_suffix(".csv.bak" if from_backup else ".csv"),
                "w",
            ) as f:
                f.write(csv_file_data)

            model = PySRRegressor.from_file(
                run_directory=output_directory,
                n_features_in=5,
                feature_names_in=["f0", "f1", "f2", "f3", "f4"],
                operators=operators,
                precision=64,
            )

            X = self.rstate.rand(100, 5)
            y_truth = 2.2683423 ** np.cos(X[:, 3])
            y_test = model.predict(X, 2)
            np.testing.assert_allclose(y_truth, y_test)

    def test_load_model_simple(self):
        # Test that we can simply load a model from its equation file.
        y = self.X[:, [0, 1]] ** 2
        model = PySRRegressor(
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x**2},
            **self.default_test_kwargs,
            procs=0,
            denoise=True,
            early_stop_condition="stop_if(loss, complexity) = loss < 0.05 && complexity == 2",
        )
        rand_dir = Path(tempfile.mkdtemp())
        equation_file = rand_dir / "1" / "hall_of_fame.csv"
        model.set_params(temp_equation_file=False)
        model.set_params(output_directory=rand_dir)
        model.set_params(run_id="1")
        model.fit(self.X, y)

        # lambda functions are removed from the pickling, so we need
        # to pass it during the loading:
        model2 = PySRRegressor.from_file(
            run_directory=rand_dir / "1", extra_sympy_mappings={"sq": lambda x: x**2}
        )

        np.testing.assert_allclose(model.predict(self.X), model2.predict(self.X))

        # Try again, but using only the pickle file:
        for file_to_delete in [str(equation_file), str(equation_file) + ".bak"]:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

        # pickle_file = rand_dir / "equations.pkl"
        model3 = PySRRegressor.from_file(
            run_directory=rand_dir / "1", extra_sympy_mappings={"sq": lambda x: x**2}
        )
        np.testing.assert_allclose(model.predict(self.X), model3.predict(self.X))

    def test_jl_function_error(self):
        # TODO: Move this to better class
        with self.assertRaises(ValueError) as cm:
            PySRRegressor(unary_operators=["1"]).fit([[1]], [1])

        self.assertIn(
            "When building operators for arity 1, `'1'` did not return a Julia function",
            str(cm.exception),
        )

    def test_template_expressions_and_custom_complexity(self):
        # Create random data between -1 and 1
        X = self.rstate.uniform(-1, 1, (100, 2))

        # Ground truth: sin(x + y)
        y = np.sin(X[:, 0] + X[:, 1])

        # Create model with template that includes the missing sin operator
        model = PySRRegressor(
            expression_spec=TemplateExpressionSpec(
                ["f"], "sin_of_f((; f), (x, y)) = sin(f(x, y))"
            ),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],  # No sin operator!
            maxsize=10,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-10 && complexity == 6",
            # Custom complexity *function*:
            complexity_mapping="my_complexity(ex) = sum(t -> 2, get_tree(ex))",
            **self.default_test_kwargs,
        )

        model.fit(X, y)

        # Test on out of domain data - this should still work due to sin template!
        X_test = self.rstate.uniform(2, 10, (25, 2))
        y_test = np.sin(X_test[:, 0] + X_test[:, 1])
        y_pred = model.predict(X_test)

        test_mse = np.mean((y_test - y_pred) ** 2)
        self.assertLess(test_mse, 1e-5)

        # Check there is a row with complexity 6 and MSE < 1e-10
        df = model.equations_
        good_rows = df[(df.complexity == 6) & (df.loss < 1e-10)]
        self.assertGreater(len(good_rows), 0)

        # Check there are NO rows with lower complexity and MSE < 1e-10
        simpler_good_rows = df[(df.complexity < 6) & (df.loss < 1e-10)]
        self.assertEqual(len(simpler_good_rows), 0)

        # Make sure that a nice error is raised if we try to get the sympy expression:
        # f"`expression_spec={self.expression_spec_}` does not support sympy export."
        with self.assertRaises(ValueError) as cm:
            model.sympy()
        self.assertRegex(
            str(cm.exception),
            r"`expression_spec=.*TemplateExpressionSpec.*` does not support sympy export.",
        )
        with self.assertRaises(ValueError):
            model.latex()
        with self.assertRaises(ValueError):
            model.jax()
        with self.assertRaises(ValueError):
            model.pytorch()
        with self.assertRaises(ValueError):
            model.latex_table()

    def test_template_expression_with_parameters_multiout(self):
        # Create random data
        X_continuous = self.rstate.uniform(-1, 1, (100, 2))
        category = self.rstate.randint(0, 3, 100)  # 3 classes
        X = np.hstack([X_continuous, category[:, None] + 1])

        # Ground truth: p[class] * x1^2 + x2 where p = [0.5, 1.0, 2.0]
        true_p = [0.5, 1.0, 2.0]
        y_1 = np.array(
            [true_p[c] * x1**2 + x2 for x1, x2, c in zip(X[:, 0], X[:, 1], category)]
        )
        y_2 = y_1 * 2
        y = np.column_stack([y_1, y_2])

        # Create model with template that includes parameters
        model = PySRRegressor(
            **self.default_test_kwargs,
            expression_spec=TemplateExpressionSpec(
                "p[class] * x1^2 + f(x2)",
                expressions=["f"],
                parameters={"p": 3},
                variable_names=["x1", "x2", "class"],
            ),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            maxsize=10,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-10 && complexity <= 3",
        )

        model.fit(X, y)

        # Test on new data
        X_continuous_test = self.rstate.uniform(-1, 1, (25, 2))
        category_test = self.rstate.randint(0, 3, 25)
        X_test = np.hstack([X_continuous_test, category_test[:, None] + 1])
        y_test_1 = np.array(
            [
                true_p[c] * x1**2 + x2
                for x1, x2, c in zip(X_test[:, 0], X_test[:, 1], category_test)
            ]
        )
        y_test_2 = y_test_1 * 2
        y_test = np.column_stack([y_test_1, y_test_2])
        y_pred = model.predict(X_test)

        test_mse = np.mean((y_test - y_pred) ** 2)
        self.assertLess(test_mse, 1e-5)

    def test_tensorboard_logger(self):

        if platform.system() == "Windows":
            self.skipTest("Skipping test on Windows")

        """Test TensorBoard logger functionality."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
                EventAccumulator,
            )
        except ImportError:
            self.skipTest("TensorBoard not installed. Skipping test.")

        y = self.X[:, 0]
        for warm_start in [False, True]:
            with tempfile.TemporaryDirectory() as tmpdir:
                logger_spec = TensorBoardLoggerSpec(
                    log_dir=tmpdir, log_interval=2, overwrite=True
                )
                model = PySRRegressor(
                    **self.default_test_kwargs,
                    logger_spec=logger_spec,
                    early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
                    warm_start=warm_start,
                )
                model.fit(self.X, y)
                logger = model.logger_
                # Should restart from same logger if warm_start is True
                model.fit(self.X, y)
                logger2 = model.logger_

                if warm_start:
                    self.assertEqual(logger, logger2)
                else:
                    self.assertNotEqual(logger, logger2)

                # Verify log directory exists and contains TensorBoard files
                log_dir = Path(tmpdir)
                assert log_dir.exists()
                files = list(log_dir.glob("events.out.tfevents.*"))
                assert len(files) == 1 if warm_start else 2

                # Load and verify TensorBoard events
                event_acc = EventAccumulator(str(log_dir))
                event_acc.Reload()

                # Check that we have the expected scalar summaries
                scalars = event_acc.Tags()["scalars"]
                self.assertIn("search/data/summaries/pareto_volume", scalars)
                self.assertIn("search/data/summaries/min_loss", scalars)

                # Check that we have multiple events for each summary
                pareto_events = event_acc.Scalars("search/data/summaries/pareto_volume")
                min_loss_events = event_acc.Scalars("search/data/summaries/min_loss")

                self.assertGreater(len(pareto_events), 0)
                self.assertGreater(len(min_loss_events), 0)

                # Verify model still works as expected
                self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_negative_losses(self):
        X = self.rstate.rand(100, 3) * 20.0
        variance = X[:, 0] * 0.1
        eps = self.rstate.randn(100) * np.sqrt(variance)
        y = X[:, 0] + X[:, 1] ** 2 + eps
        spec = TemplateExpressionSpec(
            expressions=["f_mu", "f_logvar"],
            variable_names=["x1", "x2", "x3", "y"],
            combine="mu = f_mu(x1, x2, x3); logvar = f_logvar(x1, x2, x3); 0.5f0 * (logvar + (mu - y)^2 / exp(logvar)) - 2",
        )
        model = PySRRegressor(
            **self.default_test_kwargs,
            expression_spec=spec,
            binary_operators=["+", "*", "-"],
            unary_operators=["log", "exp"],
            elementwise_loss="(pred, targ) -> pred",
            loss_scale="linear",
            early_stop_condition="stop_if_under_n1(loss, complexity) = loss < -1.0",
        )
        model.fit(np.column_stack([X, y]), 0 * y)
        self.assertLessEqual(model.get_best()["loss"], -1.0)

    def test_comparison_operator(self):
        X = self.rstate.randn(100, 2)
        y = ((X[:, 0] + X[:, 1]) < (X[:, 0] * X[:, 1])).astype(float)

        model = PySRRegressor(
            binary_operators=["<", "+", "*"],
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity <= 7",
        )

        model.fit(X, y)

        best_equation = model.get_best()["equation"]
        self.assertIn("less", best_equation)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

        y_pred = model.predict(X)
        np.testing.assert_array_almost_equal(y, y_pred, decimal=3)

    def test_operators_parameter(self):
        X = self.rstate.randn(100, 3)
        # Create a function that would be perfect for muladd: muladd(x, y, z) = x*y + z
        y = X[:, 0] * X[:, 1] + X[:, 2] + np.sin(X[:, 0])
        # Test that operators parameter works with arity > 2
        model = PySRRegressor(
            operators={1: ["sin"], 3: ["muladd"]},
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity <= 10",
        )
        model.fit(X, y)
        # Should work with both sin and muladd operators
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_constraints_n_arity_validation(self):
        X = self.rstate.randn(10, 2)
        y = X[:, 0] + X[:, 1]

        model = PySRRegressor(
            operators={1: ["sin"], 2: ["+", "*"], 3: ["muladd"]},
            constraints={
                "sin": -1,
                "+": (-1, -1),
                "*": (-1, 1),
                "muladd": (-1, -1, 1),
            },
            niterations=1,
            progress=False,
            temp_equation_file=False,
        )
        try:
            model.fit(X, y)
        except Exception as e:
            if "constraint tuple has length" in str(e):
                self.fail(f"Valid constraints should not raise validation error: {e}")

        with self.assertRaises(ValueError) as cm:
            invalid_model = PySRRegressor(
                operators={3: ["muladd"]},
                constraints={"muladd": (-1, -1)},
                niterations=1,
                progress=False,
                temp_equation_file=False,
            )
            invalid_model.fit(X, y)

        self.assertIn("arity 3 but constraint tuple has length 2", str(cm.exception))


class TestGuesses(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(1)
        self.default_test_kwargs = dict(
            niterations=0, progress=False, temp_equation_file=False
        )

    def test_single_output_string_guesses(self):
        X = self.rstate.randn(100, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5
        model = PySRRegressor(
            guesses=["2.0*x0 + 3.0*x1 + 0.5", "x0 + x1"],
            **self.default_test_kwargs,
        )
        model.fit(X, y)
        # Check that the exact guess is in the hall of fame
        self.assertTrue(any(model.equations_["loss"] < 1e-10))

    def test_custom_variable_names_guesses(self):
        X = self.rstate.randn(100, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5
        model = PySRRegressor(
            guesses=["2.0*feature1 + 3.0*feature2 + 0.5"],
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity <= 5",
            **self.default_test_kwargs,
        )
        model.fit(X, y, variable_names=["feature1", "feature2"])
        # Check that the exact guess is in the hall of fame
        self.assertTrue(any(model.equations_["loss"] < 1e-10))

    def test_multi_output_guesses(self):
        X = self.rstate.randn(100, 2)
        Y = np.column_stack([2.0 * X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
        model = PySRRegressor(
            guesses=[["2.0*x0 + x1"], ["x0 - x1"]],
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity <= 5",
            **self.default_test_kwargs,
        )
        model.fit(X, Y)
        # Check both outputs have good fits
        for i, eqs_df in enumerate(model.equations_):
            self.assertTrue(any(eqs_df["loss"] < 1e-10))

    def test_template_expression_guesses(self):
        X = self.rstate.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        template = TemplateExpressionSpec(
            expressions=["f"], combine="f(x0, x1)", variable_names=["x0", "x1"]
        )
        model = PySRRegressor(
            expression_spec=template,
            guesses=[{"f": "#1 + #2"}],
            **self.default_test_kwargs,
        )
        model.fit(X, y)
        # Check that a good fit was found
        self.assertTrue(any(model.equations_["loss"] < 1e-10))

    def test_multi_output_template_expression_guesses(self):
        X = self.rstate.randn(100, 2)
        Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
        template = TemplateExpressionSpec(
            expressions=["f"], combine="f(x0, x1)", variable_names=["x0", "x1"]
        )
        model = PySRRegressor(
            expression_spec=template,
            guesses=[
                [{"f": "#1 + #2"}],
                [{"f": "#1 - #2"}],
            ],
            **self.default_test_kwargs,
        )
        model.fit(X, Y)
        # Check both outputs have good fits
        for i, eqs_df in enumerate(model.equations_):
            self.assertTrue(any(eqs_df["loss"] < 1e-10))

    def test_invalid_multi_output_format_guesses(self):
        X = self.rstate.randn(100, 2)
        Y = np.column_stack([X[:, 0], X[:, 1]])
        model = PySRRegressor(guesses=["x0", "x1"])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, Y)
        self.assertIn("must be a list of lists", str(cm.exception))

    def test_wrong_number_of_guess_lists(self):
        X = self.rstate.randn(100, 2)
        Y = np.column_stack([X[:, 0], X[:, 1]])
        model = PySRRegressor(guesses=[["x0"]])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, Y)
        self.assertIn("must match number of outputs", str(cm.exception))

    @skip_if_beartype
    def test_non_list_guesses_single_output(self):
        X = self.rstate.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        model = PySRRegressor(guesses="x0 + x1")
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn(
            "guesses must be a list for single-output regression", str(cm.exception)
        )

    def test_multiple_lists_single_output_guesses(self):
        X = self.rstate.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        model = PySRRegressor(guesses=[["x0 + x1"], ["x0 - x1"]])
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y)
        self.assertIn(
            "For single output, provide a list of strings/dicts", str(cm.exception)
        )

    def test_vector_of_vectors_single_output_guesses(self):
        X = self.rstate.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        model = PySRRegressor(
            guesses=[["x0 + x1", "x0 - x1"]],
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity <= 5",
            **self.default_test_kwargs,
        )
        model.fit(X, y)
        self.assertTrue(any(model.equations_["loss"] < 1e-10))

    def test_guesses_use_zero_based_indexing(self):
        # Test that guesses use 0-based indexing (x0, x1, x2)
        # not 1-based (x1, x2, x3)
        X = self.rstate.randn(100, 3)
        y = X[:, 0] * X[:, 1] + X[:, 2]  # True function

        # Test with correct guess (should have near-zero loss)
        model_correct = PySRRegressor(
            binary_operators=["+", "*"],
            guesses=["x0 * x1 + x2"],  # Correct 0-based indexing
            **self.default_test_kwargs,
        )
        model_correct.fit(X, y)
        self.assertLess(model_correct.equations_.iloc[-1]["loss"], 1e-10)

        # Test with wrong guess (off-by-one indexing, should have high loss)
        model_wrong = PySRRegressor(
            binary_operators=["+", "*"],
            guesses=["x1 * x2 + x0"],  # Wrong columns if 0-indexed
            **self.default_test_kwargs,
        )
        model_wrong.fit(X, y)
        self.assertGreater(model_wrong.equations_.iloc[-1]["loss"], 1.0)

    def test_unary_operators_in_guesses(self):
        # Test that unary operators (like log) can be used in guesses
        X = np.abs(self.rstate.randn(100, 2)) + 1  # Ensure positive for log
        y = np.log(X[:, 0]) + 2.5 * X[:, 1]

        # Test that log operator is parsed and used correctly
        model = PySRRegressor(
            binary_operators=["+", "*"],
            unary_operators=["log"],
            guesses=["log(x0) + 1.0 * x1"],  # Uses log operator (wrong constant)
            niterations=0,  # MUST use 0 to test the guess itself
            progress=False,
            temp_equation_file=False,
        )
        model.fit(X, y)
        # With niterations=0, constants still get optimized, so loss should be near-zero
        self.assertLess(model.equations_.iloc[-1]["loss"], 1e-10)
        # Verify log is in the equation
        self.assertIn("log", str(model.equations_.iloc[-1]["sympy_format"]))

    def test_empty_guesses_single_output(self):
        X = self.rstate.randn(50, 2)
        y = X[:, 0] + 0.1 * X[:, 1]
        model = PySRRegressor(
            guesses=[], **{**self.default_test_kwargs, "niterations": 0}
        )
        model.fit(X, y)
        self.assertIsNotNone(model.equations_)


def manually_create_model(equations, feature_names=None):
    if feature_names is None:
        feature_names = ["x0", "x1"]

    output_directory = tempfile.mkdtemp()
    run_id = "test"
    model = PySRRegressor(
        progress=False,
        niterations=1,
        extra_sympy_mappings={},
        output_jax_format=False,
        model_selection="accuracy",
        output_directory=output_directory,
        run_id=run_id,
    )
    model.output_directory_ = output_directory
    model.run_id_ = run_id
    os.makedirs(Path(output_directory) / run_id, exist_ok=True)

    # Set up internal parameters as if it had been fitted:
    if isinstance(equations, list):
        # Multi-output.
        model.nout_ = len(equations)
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(feature_names, dtype=object)
        for i in range(model.nout_):
            equations[i]["complexity loss equation".split(" ")].to_csv(
                str(
                    Path(output_directory)
                    / run_id
                    / f"hall_of_fame_output{i+1}.csv.bak"
                )
            )
    else:
        model.nout_ = 1
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(feature_names, dtype=object)
        equations["complexity loss equation".split(" ")].to_csv(
            str(Path(output_directory) / run_id / "hall_of_fame.csv.bak")
        )

    model.refresh()

    return model


class TestBest(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(10, 2)
        self.y = np.cos(self.X[:, 0]) ** 2
        equations = pd.DataFrame(
            {
                "equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "loss": [1.0, 0.1, 1e-5],
                "complexity": [1, 2, 3],
            }
        )
        self.model = manually_create_model(equations)
        self.equations_ = self.model.equations_

    def test_best(self):
        self.assertEqual(self.model.sympy(), sympy.cos(sympy.Symbol("x0")) ** 2)

    def test_index_selection(self):
        self.assertEqual(self.model.sympy(-1), sympy.cos(sympy.Symbol("x0")) ** 2)
        self.assertEqual(self.model.sympy(2), sympy.cos(sympy.Symbol("x0")) ** 2)
        self.assertEqual(self.model.sympy(1), sympy.cos(sympy.Symbol("x0")))
        self.assertEqual(self.model.sympy(0), 1.0)

    def test_best_tex(self):
        self.assertEqual(self.model.latex(), "\\cos^{2}{\\left(x_{0} \\right)}")

    def test_best_lambda(self):
        X = self.X
        y = self.y
        for f in [self.model.predict, self.equations_.iloc[-1]["lambda_format"]]:
            np.testing.assert_almost_equal(f(X), y, decimal=3)

    def test_all_selection_strategies(self):
        equations = pd.DataFrame(
            dict(
                loss=[1.0, 0.1, 0.01, 0.001 * 1.4, 0.001],
                score=[0.5, 1.0, 0.5, 0.5, 0.3],
            )
        )
        idx_accuracy = idx_model_selection(equations, "accuracy")
        self.assertEqual(idx_accuracy, 4)
        idx_best = idx_model_selection(equations, "best")
        self.assertEqual(idx_best, 3)
        idx_score = idx_model_selection(equations, "score")
        self.assertEqual(idx_score, 1)


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(0)

    def test_feature_selection(self):
        X = self.rstate.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        selected = run_feature_selection(X, y, select_k_features=2)
        np.testing.assert_array_equal(selected, [False, False, True, True, False])

    def test_feature_selection_handler(self):
        X = self.rstate.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        var_names = [f"x{i}" for i in range(5)]
        selected_X, selection = _handle_feature_selection(
            X,
            select_k_features=2,
            variable_names=var_names,
            y=y,
        )
        np.testing.assert_array_equal(selection, [False, False, True, True, False])
        selected_var_names = [var_names[i] for i in range(5) if selection[i]]
        self.assertEqual(set(selected_var_names), set("x2 x3".split(" ")))
        np.testing.assert_array_equal(
            np.sort(selected_X, axis=1), np.sort(X[:, [2, 3]], axis=1)
        )


class TestMiscellaneous(unittest.TestCase):
    """Test miscellaneous functions."""

    def test_pickle_inv_sympy_expression(self):
        """Test that sympy expressions with the inv operator can be pickled and unpickled correctly."""
        expr_str = "inv(x0) + x1"
        sympy_expr = pysr2sympy(expr_str, feature_names_in=["x0", "x1"])

        # Evaluate the original expression at a test point
        test_vals = {sympy.Symbol("x0"): 2.0, sympy.Symbol("x1"): 3.0}
        original_result = float(sympy_expr.subs(test_vals))

        # Pickle and unpickle the sympy expression
        serialized = pkl.dumps(sympy_expr)
        deserialized_expr = pkl.loads(serialized)

        # Evaluate the unpickled expression at the same test point
        unpickled_result = float(deserialized_expr.subs(test_vals))

        # Verify the results match
        self.assertEqual(original_result, unpickled_result)

        # Check that the same operator mapping was used (1/x for inv)
        self.assertEqual(original_result, 0.5 + 3.0)  # 1/2 + 3 = 3.5

    def test_predict_replaces_spaces_in_dataframe_columns(self):
        # Regression for #690.
        model = PySRRegressor(
            niterations=1,
            populations=1,
            procs=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
        )

        X_fit = pd.DataFrame({"a b": [1.0, 2.0], "c d": [3.0, 4.0]})
        y = X_fit["a b"] + X_fit["c d"]
        X_pred = X_fit.copy()

        model.fit(X_fit, y)
        np.testing.assert_array_equal(model.feature_names_in_, np.array(["a_b", "c_d"]))

        y_pred = model.predict(X_pred)
        assert np.isfinite(y_pred).all()

    def test_pickle_with_temp_equation_file(self):
        """If we have a temporary equation file, unpickle the estimator."""
        model = PySRRegressor(
            populations=int(1 + DEFAULT_POPULATIONS / 5),
            temp_equation_file=True,
            parallelism="serial",
        )
        nout = 3
        X = np.random.randn(100, 2)
        y = np.random.randn(100, nout)
        model.fit(X, y)
        contents = model.equation_file_contents_.copy()

        y_predictions = model.predict(X)

        equation_file_base = Path("outputs") / model.run_id_ / "hall_of_fame"
        for i in range(1, nout + 1):
            assert not os.path.exists(str(equation_file_base) + f"_output{i}.csv.bak")

        equation_file_base = (
            Path(model.output_directory_) / model.run_id_ / "hall_of_fame"
        )
        for i in range(1, nout + 1):
            assert os.path.exists(str(equation_file_base) + f"_output{i}.csv.bak")

        with tempfile.NamedTemporaryFile() as pickle_file:
            pkl.dump(model, pickle_file)
            pickle_file.seek(0)
            model2 = pkl.load(pickle_file)

        contents2 = model2.equation_file_contents_
        cols_to_check = ["equation", "loss", "complexity"]
        for frame1, frame2 in zip(contents, contents2):
            pd.testing.assert_frame_equal(frame1[cols_to_check], frame2[cols_to_check])

        y_predictions2 = model2.predict(X)
        np.testing.assert_array_almost_equal(y_predictions, y_predictions2)

    def test_scikit_learn_compatibility(self):
        """Test PySRRegressor compatibility with scikit-learn."""
        model = PySRRegressor(
            niterations=int(1 + DEFAULT_NITERATIONS / 10),
            populations=int(1 + DEFAULT_POPULATIONS / 3),
            ncycles_per_iteration=int(2 + DEFAULT_NCYCLES / 10),
            verbosity=0,
            progress=False,
            random_state=0,
            deterministic=True,  # Deterministic as tests require this.
            parallelism="serial",
            warm_start=False,
            temp_equation_file=True,
        )  # Return early.

        exception_messages = []
        for _, check in estimator_checks_generator(model):
            if check.func.__name__ in {
                # We can use complex data, so avoid this check
                "check_complex_data",
                # We handle kwargs manually, so skip this check
                "check_do_not_raise_errors_in_init_or_set_params",
                # TODO:
                "check_n_features_in_after_fitting",
            }:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    check(model)
                print("Passed", check.func.__name__)
            except unittest.SkipTest as exc:
                print("Skipped", check.func.__name__, "with:", exc)
            except Exception:
                error_message = str(traceback.format_exc())
                exception_messages.append(
                    f"{check.func.__name__}:\n" + error_message + "\n"
                )
                print("Failed", check.func.__name__, "with:")
                # Add a leading tab to error message, which
                # might be multi-line:
                print("\n".join([(" " * 4) + row for row in error_message.split("\n")]))
        # If any checks failed don't let the test pass.
        self.assertEqual(len(exception_messages), 0)

    def test_invalid_batch_size_corrects_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PySRRegressor(
                batch_size=0,
                niterations=1,
                progress=False,
                populations=3,
            )
            X = np.random.randn(10, 2)
            y = X[:, 0]
            model.fit(X, y)

        self.assertTrue(any("batch_size" in str(w.message) for w in caught))

    def test_progress_disabled_when_stdout_lacks_buffer(self):
        fake_stdout = type(
            "FakeStdout", (), {"write": lambda self, *_args, **_kwargs: None}
        )()
        fake_stdout.__dir__ = lambda: ["write"]  # Ensure "buffer" is absent

        with mock.patch("pysr.sr.sys.stdout", fake_stdout):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PySRRegressor(
                    progress=True,
                    niterations=1,
                    populations=3,
                )
                X = np.random.randn(10, 2)
                y = X[:, 0]
                model.fit(X, y)

        self.assertTrue(any("progress bar" in str(w.message) for w in caught))

    def test_param_groupings(self):
        """Test that param_groupings are complete"""
        param_groupings_file = Path(__file__).parent.parent / "param_groupings.yml"
        if not param_groupings_file.exists():
            return

        # Read the file, discarding lines ending in ":",
        # and removing leading "\s*-\s*":
        params = []
        with open(param_groupings_file, "r") as f:
            for line in f.readlines():
                if line.strip().endswith(":"):
                    continue
                if line.strip().startswith("-"):
                    params.append(line.strip()[1:].strip())

        regressor_params = [
            p for p in DEFAULT_PARAMS.keys() if p not in ["self", "kwargs"]
        ]

        # Check the sets are equal:
        self.assertSetEqual(set(params), set(regressor_params))

    def test_load_all_packages(self):
        """Test we can load all packages at once."""
        load_all_packages()
        self.assertTrue(jl.seval("ClusterManagers isa Module"))
        self.assertTrue(jl.seval("SlurmClusterManager isa Module"))

        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_NTASKS": "2"}):
            with self.assertRaisesRegex(JuliaError, "allocation has 2 tasks"):
                _load_cluster_manager("slurm")(1)

    def test_get_batch_size(self):
        """Test the _get_batch_size function."""
        from pysr.sr import _get_batch_size

        # Test None (auto) mode with different dataset sizes
        self.assertEqual(_get_batch_size(500, None), 500)
        self.assertEqual(_get_batch_size(999, None), 999)
        self.assertEqual(_get_batch_size(1000, None), 1000)
        self.assertEqual(_get_batch_size(1001, None), 128)
        self.assertEqual(_get_batch_size(1500, None), 128)
        self.assertEqual(_get_batch_size(4999, None), 128)
        self.assertEqual(_get_batch_size(5000, None), 256)
        self.assertEqual(_get_batch_size(10000, None), 256)
        self.assertEqual(_get_batch_size(49999, None), 256)
        self.assertEqual(_get_batch_size(50000, None), 512)
        self.assertEqual(_get_batch_size(100000, None), 512)

        # Test explicit batch_size
        self.assertEqual(_get_batch_size(1000, 64), 64)
        self.assertEqual(_get_batch_size(1000, 2000), 1000)  # Capped at dataset size
        self.assertEqual(_get_batch_size(50, 100), 50)  # Capped at dataset size

    def test_batching_auto(self):
        """Test that batching='auto' works correctly."""
        model = PySRRegressor()
        self.assertEqual(model.batching, "auto")

        X_small = np.random.randn(100, 2)
        y_small = np.random.randn(100)
        model = PySRRegressor(batching="auto", niterations=0)
        model.fit(X_small, y_small)

        X_large = np.random.randn(1001, 2)
        y_large = np.random.randn(1001)
        model2 = PySRRegressor(batching="auto", niterations=0)
        model2.fit(X_large, y_large)

    def test_batch_size_negative_warning(self):
        """Test that batch_size < 1 gives a warning for integers only."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning) as context:
                model = PySRRegressor(batch_size=0, niterations=0)
                model.fit(X, y)
            self.assertIn("batch_size", str(context.exception))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = PySRRegressor(batch_size=None, niterations=0)
            model.fit(X, y)


class TestHelpMessages(unittest.TestCase):
    """Test user help messages."""

    @skip_if_beartype
    def test_deprecation(self):
        """Ensure that deprecation works as expected.

        This should give a warning, and sets the correct value.
        """
        with self.assertWarns(FutureWarning):
            model = PySRRegressor(fractionReplaced=0.2)
        # This is a deprecated parameter, so we should get a warning.

        # The correct value should be set:
        self.assertEqual(model.fraction_replaced, 0.2)

        with self.assertRaises(NotImplementedError):
            model.equation_file_

        with self.assertRaises(ValueError) as cm:
            PySRRegressor.from_file(equation_file="", run_directory="")

        self.assertIn("Passing `equation_file` is deprecated", str(cm.exception))

    def test_deprecated_functions(self):
        with self.assertWarns(FutureWarning):
            install()

        _jl = None

        with self.assertWarns(FutureWarning):
            _jl = init_julia()

        self.assertEqual(_jl, jl)

    def test_power_law_warning(self):
        """Ensure that a warning is given for a power law operator."""
        with self.assertWarns(UserWarning):
            _process_constraints({2: ["^"]}, {})

    def test_from_file_requires_operator_configuration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir()
            # Minimal hall_of_fame.csv to satisfy the existence check
            (run_dir / "hall_of_fame.csv").write_text("complexity,loss,equation\n")

            with self.assertRaises(ValueError) as cm:
                PySRRegressor.from_file(run_directory=run_dir, n_features_in=1)

            self.assertIn("must provide either `operators`", str(cm.exception))

    def test_size_warning(self):
        """Ensure that a warning is given for a large input size."""
        model = PySRRegressor()
        X = np.random.randn(50001, 2)
        y = np.random.randn(50001)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                model.fit(X, y)
            self.assertIn("more than 50,000", str(context.exception))

    def test_deterministic_warnings(self):
        """Ensure that warnings are given for determinism"""
        model = PySRRegressor(random_state=0)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                model.fit(X, y)
            self.assertIn("`deterministic=True`", str(context.exception))

    def test_deterministic_errors(self):
        """Setting deterministic without random_state should error"""
        model = PySRRegressor(deterministic=True)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_extra_sympy_mappings_undefined(self):
        """extra_sympy_mappings=None errors for custom operators"""
        model = PySRRegressor(unary_operators=["square2(x) = x^2"])
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_sympy_function_fails_as_variable(self):
        model = PySRRegressor()
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["x1", "N"])
        self.assertIn("Variable name", str(cm.exception))

    def test_bad_variable_names_fail(self):
        model = PySRRegressor()
        X = np.random.randn(100, 1)
        y = np.random.randn(100)

        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["Tr(Tij)"])
        self.assertIn("Invalid variable name", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["f{c}"])
        self.assertIn("Invalid variable name", str(cm.exception))

    @skip_if_beartype
    def test_bad_kwargs(self):
        bad_kwargs = [
            dict(
                kwargs=dict(
                    elementwise_loss="g(x, y) = 0.0", loss_function="f(args...) = 0.0"
                ),
                error=JuliaError,
            ),
            dict(
                kwargs=dict(maxsize=3),
                error=ValueError,
            ),
            dict(
                kwargs=dict(tournament_selection_n=10, population_size=3),
                error=ValueError,
            ),
            dict(
                kwargs=dict(optimizer_algorithm="COBYLA"),
                error=NotImplementedError,
            ),
            dict(
                kwargs=dict(
                    constraints={
                        "+": (3, 5),
                    }
                ),
                error=NotImplementedError,
            ),
            dict(
                kwargs=dict(binary_operators=["α(x, y) = x - y"]),
                error=ValueError,
            ),
            dict(
                kwargs=dict(model_selection="unknown"),
                error=NotImplementedError,
            ),
        ]
        for opt in bad_kwargs:
            model = PySRRegressor(**opt["kwargs"], niterations=1)
            with self.assertRaises(opt["error"]):
                model.fit([[1]], [1])
                model.get_best()
                print("Failed", opt["kwargs"])

    def test_suggest_keywords(self):
        # Easy
        self.assertEqual(
            _suggest_keywords(PySRRegressor, "loss_function"),
            ["loss_function", "loss_function_expression"],
        )

        # More complex, and with error
        with self.assertRaises(TypeError) as cm:
            PySRRegressor(ncyclesperiterationn=5)

        self.assertIn(
            "`ncyclesperiterationn` is not a valid keyword", str(cm.exception)
        )
        self.assertIn("Did you mean", str(cm.exception))
        self.assertIn("`ncycles_per_iteration`, ", str(cm.exception))
        self.assertIn("`niterations`", str(cm.exception))

        # Farther matches (this might need to be changed)
        with self.assertRaises(TypeError) as cm:
            PySRRegressor(nary_operators=["+", "-"])

        self.assertIn("`unary_operators`, `binary_operators`", str(cm.exception))

    def test_issue_666(self):
        # Try the equivalent of `from pysr import *`
        pysr_module = importlib.import_module("pysr")
        names_to_import = pysr_module.__all__

        for name in names_to_import:
            getattr(pysr_module, name)


TRUE_PREAMBLE = "\n".join(
    [
        r"\usepackage{breqn}",
        r"\usepackage{booktabs}",
        "",
        "...",
        "",
    ]
)


class TestLaTeXTable(unittest.TestCase):
    def setUp(self):
        equations = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", "x0 + x1 - cos(x1 * x0)"],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 8],
            )
        )
        self.model = manually_create_model(equations)
        self.maxDiff = None

    def create_true_latex(self, middle_part, include_score=False):
        if include_score:
            true_latex_table_str = r"""
                \begin{table}[h]
                \begin{center}
                \begin{tabular}{@{}cccc@{}}
                \toprule
                Equation & Complexity & Loss & Score \\
                \midrule"""
        else:
            true_latex_table_str = r"""
                \begin{table}[h]
                \begin{center}
                \begin{tabular}{@{}ccc@{}}
                \toprule
                Equation & Complexity & Loss \\
                \midrule"""
        true_latex_table_str += middle_part
        true_latex_table_str += r"""\bottomrule
            \end{tabular}
            \end{center}
            \end{table}
        """
        # First, remove empty lines:
        true_latex_table_str = "\n".join(
            [line.strip() for line in true_latex_table_str.split("\n") if len(line) > 0]
        )
        return true_latex_table_str.strip()

    def test_simple_table(self):
        latex_table_str = self.model.latex_table(
            columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.05$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_other_precision(self):
        latex_table_str = self.model.latex_table(
            precision=5, columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.0520$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.023150$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.1235 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_include_score(self):
        latex_table_str = self.model.latex_table()
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ & $5.11$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE
            + "\n"
            + self.create_true_latex(middle_part, include_score=True)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_last_equation(self):
        latex_table_str = self.model.latex_table(
            indices=[2], columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_multi_output(self):
        equations1 = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", "x0 + x1 - cos(x1 * x0)"],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 8],
            )
        )
        equations2 = pd.DataFrame(
            dict(
                equation=["x1", "cos(x1)", "x0 * x0 * x1"],
                loss=[1.32, 0.052, 2e-15],
                complexity=[1, 2, 5],
            )
        )
        equations = [equations1, equations2]
        model = manually_create_model(equations)
        middle_part_1 = r"""
            $y_{0} = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
            $y_{0} = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
            $y_{0} = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ & $5.11$ \\
        """
        middle_part_2 = r"""
            $y_{1} = x_{1}$ & $1$ & $1.32$ & $0.0$ \\
            $y_{1} = \cos{\left(x_{1} \right)}$ & $2$ & $0.0520$ & $3.23$ \\
            $y_{1} = x_{0} x_{0} x_{1}$ & $5$ & $2.00 \cdot 10^{-15}$ & $10.3$ \\
        """
        true_latex_table_str = "\n\n".join(
            self.create_true_latex(part, include_score=True)
            for part in [middle_part_1, middle_part_2]
        )
        true_latex_table_str = TRUE_PREAMBLE + "\n" + true_latex_table_str
        latex_table_str = model.latex_table()

        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_latex_float_precision(self):
        """Test that we can print latex expressions with custom precision"""
        expr = sympy.Float(4583.4485748, dps=50)
        self.assertEqual(sympy2latex(expr, prec=6), r"4583.45")
        self.assertEqual(sympy2latex(expr, prec=5), r"4583.4")
        self.assertEqual(sympy2latex(expr, prec=4), r"4583.")
        self.assertEqual(sympy2latex(expr, prec=3), r"4.58 \cdot 10^{3}")
        self.assertEqual(sympy2latex(expr, prec=2), r"4.6 \cdot 10^{3}")

        # Multiple numbers:
        x = sympy.Symbol("x")
        expr = x * 3232.324857384 - 1.4857485e-10
        self.assertEqual(
            sympy2latex(expr, prec=2), r"3.2 \cdot 10^{3} x - 1.5 \cdot 10^{-10}"
        )
        self.assertEqual(
            sympy2latex(expr, prec=3), r"3.23 \cdot 10^{3} x - 1.49 \cdot 10^{-10}"
        )
        self.assertEqual(
            sympy2latex(expr, prec=8), r"3232.3249 x - 1.4857485 \cdot 10^{-10}"
        )

    def test_latex_break_long_equation(self):
        """Test that we can break a long equation inside the table"""
        long_equation = """
        - cos(x1 * x0) + 3.2 * x0 - 1.2 * x1 + x1 * x1 * x1 + x0 * x0 * x0
        + 5.2 * sin(0.3256 * sin(x2) - 2.6 * x0) + x0 * x0 * x0 * x0 * x0
        + cos(cos(x1 * x0) + 3.2 * x0 - 1.2 * x1 + x1 * x1 * x1 + x0 * x0 * x0)
        """
        long_equation = "".join(long_equation.split("\n")).strip()
        equations = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", long_equation],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 30],
            )
        )
        model = manually_create_model(equations)
        latex_table_str = model.latex_table()
        middle_part = r"""
        $y = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
        $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
        \begin{minipage}{0.8\linewidth} \vspace{-1em} \begin{dmath*} y = x_{0} x_{0} x_{0} + x_{0} x_{0} x_{0} x_{0} x_{0} + 3.20 x_{0} - 1.20 x_{1} + x_{1} x_{1} x_{1} + 5.20 \sin{\left(- 2.60 x_{0} + 0.326 \sin{\left(x_{2} \right)} \right)} - \cos{\left(x_{0} x_{1} \right)} + \cos{\left(x_{0} x_{0} x_{0} + 3.20 x_{0} - 1.20 x_{1} + x_{1} x_{1} x_{1} + \cos{\left(x_{0} x_{1} \right)} \right)} \end{dmath*} \end{minipage} & $30$ & $1.12 \cdot 10^{-15}$ & $1.09$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE
            + "\n"
            + self.create_true_latex(middle_part, include_score=True)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)


class TestDimensionalConstraints(unittest.TestCase):
    def setUp(self):
        self.default_test_kwargs = dict(
            progress=False,
            model_selection="accuracy",
            niterations=DEFAULT_NITERATIONS * 2,
            populations=DEFAULT_POPULATIONS * 2,
            temp_equation_file=True,
        )
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(100, 5)

    def test_dimensional_constraints(self):
        y = np.cos(self.X[:, [0, 1]])
        model = PySRRegressor(
            binary_operators=[
                "my_add(x, y) = x + y",
                "my_sub(x, y) = x - y",
                "my_mul(x, y) = x * y",
            ],
            unary_operators=["my_cos(x) = cos(x)"],
            **self.default_test_kwargs,
            early_stop_condition=1e-8,
            select_k_features=3,
            extra_sympy_mappings={
                "my_cos": sympy.cos,
                "my_add": lambda x, y: x + y,
                "my_sub": lambda x, y: x - y,
                "my_mul": lambda x, y: x * y,
            },
        )
        model.fit(self.X, y, X_units=["m", "m", "m", "m", "m"], y_units=["m", "m"])

        # The best expression should have complexity larger than just 2:
        for i in range(2):
            self.assertGreater(model.get_best()[i]["complexity"], 2)
            self.assertLess(model.get_best()[i]["loss"], 1e-6)
            simple_eqs = model.equations_[i].query("complexity <= 2")
            self.assertTrue(len(simple_eqs) == 0 or simple_eqs.loss.min() > 1e-6)

    def test_unit_checks(self):
        """This just checks the number of units passed"""
        use_custom_variable_names = False
        variable_names = None
        complexity_of_variables = 1
        weights = None
        args = (
            use_custom_variable_names,
            variable_names,
            complexity_of_variables,
            weights,
        )
        valid_units = [
            (np.ones((10, 2)), np.ones(10), ["m/s", "s"], "m"),
            (np.ones((10, 1)), np.ones(10), ["m/s"], None),
            (np.ones((10, 1)), np.ones(10), None, "km/s"),
            (np.ones((10, 1)), np.ones(10), None, ["m/s"]),
            (np.ones((10, 1)), np.ones((10, 1)), None, ["m/s"]),
            (np.ones((10, 1)), np.ones((10, 2)), None, ["m/s", ""]),
        ]
        for X, y, X_units, y_units in valid_units:
            _check_assertions(
                X,
                *args,
                y,
                X_units,
                y_units,
            )
        invalid_units = [
            (np.ones((10, 2)), np.ones(10), ["m/s", "s", "s^2"], None),
            (np.ones((10, 2)), np.ones(10), ["m/s", "s", "s^2"], "km"),
            (np.ones((10, 2)), np.ones((10, 2)), ["m/s", "s"], ["m"]),
            (np.ones((10, 1)), np.ones((10, 1)), "m/s", ["m"]),
        ]
        for X, y, X_units, y_units in invalid_units:
            with self.assertRaises(ValueError):
                _check_assertions(
                    X,
                    *args,
                    y,
                    X_units,
                    y_units,
                )

    def test_unit_propagation(self):
        """Check that units are propagated correctly.

        This also tests that variables have the correct names.
        """
        X = np.ones((100, 3))
        y = np.ones((100, 1))
        output_dir = tempfile.mkdtemp()
        run_id = "test"
        model = PySRRegressor(
            binary_operators=["+", "*"],
            early_stop_condition="(l, c) -> l < 1e-6 && c == 3",
            progress=False,
            model_selection="accuracy",
            niterations=DEFAULT_NITERATIONS * 2,
            populations=DEFAULT_POPULATIONS * 2,
            complexity_of_constants=10,
            weight_mutate_constant=0.0,
            should_optimize_constants=False,
            parallelism="serial",
            deterministic=True,
            random_state=0,
            output_directory=output_dir,
            run_id=run_id,
            warm_start=True,
        )
        model.fit(
            X,
            y,
            X_units=["m", "s", "A"],
            y_units=["m*A"],
        )
        best = model.get_best()
        self.assertIn("x0", best["equation"])
        self.assertNotIn("x1", best["equation"])
        self.assertIn("x2", best["equation"])
        self.assertEqual(best["complexity"], 3)
        self.assertTrue(
            model.equations_.iloc[0].complexity > 1
            or model.equations_.iloc[0].loss > 1e-6
        )

        # With pkl file:
        run_directory = str(Path(output_dir) / run_id)
        model2 = PySRRegressor.from_file(run_directory=run_directory)
        best2 = model2.get_best()
        self.assertIn("x0", best2["equation"])

        # From csv file alone (we need to delete pkl file:)
        # First, we delete the pkl file:
        os.remove(Path(run_directory) / "checkpoint.pkl")
        model3 = PySRRegressor.from_file(
            run_directory=run_directory,
            binary_operators=["+", "*"],
            n_features_in=X.shape[1],
        )
        best3 = model3.get_best()
        self.assertIn("x0", best3["equation"])

        # Try warm start, but with no units provided (should
        # be a different dataset, and thus different result):
        model.early_stop_condition = "(l, c) -> l < 1e-6 && c == 1"
        model.fit(X, y)
        self.assertEqual(model.equations_.iloc[0].complexity, 1)
        self.assertLess(model.equations_.iloc[0].loss, 1e-6)

    def test_process_constraints_swaps_multiplication_constraints(self):
        operators = {2: ["mult"]}
        constraints = {"mult": (1, -1)}

        processed = _process_constraints(operators, constraints)

        self.assertEqual(processed["mult"], (-1, 1))


# TODO: Determine desired behavior if second .fit() call does not have units


class TestTemplateExpressionSpec(unittest.TestCase):
    def test_num_features_symbol_keys(self):
        # ponytail: one check — dict keys must reach Julia as Symbols
        spec = TemplateExpressionSpec(
            ["f", "g"],
            "combine(fs, vars) = fs.f(vars[1], vars[2]) + fs.g(vars[3])",
            {"f": 2, "g": 1},
        )
        options = spec.julia_expression_options()
        names = jl.seval("x -> propertynames(x.structure.num_features)")(options)
        self.assertEqual(names, (jl.Symbol("f"), jl.Symbol("g")))

    def _check_macro_str(self, spec, expected_str):
        self.assertEqual(
            spec._template_macro_str().strip(), dedent(expected_str).strip()
        )

    def test_single_expression_no_params_single_variable(self):
        spec = TemplateExpressionSpec(
            combine="f(x)", expressions=["f"], variable_names=["x"]
        )
        self._check_macro_str(
            spec,
            """\
            @template_spec(expressions=(f,),) do x
                f(x)
            end
            """,
        )

    def test_multiple_expressions_no_params_multiple_variables(self):
        spec = TemplateExpressionSpec(
            combine="f(x, y) + g(z)",
            expressions=["f", "g"],
            variable_names=["x", "y", "z"],
        )
        self._check_macro_str(
            spec,
            """
            @template_spec(expressions=(f, g,),) do x, y, z
                f(x, y) + g(z)
            end
            """,
        )

    def test_single_expression_single_param_single_variable(self):
        spec = TemplateExpressionSpec(
            combine="p[1] * f(x)",
            expressions=["f"],
            variable_names=["x"],
            parameters={"p": 1},
        )
        self._check_macro_str(
            spec,
            """
            @template_spec(expressions=(f,), parameters=(p=1,),) do x
                p[1] * f(x)
            end
            """,
        )

    def test_multiple_expressions_multiple_params_multiple_variables(self):
        spec = TemplateExpressionSpec(
            combine="p1[1]*f(x,y) + p2[1]*g(z)",
            expressions=["f", "g"],
            variable_names=["x", "y", "z"],
            parameters={"p1": 2, "p2": 3},
        )
        self._check_macro_str(
            spec,
            """
            @template_spec(expressions=(f, g,), parameters=(p1=2, p2=3,),) do x, y, z
                p1[1]*f(x,y) + p2[1]*g(z)
            end
            """,
        )

    def test_complex_variable_names(self):
        spec = TemplateExpressionSpec(
            combine="f(var1) * g(var2)",
            expressions=["f", "g"],
            variable_names=["var1", "var2"],
        )
        self._check_macro_str(
            spec,
            """
            @template_spec(expressions=(f, g,),) do var1, var2
                f(var1) * g(var2)
            end
            """,
        )

    def test_mixed_parameter_types(self):
        spec = TemplateExpressionSpec(
            combine="alpha*f(x) + beta*g(y)",
            expressions=["f", "g"],
            variable_names=["x", "y"],
            parameters={"alpha": 1, "beta": 2},
        )
        self._check_macro_str(
            spec,
            """
            @template_spec(expressions=(f, g,), parameters=(alpha=1, beta=2,),) do x, y
                alpha*f(x) + beta*g(y)
            end
            """,
        )

    def test_empty_parameters_case(self):
        spec = TemplateExpressionSpec(
            combine="f(x)", expressions=["f"], variable_names=["x"], parameters={}
        )
        self.assertNotIn("parameters", spec._template_macro_str())

    def test_maximum_parameters_expressions(self):
        spec = TemplateExpressionSpec(
            combine=" + ".join([f"f{i}(x)" for i in range(5)]),
            expressions=[f"f{i}" for i in range(5)],
            variable_names=["x"],
            parameters={f"p{i}": i + 1 for i in range(5)},
        )
        macro_str = spec._template_macro_str()
        self.assertIn("expressions=(f0, f1, f2, f3, f4,),", macro_str)
        self.assertIn("parameters=(p0=1, p1=2, p2=3, p3=4, p4=5,),", macro_str)


def runtests(just_tests=False):
    """Run all tests in test.py."""
    test_cases = [
        TestPipeline,
        TestBest,
        TestFeatureSelection,
        TestMiscellaneous,
        TestHelpMessages,
        TestLaTeXTable,
        TestDimensionalConstraints,
        TestGuesses,
        TestTypeSpecs,
    ]
    if just_tests:
        return test_cases
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
