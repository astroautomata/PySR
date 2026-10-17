"""Real-wheel acceptance tests for the optional Rust backend."""

import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy  # type: ignore[import-untyped]

pytest.importorskip("symbolic_regression_rs")

from pysr import PySRRegressor, TemplateExpressionSpec
from pysr._backend import JULIA_DISABLED


@pytest.fixture
def data():
    i = np.arange(64, dtype=np.float64)
    X = np.column_stack(((i - 32) / 16, ((i * 13) % 64 - 32) / 16))
    return X, X[:, 0] * X[:, 0] + X[:, 1]


@pytest.fixture
def model_factory(tmp_path):
    def make(**kwargs):
        options = dict(
            backend="rust",
            binary_operators=["+", "*"],
            unary_operators=[],
            parallelism="serial",
            deterministic=True,
            random_state=17,
            populations=1,
            population_size=24,
            tournament_selection_n=4,
            ncycles_per_iteration=16,
            niterations=0,
            maxsize=10,
            precision=64,
            model_selection="accuracy",
            progress=False,
            verbosity=0,
            output_directory=str(tmp_path),
        )
        return PySRRegressor(**(options | kwargs))

    return make


@pytest.fixture
def fitted(model_factory, data):
    X, y = data
    return model_factory(guesses=["x0*x0+x1"]).fit(X, y)


def test_simple_fit_predict_without_guesses(model_factory, data):
    X, _ = data
    y = X[:, 0] * X[:, 0]
    model = model_factory(
        binary_operators=["*"], niterations=12, ncycles_per_iteration=64
    ).fit(X, y)
    assert model.backend == "rust"
    assert model.rust_state_.total_evals > 0
    assert model.equations_.loss.min() < 1e-10
    np.testing.assert_allclose(model.predict(X), y, rtol=1e-5, atol=1e-5)


def test_equation_columns_and_dtypes_match_julia(model_factory, fitted, data):
    frame = fitted.equations_
    assert list(frame.columns) == [
        "complexity",
        "loss",
        "equation",
        "score",
        "sympy_format",
        "lambda_format",
    ]
    assert frame.complexity.dtype == np.dtype("int64")
    assert frame.loss.dtype == np.dtype("float64")
    assert frame.score.dtype == np.dtype("float64")
    assert pd.api.types.is_string_dtype(frame.equation.dtype)
    assert frame.sympy_format.dtype == np.dtype("object")
    assert frame.lambda_format.dtype == np.dtype("object")
    assert frame.complexity.is_monotonic_increasing
    assert np.isfinite(frame.loss).all()
    assert frame.loss.min() < 1e-12
    if not JULIA_DISABLED:
        X, y = data
        julia = model_factory(backend="julia", guesses=["x0*x0+x1"]).fit(X, y)
        assert list(frame.columns) == list(julia.equations_.columns)
        pd.testing.assert_series_equal(frame.dtypes, julia.equations_.dtypes)


def test_operator_complexity_changes_seeded_guess(model_factory, data):
    X, y = data
    options = dict(
        guesses=["x0*x0+x1"],
        should_simplify=False,
        should_optimize_constants=False,
    )
    default = model_factory(**options).fit(X, y)
    weighted = model_factory(**options, complexity_of_operators={"*": 3, "+": 2}).fit(
        X, y
    )
    assert default.get_best()["loss"] == weighted.get_best()["loss"] == 0.0
    assert default.get_best()["complexity"] == 5
    assert weighted.get_best()["complexity"] == 8
    assert default.rust_state_.pareto_fronts[0][-1][0] == 5
    assert weighted.rust_state_.pareto_fronts[0][-1][0] == 8


def test_nested_constraints_remove_nested_multiplication(model_factory, data):
    X, _ = data
    y = X[:, 0] ** 3
    options = dict(
        binary_operators=["*"],
        should_simplify=False,
        should_optimize_constants=False,
    )
    unrestricted = model_factory(**options, guesses=["x0*x0*x0"]).fit(X, y)
    constrained = model_factory(
        **options, nested_constraints={"*": {"*": 0}}, niterations=4
    ).fit(X, y)
    assert unrestricted.get_best()["loss"] == 0.0
    assert any(
        equation.count("*") > 1
        for _, _, equation in unrestricted.rust_state_.pareto_fronts[0]
    )
    front = constrained.rust_state_.pareto_fronts[0]
    assert front
    assert all(equation.count("*") <= 1 for _, _, equation in front)
    assert constrained.get_best()["loss"] > 0.0


def test_random_state_controls_deterministic_search_fronts(model_factory, data):
    X, _ = data
    y = np.sin(X[:, 0]) + X[:, 1] ** 3
    options = dict(
        niterations=2, ncycles_per_iteration=8, should_optimize_constants=False
    )
    first = model_factory(**options, random_state=17).fit(X, y)
    repeated = model_factory(**options, random_state=17).fit(X, y)
    different = model_factory(**options, random_state=23).fit(X, y)
    assert first.rust_state_.total_evals > 0
    assert first.rust_state_.pareto_fronts == repeated.rust_state_.pareto_fronts
    assert first.rust_state_.pareto_fronts != different.rust_state_.pareto_fronts


@pytest.mark.parametrize("maxsize", [7, 9])
def test_maxsize_caps_every_output_front(model_factory, data, maxsize):
    X, _ = data
    Y = np.column_stack((X[:, 0] ** 5 + X[:, 1], X[:, 1] ** 5 + X[:, 0]))
    model = model_factory(maxsize=maxsize, niterations=4).fit(X, Y)
    assert len(model.rust_state_.pareto_fronts) == 2
    assert len(model.equations_) == 2
    for front, frame in zip(model.rust_state_.pareto_fronts, model.equations_):
        assert front
        assert all(complexity <= maxsize for complexity, _, _ in front)
        assert (frame.complexity <= maxsize).all()


def test_generated_run_id_matches_julia_format(fitted):
    assert re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9a-zA-Z]{6}", fitted.run_id_)
    assert fitted.rust_state_.run_id == fitted.run_id_


@pytest.mark.parametrize("backend", ["rust", "julia"])
def test_random_state_without_deterministic_warns(model_factory, data, backend):
    if backend == "julia" and JULIA_DISABLED:
        pytest.skip("Julia is disabled by PYSR_BACKEND=rust")
    X, y = data
    message = (
        "Note: Setting `random_state` without also setting `deterministic=True` "
        "and `parallelism='serial'` will result in non-deterministic searches."
    )
    with pytest.warns(UserWarning, match=re.escape(message)):
        model_factory(backend=backend, deterministic=False, guesses=["x0*x0+x1"]).fit(
            X, y
        )


@pytest.mark.parametrize("backend", ["rust", "julia"])
def test_deterministic_multithreading_raises_shared_error(model_factory, data, backend):
    if backend == "julia" and JULIA_DISABLED:
        pytest.skip("Julia is disabled by PYSR_BACKEND=rust")
    X, y = data
    message = (
        "To ensure deterministic searches, you must set `parallelism='serial'`. "
        "Additionally, make sure to set `random_state` to a seed."
    )
    with pytest.raises(ValueError) as error:
        model_factory(backend=backend, parallelism="multithreading").fit(X, y)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "name",
    [
        "numprocs",
        "define_helper_functions",
        "loss_type",
        "return_state",
        "external_stop",
    ],
)
def test_non_pysr_kwargs_raise_normal_type_error(model_factory, name):
    with pytest.raises(TypeError) as error:
        model_factory(**{name: None})
    assert f"`{name}` is not a valid keyword argument for PySRRegressor." in str(
        error.value
    )


def test_sympy_and_numpy_lambda_exports(fitted, data):
    X, y = data
    expected = sympy.Symbol("x0") ** 2 + sympy.Symbol("x1")
    assert sympy.simplify(fitted.sympy() - expected) == 0
    assert callable(fitted.get_best()["lambda_format"])
    np.testing.assert_allclose(fitted.get_best()["lambda_format"](X), y)
    new_X = np.array([[-3.0, 0.25], [5.0, 2.0]])
    np.testing.assert_allclose(fitted.predict(new_X), new_X[:, 0] ** 2 + new_X[:, 1])


def test_requested_torch_export(model_factory, data):
    torch = pytest.importorskip("torch")
    X, y = data
    model = model_factory(guesses=["x0*x0+x1"], output_torch_format=True).fit(X, y)
    assert "torch_format" in model.equations_
    prediction = model.pytorch()(torch.tensor(X, dtype=torch.float64))
    np.testing.assert_allclose(prediction.detach().numpy(), y)


def test_requested_jax_export(model_factory, data):
    jnp = pytest.importorskip("jax.numpy")
    X, y = data
    model = model_factory(guesses=["x0*x0+x1"], output_jax_format=True).fit(X, y)
    assert "jax_format" in model.equations_
    exported = model.jax()
    prediction = exported["callable"](jnp.asarray(X), exported["parameters"])
    np.testing.assert_allclose(np.asarray(prediction), y, rtol=1e-6, atol=1e-6)


def test_multi_output_guesses(model_factory, data):
    X, y = data
    Y = np.column_stack((y, X[:, 0] * X[:, 1]))
    model = model_factory(guesses=[["x0*x0+x1"], ["x0*x1"]]).fit(X, Y)
    assert model.nout_ == 2
    assert len(model.equations_) == 2
    assert len(model.rust_state_.pareto_fronts) == 2
    assert all(frame.loss.min() < 1e-12 for frame in model.equations_)
    assert model.predict(X).shape == Y.shape
    np.testing.assert_allclose(model.predict(X), Y)
    for output in range(2):
        np.testing.assert_allclose(
            model.rust_state_.evaluate(X, output=output), Y[:, output]
        )


def test_units_penalize_incompatible_guesses(model_factory, data):
    X, _ = data
    y = X[:, 0].copy()
    options = dict(guesses=["x0"], dimensionless_constants_only=True)
    valid = model_factory(**options).fit(X, y, X_units=["m", "m"], y_units="m")
    invalid = model_factory(**options, dimensional_constraint_penalty=1000.0).fit(
        X, y, X_units=["m", "m"], y_units="s"
    )
    assert valid.equations_.loss.min() == 0.0
    assert invalid.equations_.loss.min() >= 1000.0
    np.testing.assert_allclose(valid.predict(X), y)


def test_named_guesses_are_used_without_search(model_factory, data):
    X, y = data
    model = model_factory(guesses=["a*a+b"]).fit(X, y, variable_names=["a", "b"])
    assert model.get_best()["loss"] == 0.0
    assert set(model.sympy().free_symbols) == {sympy.Symbol("a"), sympy.Symbol("b")}
    np.testing.assert_allclose(model.predict(X), y)


def test_template_predict_uses_state_evaluator(model_factory, data):
    X, _ = data
    y = X[:, 0] ** 2 + 2
    model = model_factory(
        expression_spec=TemplateExpressionSpec(
            combine="f(x0) + p[1]",
            expressions=["f"],
            variable_names=["x0", "x1"],
            parameters={"p": 1},
        ),
        guesses=[{"f": "#1*#1", "p": [2.0]}],
    ).fit(X, y)
    assert model.get_best()["loss"] == 0.0
    assert "f =" in model.get_best()["equation"]
    new_X = np.array([[-3.0, 0.25], [5.0, 2.0]])
    complexity = int(model.get_best()["complexity"])
    state_prediction = model.rust_state_.evaluate(
        new_X, output=0, complexity=complexity
    )
    np.testing.assert_array_equal(model.predict(new_X), state_prediction)
    np.testing.assert_allclose(state_prediction, new_X[:, 0] ** 2 + 2)
    with pytest.raises(ValueError, match="sympy export"):
        model.sympy()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(new_X), model.predict(new_X))


@pytest.mark.parametrize(
    "loss,weighted,expected",
    [
        ("L1DistLoss()", False, 0.0),
        ("loss(prediction, target) = (prediction - target)^2 + 0.25", False, 0.25),
        (
            "loss(prediction, target, weight) = weight*(prediction - target)^2 + 0.25",
            True,
            0.125,
        ),
    ],
)
def test_elementwise_loss_dsl(model_factory, data, loss, weighted, expected):
    X, y = data
    weights = np.full_like(y, 2) if weighted else None
    model = model_factory(guesses=["x0*x0+x1"], elementwise_loss=loss).fit(
        X, y, weights=weights
    )
    assert model.equations_.loss.min() == expected


def test_early_stop_dsl_reduces_evaluations(model_factory, data):
    X, y = data
    options = dict(guesses=["x0*x0+x1"], niterations=4)
    full = model_factory(**options).fit(X, y)
    stopped = model_factory(
        **options,
        early_stop_condition="stop(loss, complexity) = loss < 1e-12 && complexity <= 6",
    ).fit(X, y)
    assert stopped.equations_.loss.min() == 0.0
    assert stopped.rust_state_.total_evals < full.rust_state_.total_evals
    assert stopped.rust_state_.interrupted is False


def test_warm_start_resumes_search(model_factory, data):
    X, y = data
    model = model_factory(guesses=["x0*x0+x1"], niterations=1, warm_start=True).fit(
        X, y
    )
    evaluations = model.rust_state_.total_evals
    best_loss = model.equations_.loss.min()
    run_id = model.run_id_
    model.fit(X, y)
    assert model.rust_state_.total_evals > evaluations
    assert model.equations_.loss.min() <= best_loss
    assert model.run_id_ == run_id
    np.testing.assert_allclose(model.predict(X), y)


def test_regressor_pickle_preserves_predictions(fitted, data):
    X, _ = data
    restored = pickle.loads(pickle.dumps(fitted, protocol=5))
    assert restored.backend == "rust"
    assert restored.rust_state_.pareto_fronts == fitted.rust_state_.pareto_fronts
    np.testing.assert_allclose(restored.predict(X), fitted.predict(X), rtol=1e-12)
    np.testing.assert_allclose(
        restored.predict(X[::-1]), fitted.predict(X[::-1]), rtol=1e-12
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("loss_function", "objective(tree, data, options) = 0.0"),
        ("loss_function_expression", "objective(ex, data, options) = 0.0"),
        ("complexity_mapping", "complexity(ex) = 1"),
        ("turbo", True),
        ("bumper", True),
        ("autodiff_backend", "Zygote"),
        ("cluster_manager", "slurm"),
        ("procs", 2),
        ("heap_size_hint_in_bytes", 1024),
        ("worker_timeout", 1.0),
        ("worker_imports", ["LinearAlgebra"]),
        ("fast_cycle", True),
        ("update", True),
        ("logger_spec", "tensorboard"),
        ("type_spec", "custom"),
    ],
)
def test_julia_only_parameter_errors_clearly(model_factory, data, name, value):
    from pysr import TensorBoardLoggerSpec, TypeSpec

    if name == "logger_spec":
        value = TensorBoardLoggerSpec()
    elif name == "type_spec":
        value = TypeSpec(
            name="TestValue",
            fields={"x": "Float64"},
            sample="rng -> TestValue(1.0)",
            mutate="(value, rng) -> value",
        )
    X, y = data
    with pytest.raises(ValueError, match=f"(?i)(rust.*{name}|{name}.*rust)"):
        model_factory(**{name: value}).fit(X, y)


@pytest.mark.parametrize("precision,dtype", [(32, np.float32), (64, np.float64)])
def test_precision(model_factory, data, precision, dtype):
    X, y = (array.astype(dtype) for array in data)
    model = model_factory(precision=precision, guesses=["x0*x0+x1"]).fit(X, y)
    prediction = model.rust_state_.evaluate(X)
    assert prediction.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(prediction, y)
    np.testing.assert_allclose(model.predict(X), y, rtol=1e-6, atol=1e-6)


def test_csv_and_from_file(model_factory, data, tmp_path):
    X, y = data
    model = model_factory(guesses=["x0*x0+x1"], run_id="csv").fit(X, y)
    run_directory = Path(model.output_directory_) / model.run_id_
    csv = run_directory / "hall_of_fame.csv"
    frame = pd.read_csv(csv)
    assert list(frame.columns) == ["Complexity", "Loss", "Equation"]
    assert frame.Loss.min() == 0.0
    restored = PySRRegressor.from_file(run_directory=run_directory)
    assert restored.backend == "rust"
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    csv_only = tmp_path / "csv-only"
    csv_only.mkdir()
    shutil.copy2(csv, csv_only / csv.name)
    restored_csv = PySRRegressor.from_file(
        run_directory=csv_only,
        backend="rust",
        binary_operators=["+", "*"],
        unary_operators=[],
        n_features_in=2,
        model_selection="accuracy",
        output_directory=str(tmp_path),
        progress=False,
        verbosity=0,
    )
    np.testing.assert_allclose(restored_csv.predict(X), y)


def test_missing_rust_wheel_explains_installation(tmp_path):
    script = r"""
import sys
sys.modules["symbolic_regression_rs"] = None
import numpy as np
from pysr import PySRRegressor
X = np.arange(16, dtype=np.float64).reshape(-1, 1)
model = PySRRegressor(
    backend="rust", binary_operators=["+"], unary_operators=[],
    niterations=0, parallelism="serial", deterministic=True, random_state=17,
    output_directory=sys.argv[1], progress=False, verbosity=0,
)
try:
    model.fit(X, X[:, 0])
except ImportError as error:
    assert 'pip install "pysr[rust]"' in str(error), str(error)
    assert "https://pysr.ai/backend#rust-backend" in str(error), str(error)
else:
    raise AssertionError("Fitting without the Rust wheel must raise ImportError")
assert "juliacall" not in sys.modules
print("missing-rust-wheel-ok")
"""
    env = os.environ.copy()
    env["PYSR_BACKEND"] = "rust"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing-rust-wheel-ok" in result.stdout


@pytest.mark.parametrize("backend", ["invalid", "julia"])
def test_backend_validation_is_deferred_until_fit_in_rust_mode(tmp_path, backend):
    script = r"""
import sys
import numpy as np
from sklearn.base import clone
from pysr import PySRRegressor
backend = sys.argv[2]
model = PySRRegressor(
    backend=backend, output_directory=sys.argv[1], progress=False, verbosity=0
)
assert model.get_params()["backend"] == backend
assert clone(model).backend == backend
X = np.arange(16, dtype=np.float64).reshape(-1, 1)
expected_type = ValueError if backend == "invalid" else RuntimeError
expected_message = (
    "`backend` must be 'julia' or 'rust'." if backend == "invalid"
    else "unset PYSR_BACKEND"
)
try:
    model.fit(X, X[:, 0])
except expected_type as error:
    assert expected_message in str(error), str(error)
else:
    raise AssertionError("Backend validation must run during fit")
assert "juliacall" not in sys.modules
print("deferred-backend-validation-ok")
"""
    env = os.environ.copy()
    env["PYSR_BACKEND"] = "rust"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), backend],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deferred-backend-validation-ok" in result.stdout


def test_rust_environment_import_fit_and_julia_error(tmp_path):
    script = r"""
import sys
import numpy as np
import pysr
from pysr import PySRRegressor, TemplateExpressionSpec, ConstantMutation
from pysr._backend import DEFAULT_BACKEND, JULIA_DISABLED
assert DEFAULT_BACKEND == "rust"
assert JULIA_DISABLED
assert PySRRegressor().backend == "rust"
TemplateExpressionSpec(combine="f(x)", expressions=["f"], variable_names=["x"])
ConstantMutation()
assert "pysr.plugins" in sys.modules
X = np.arange(16, dtype=np.float64).reshape(-1, 1)
y = X[:, 0]
model = PySRRegressor(
    binary_operators=["+"], unary_operators=[], guesses=["x0"],
    niterations=0, populations=1, population_size=24, tournament_selection_n=4,
    parallelism="serial", deterministic=True, random_state=17, precision=64,
    output_directory=sys.argv[1], progress=False, verbosity=0,
).fit(X, y)
np.testing.assert_array_equal(model.predict(X), y)
try:
    PySRRegressor(backend="julia", output_directory=sys.argv[1],
                  progress=False, verbosity=0).fit(X, y)
except RuntimeError as error:
    assert "unset PYSR_BACKEND" in str(error)
else:
    raise AssertionError("Julia backend must reject PYSR_BACKEND=rust")
for name in ("jl", "SymbolicRegression"):
    try:
        getattr(pysr, name)
    except RuntimeError as error:
        assert "unset PYSR_BACKEND" in str(error)
    else:
        raise AssertionError(name + " must explain how to enable Julia")
assert not any(name == "juliacall" or name.startswith("juliacall.") for name in sys.modules)
print("rust-import-fit-ok")
"""
    env = os.environ.copy()
    env["PYSR_BACKEND"] = "rust"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rust-import-fit-ok" in result.stdout


def test_pickle_preserves_constant_division_arithmetic(model_factory):
    X = np.random.default_rng(19).normal(size=(64, 1))
    constant = 7.263271e-7
    y = (X[:, 0] * (X[:, 0] * constant)) / constant
    model = model_factory(
        binary_operators=["*", "/"],
        guesses=["(x0 * (x0 * 7.263271e-7)) / 7.263271e-7"],
        should_simplify=False,
        should_optimize_constants=False,
    ).fit(X, y)
    assert model.get_best()["loss"] == 0.0
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    np.testing.assert_allclose(model.predict(X), y, rtol=1e-12)
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-12)
    new_X = np.random.default_rng(23).normal(size=(64, 1))
    np.testing.assert_allclose(
        restored.predict(new_X), model.predict(new_X), rtol=1e-12
    )


def test_builtin_mutations_and_plugins_fit(model_factory, data):
    from pysr import AdaptiveParsimonyPlugin, ConstantMutation

    X, y = data
    model = model_factory(
        guesses=["x0*x0+x1"],
        niterations=1,
        mutations={ConstantMutation(): 1.0},
        plugins=[AdaptiveParsimonyPlugin()],
    ).fit(X, y)
    assert model.rust_state_.total_evals > 0
    np.testing.assert_allclose(model.predict(X), y)


def test_feature_selection_predict_and_lambda(model_factory):
    X = np.random.default_rng(17).normal(size=(128, 3))
    y = X[:, 2]
    model = model_factory(select_k_features=1, guesses=["x2"]).fit(X, y)
    np.testing.assert_array_equal(model.selection_mask_, [False, False, True])
    np.testing.assert_allclose(model.predict(X), y)
    np.testing.assert_allclose(
        model.get_best()["lambda_format"](X[:, model.selection_mask_]), y
    )


def test_multi_output_csv_from_file(model_factory, data, tmp_path):
    X, y = data
    Y = np.column_stack((y, X[:, 0] * X[:, 1]))
    model = model_factory(guesses=[["x0*x0+x1"], ["x0*x1"]], run_id="multi-csv").fit(
        X, Y
    )
    directory = Path(model.output_directory_) / model.run_id_
    csv_only = tmp_path / "multi-csv-only"
    csv_only.mkdir()
    for output in (1, 2):
        source = directory / f"hall_of_fame_output{output}.csv"
        assert pd.read_csv(source).Loss.min() == 0.0
        shutil.copy2(source, csv_only / source.name)
    restored = PySRRegressor.from_file(
        run_directory=csv_only,
        backend="rust",
        nout=2,
        binary_operators=["+", "*"],
        unary_operators=[],
        n_features_in=2,
        model_selection="accuracy",
        output_directory=str(tmp_path),
        progress=False,
        verbosity=0,
    )
    assert len(restored.equations_) == 2
    np.testing.assert_allclose(restored.predict(X), Y)


def test_rust_environment_star_import_and_cli_help():
    env = os.environ.copy()
    env["PYSR_BACKEND"] = "rust"
    script = "from pysr import *; import sys; assert 'juliacall' not in sys.modules"
    imported = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    help_result = subprocess.run(
        [sys.executable, "-m", "pysr", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "Usage:" in help_result.stdout


def test_single_output_units_list(model_factory, data):
    X, _ = data
    model = model_factory(guesses=["x0"]).fit(
        X, X[:, 0], X_units=["m", "m"], y_units=["m"]
    )
    np.testing.assert_allclose(model.predict(X), X[:, 0])


@pytest.mark.parametrize("batch_size", [-1, 0])
def test_batch_size_uses_frontend_normalization(model_factory, data, batch_size):
    X, y = data
    with pytest.warns(UserWarning, match="batch_size.*greater"):
        model = model_factory(
            guesses=["x0*x0+x1"], batching=True, batch_size=batch_size, niterations=1
        ).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y)


def test_constraints_use_frontend_validation(model_factory, data):
    X, y = data
    with pytest.raises(NotImplementedError, match="equal constraints"):
        model_factory(guesses=["x0*x0+x1"], constraints={"+": (1, 3)}).fit(X, y)
