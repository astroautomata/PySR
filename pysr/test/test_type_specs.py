import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import uuid
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from juliacall import JuliaError  # type: ignore

from pysr import PySRRegressor, TemplateExpressionSpec, TypeSpec, jl
from pysr.expression_specs import AbstractExpressionSpec, ExpressionSpec
from pysr.type_specs import (
    CallableJuliaExpression,
    compile_type_spec,
    compile_type_spec_runtime,
    load_type_spec_runtime,
    object_array_1d,
    object_array_2d,
    prepare_type_spec_fit_data,
    prepare_type_spec_prediction_data,
    type_spec_to_julia_array,
    type_spec_to_python_array,
    validate_type_spec_configuration,
    validate_type_spec_options,
    validate_type_spec_runtime,
)


def string_spec(**overrides):
    name = overrides.pop("name", "StringValue")
    return TypeSpec(
        name,
        fields=overrides.pop("fields", {"data": "String"}),
        sample=overrides.pop("sample", f'rng -> {name}(rand(rng, ("a", "b")))'),
        mutate=overrides.pop(
            "mutate",
            f'(rng, value, temperature) -> {name}(rand(rng, ("a", "b")))',
        ),
        **overrides,
    )


def vector_spec(**overrides):
    name = overrides.pop("name", "VectorValue")
    return TypeSpec(
        name,
        fields=overrides.pop("fields", {"data": "Vector{Float64}"}),
        sample=overrides.pop("sample", f"rng -> {name}([3.0, 4.0])"),
        scalar_constants=overrides.pop("scalar_constants", "value -> value.data"),
        with_scalar_constants=overrides.pop(
            "with_scalar_constants",
            f"(value, scalar_constants) -> {name}(collect(scalar_constants))",
        ),
        **overrides,
    )


def tiny_model(spec, *, parallelism="serial", **overrides):
    type_name = spec.name
    parameters = {
        "type_spec": spec,
        "operators": {1: [f"identity_{type_name}(x::{type_name}) = x"]},
        "elementwise_loss": (
            f"loss_{type_name}(x::{type_name}, y::{type_name})::Float64 = "
            "x == y ? 0.0 : 1.0"
        ),
        "niterations": 1,
        "ncycles_per_iteration": 2,
        "populations": 1,
        "population_size": 8,
        "tournament_selection_n": 3,
        "maxsize": 7,
        "parallelism": parallelism,
        "deterministic": parallelism == "serial",
        "random_state": 0 if parallelism == "serial" else None,
        "progress": False,
        "verbosity": 0,
        "temp_equation_file": True,
        "should_optimize_constants": False,
    }
    parameters.update(overrides)
    return PySRRegressor(**parameters)


def identity_template():
    return TemplateExpressionSpec(
        combine="f(x)",
        expressions=["f"],
        variable_names=["x"],
    )


def string_data(*, constant: bool = False):
    X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
    y = np.full(len(X), "a", dtype=object) if constant else X[:, 0].copy()
    return X, y


class TestTypeSpecs(unittest.TestCase):
    def test_serial_string_type_spec(self):
        X, y = string_data()
        model = tiny_model(string_spec())
        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertTrue(
            model._type_spec_runtime_definition_.module_name.startswith("_PySRConfig_")
        )
        self.assertIn("lambda_format", model.equations_.columns)

    def test_named_type_is_imported_into_main(self):
        suffix = uuid.uuid4().hex
        type_name = f"MainVisibleValue_{suffix}"
        runtime = load_type_spec_runtime(compile_type_spec(string_spec(name=type_name)))

        self.assertTrue(bool(jl.isdefined(jl.Main, jl.Symbol(type_name))))
        self.assertTrue(
            bool(
                jl.seval("(actual, expected) -> actual === expected")(
                    jl.getproperty(jl.Main, jl.Symbol(type_name)),
                    runtime.value_type,
                )
            )
        )

    def test_serial_vector_type_spec_with_scalar_constants(self):
        spec = vector_spec()
        X = np.empty((4, 1), dtype=object)
        X[:, 0] = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([7.0, 8.0]),
        ]
        model = tiny_model(spec)
        model.fit(X, X[:, 0].copy())

        self.assertEqual(
            [list(value) for value in model.predict(X)],
            [list(value) for value in X[:, 0]],
        )
        runtime = model._load_type_spec_runtime()
        value = runtime.module._convert_value([1.0, 2.0])
        packed, next_index = jl.seval("""
            function (value)
                DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
                buffer = fill(-99.0, 6)
                next_index = DE.pack_scalar_constants!(buffer, 3, value)
                return buffer, next_index
            end
            """)(value)
        self.assertEqual(list(packed), [-99.0, -99.0, 1.0, 2.0, -99.0, -99.0])
        self.assertEqual(int(next_index), 5)

    def test_multithreaded_type_spec(self):
        X, y = string_data()
        model = tiny_model(string_spec(), parallelism="multithreading")
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_configuration_modules_isolate_colliding_operator_names(self):
        X, y = string_data()
        type_name = "IsolatedConfigurationValue"
        operator_name = "colliding_identity"
        first = tiny_model(
            string_spec(name=type_name),
            operators={1: [f"{operator_name}(x::{type_name}) = x"]},
        )
        second = tiny_model(
            string_spec(name=type_name),
            operators={1: [f'{operator_name}(x::{type_name}) = {type_name}("a")']},
        )

        first.fit(X, y)
        second.fit(X, np.full(len(y), "a", dtype=object))

        np.testing.assert_array_equal(first.predict(X), y)
        np.testing.assert_array_equal(
            second.predict(X), np.full(len(y), "a", dtype=object)
        )
        self.assertIsNot(
            first._load_type_spec_runtime().configuration_module,
            second._load_type_spec_runtime().configuration_module,
        )

    def test_preamble_struct_has_one_identity_in_hooks_and_operators(self):
        type_name = "PreambleStructValue"
        spec = TypeSpec(
            type_name,
            fields={"data": "PrivatePayload"},
            preamble="struct PrivatePayload\n data::String\n end",
            sample=f'rng -> {type_name}(PrivatePayload("a"))',
            mutate="(rng, value, temperature) -> value",
        )
        definition = compile_type_spec_runtime(
            spec,
            {1: [f"x -> {type_name}(PrivatePayload(x.data.data))"]},
            elementwise_loss="(prediction, target) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
        )
        runtime = load_type_spec_runtime(definition)

        same_object = jl.seval("(a, b) -> a === b")
        self.assertTrue(
            bool(
                same_object(
                    jl.getproperty(runtime.module, jl.Symbol("PrivatePayload")),
                    jl.getproperty(
                        runtime.configuration_module, jl.Symbol("PrivatePayload")
                    ),
                )
            )
        )
        value = runtime.module._sample(runtime.module.Random.default_rng())
        result = runtime.operator_functions[1][0](value)
        self.assertTrue(bool(jl.isa(result.data, runtime.module.PrivatePayload)))

    def test_repeated_runtime_load_does_not_mutate_child_module(self):
        spec = string_spec(name="RepeatedLoadValue")
        definition = compile_type_spec_runtime(
            spec,
            {1: ["x -> x"]},
            elementwise_loss="(prediction, target) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
            expression_spec="x -> x",
            expression_spec_function_selector="spec -> spec",
        )
        first = load_type_spec_runtime(definition)
        names_before = tuple(
            str(name)
            for name in jl.names(first.configuration_module, all=True, imported=True)
        )

        second = load_type_spec_runtime(first.definition)
        names_after = tuple(
            str(name)
            for name in jl.names(second.configuration_module, all=True, imported=True)
        )

        self.assertEqual(names_after, names_before)
        self.assertTrue(
            bool(
                jl.seval("(a, b) -> a === b")(
                    first.expression_spec,
                    second.expression_spec,
                )
            )
        )

    def test_failed_configuration_source_remains_isolated(self):
        suffix = uuid.uuid4().hex
        type_name = f"FailedConfigurationValue_{suffix}"
        operator_name = f"isolated_operator_{suffix}"
        X, y = string_data()
        fitted = tiny_model(
            string_spec(name=type_name),
            operators={1: [f"{operator_name}(x::{type_name}) = x"]},
        )
        fitted.fit(X, y)
        broken = tiny_model(
            string_spec(name=type_name),
            operators={
                1: [
                    f"""
                    begin
                        {operator_name}(x::{type_name}) = {type_name}("a")
                        error("isolated source failure")
                    end
                    """
                ]
            },
        )

        with self.assertRaisesRegex(JuliaError, "isolated source failure"):
            broken.fit(X, y)

        np.testing.assert_array_equal(fitted.predict(X), y)
        self.assertFalse(bool(jl.isdefined(jl.Main, jl.Symbol(operator_name))))

    def test_failed_runtime_install_can_retry_same_definition(self):
        suffix = uuid.uuid4().hex
        flag = f"_pysr_retry_install_{suffix}"
        type_name = f"RetryInstallValue_{suffix}"
        operator_name = f"retry_identity_{suffix}"
        jl.seval(f"global {flag} = Ref(true)")
        definition = compile_type_spec_runtime(
            string_spec(name=type_name),
            {1: [f"{operator_name}(x::{type_name})::{type_name} = x"]},
            elementwise_loss=f"(x::{type_name}, y::{type_name}) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=f"""
                begin
                    Main.{flag}[] && error("retry source failure")
                    (loss, complexity) -> false
                end
                """,
        )

        with self.assertRaisesRegex(JuliaError, "retry source failure"):
            load_type_spec_runtime(definition)

        jl.seval(f"{flag}[] = false")
        runtime = load_type_spec_runtime(definition)
        self.assertEqual(runtime.operator_names, {1: (operator_name,)})
        self.assertIsNotNone(runtime.early_stop_condition)

    def test_failed_type_install_can_retry_same_definition(self):
        suffix = uuid.uuid4().hex
        flag = f"_pysr_retry_type_install_{suffix}"
        type_name = f"RetryTypeValue_{suffix}"
        operator_name = f"retry_type_identity_{suffix}"
        jl.seval(f"global {flag} = Ref(true)")
        definition = compile_type_spec_runtime(
            string_spec(
                name=type_name,
                preamble=f'Main.{flag}[] && error("retry type failure")',
            ),
            {1: [f"{operator_name}(x::{type_name})::{type_name} = x"]},
            elementwise_loss=f"(x::{type_name}, y::{type_name}) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
        )

        with self.assertRaisesRegex(JuliaError, "retry type failure"):
            load_type_spec_runtime(definition)

        jl.seval(f"{flag}[] = false")
        runtime = load_type_spec_runtime(definition)
        self.assertEqual(runtime.operator_names, {1: (operator_name,)})
        self.assertTrue(
            bool(jl.isdefined(runtime.module, jl.Symbol(type_name))),
        )

    def test_equivalent_runtime_definition_is_evaluated_once(self):
        suffix = uuid.uuid4().hex
        counter = f"_pysr_definition_count_{suffix}"
        type_name = f"ReusedValue_{suffix}"
        operator = f"reused_identity_{suffix}"
        jl.seval(f"global {counter} = 0")
        source = f"""
        begin
            Main.{counter} += 1
            {operator}(x::{type_name}) = x
        end
        """
        first = tiny_model(string_spec(name=type_name), operators={1: [source]})
        second = tiny_model(string_spec(name=type_name), operators={1: [source]})
        X, y = string_data()

        first.fit(X, y)
        second.fit(X, y)
        restored = pickle.loads(pickle.dumps(second))
        _ = restored.julia_state_
        _ = restored.julia_options_

        self.assertEqual(int(jl.seval(counter)), 1)
        self.assertTrue(
            bool(
                jl.seval("(a, b) -> a === b")(
                    first._load_type_spec_runtime().configuration_module,
                    second._load_type_spec_runtime().configuration_module,
                )
            )
        )

    def test_restored_model_rebuilds_callable_columns(self):
        X, y = string_data()
        model = tiny_model(string_spec(name="RestoredColumnValue"))
        model.fit(X, y)

        restored = pickle.loads(pickle.dumps(model))
        self.assertNotIn("lambda_format", restored.equations_.columns)

        best = restored.get_best()
        self.assertIn("lambda_format", restored.equations_.columns)
        np.testing.assert_array_equal(best["lambda_format"](X), model.predict(X))

    def test_failed_warm_start_restores_fitted_metadata_and_checkpoint(self):
        X, y = string_data()
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(
                string_spec(),
                temp_equation_file=False,
                output_directory=directory,
                run_id="warm-start-rollback",
            )
            model.fit(X, y)
            definition = model._type_spec_runtime_definition_
            state = model.julia_state_stream_.copy()
            equations = model.equations_.copy(deep=True)
            feature_names = model.feature_names_in_.copy()
            checkpoint = Path(model.get_pkl_filename())
            checkpoint_contents = checkpoint.read_bytes()

            model.set_params(
                warm_start=True,
                complexity_mapping="expression -> 1",
            )
            with self.assertRaisesRegex(
                ValueError, "Cannot warm-start after changing.*configuration"
            ):
                model.fit(X, y, variable_names=["candidate"])

            self.assertEqual(model._type_spec_runtime_definition_, definition)
            np.testing.assert_array_equal(model.julia_state_stream_, state)
            pd.testing.assert_frame_equal(model.equations_, equations)
            np.testing.assert_array_equal(model.feature_names_in_, feature_names)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_contents)

            def fail_after_mutation(*args, **kwargs):
                model.equations_.iloc[0, 0] += 100
                model.feature_names_in_[0] = "mutated"
                raise RuntimeError("search failed")

            model.set_params(complexity_mapping=None)
            with (
                patch.object(model, "_run", side_effect=fail_after_mutation),
                self.assertRaisesRegex(RuntimeError, "search failed"),
            ):
                model.fit(X, y, variable_names=["candidate"])

            self.assertEqual(model._type_spec_runtime_definition_, definition)
            np.testing.assert_array_equal(model.julia_state_stream_, state)
            pd.testing.assert_frame_equal(model.equations_, equations)
            np.testing.assert_array_equal(model.feature_names_in_, feature_names)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_contents)

    def test_warm_start_rejects_loss_type_change(self):
        X, y = string_data()
        type_name = "WarmStartLossTypeValue"
        objective = """
            function warm_start_objective(
                tree, dataset::Dataset, options, idx=nothing
            )::Float64
                _, complete = eval_tree_array(tree, dataset.X, options)
                return complete ? 0.0 : Inf
            end
        """
        model = tiny_model(
            string_spec(name=type_name, loss_type="Float64"),
            elementwise_loss=None,
            loss_function=objective,
        )
        model.fit(X, y)
        model.set_params(
            warm_start=True,
            type_spec=string_spec(name=type_name, loss_type="Float32"),
        )

        with self.assertRaisesRegex(
            ValueError, "Cannot warm-start after changing.*configuration"
        ):
            model.fit(X, y)

    def test_pre_search_checkpoint_can_resume_as_a_fresh_fit(self):
        X, y = string_data()
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(
                string_spec(name="PreSearchCheckpointValue"),
                temp_equation_file=False,
                output_directory=directory,
                run_id="pre-search-checkpoint",
            )
            with (
                patch.object(
                    PySRRegressor,
                    "_run",
                    side_effect=RuntimeError("search failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "search failed"),
            ):
                model.fit(X, y)

            checkpoint = Path(model.get_pkl_filename())
            with checkpoint.open("rb") as checkpoint_file:
                restored = pickle.load(checkpoint_file)
            self.assertIsNone(restored.equations_)
            self.assertFalse(hasattr(restored, "_type_spec_runtime_definition_"))

            restored.set_params(
                temp_equation_file=True,
                output_directory=None,
                run_id=None,
            )
            restored.fit(X, y)
            np.testing.assert_array_equal(restored.predict(X), y)

    def test_callable_supervised_loss_is_supported(self):
        type_name = "CallableLossValue"
        loss_name = "CallableTypeSpecLoss"
        X, y = string_data()
        model = tiny_model(
            string_spec(name=type_name),
            elementwise_loss=f"""
                begin
                    struct {loss_name} <: SymbolicRegression.SupervisedLoss end
                    (::{loss_name})(
                        prediction::{type_name}, target::{type_name}
                    )::Float64 = prediction == target ? 0.0 : 1.0
                    {loss_name}()
                end
            """,
        )

        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_numeric_early_stop_condition_remains_numeric(self):
        model = tiny_model(
            string_spec(name="NumericEarlyStopValue"),
            early_stop_condition=1e-4,
        )
        definition = model._compile_type_spec_runtime(model._operators_from_params())
        runtime = load_type_spec_runtime(definition)

        self.assertEqual(float(runtime.early_stop_condition), 1e-4)

    def test_pickle_restores_runtime_before_deserializing_in_fresh_process(self):
        X, y = string_data()
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(
                string_spec(name="CheckpointStringValue"),
                complexity_mapping="checkpoint_complexity(expression) = 1",
                early_stop_condition=(
                    "checkpoint_early_stop(loss, complexity) = false"
                ),
                temp_equation_file=False,
                output_directory=directory,
                run_id="typespec-checkpoint",
                delete_tempfiles=False,
            )
            model.fit(X, y)
            type_module_name = (
                model._type_spec_runtime_definition_.type_spec.module_name
            )
            run_directory = Path(directory) / "typespec-checkpoint"
            module_name = model._type_spec_runtime_definition_.module_name
            code = f"""
import json
import numpy as np
from pysr import PySRRegressor, jl
type_module_name = {type_module_name!r}
module_name = {module_name!r}
assert not bool(jl.isdefined(jl.Main, jl.Symbol(type_module_name)))
model = PySRRegressor.from_file(run_directory={str(run_directory)!r})
X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
runtime = model._load_type_spec_runtime()
print(json.dumps({{
    "prediction": model.predict(X).tolist(),
    "loaded": bool(jl.isdefined(runtime.module, jl.Symbol(module_name))),
    "complexity": str(jl.Base.nameof(runtime.complexity_mapping)),
    "early_stop": str(jl.Base.nameof(runtime.early_stop_condition)),
}}))
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            payload,
            {
                "prediction": y.tolist(),
                "loaded": True,
                "complexity": "checkpoint_complexity",
                "early_stop": "checkpoint_early_stop",
            },
        )

    def test_unsupported_configurations_fail_before_runtime_loading(self):
        X, y = string_data()
        cases = (
            (tiny_model(string_spec(), turbo=True), ValueError, "turbo"),
            (tiny_model(string_spec(), bumper=True), ValueError, "bumper"),
            (
                tiny_model(string_spec(), autodiff_backend="Zygote"),
                ValueError,
                "autodiff_backend",
            ),
            (
                tiny_model(string_spec(), output_jax_format=True),
                ValueError,
                "output_jax_format",
            ),
            (
                tiny_model(string_spec(), output_torch_format=True),
                ValueError,
                "output_torch_format",
            ),
            (
                tiny_model(string_spec(), extra_sympy_mappings={"f": lambda x: x}),
                ValueError,
                "extra_sympy_mappings",
            ),
            (
                tiny_model(
                    string_spec(), extra_jax_mappings={lambda x: x: "jnp.asarray"}
                ),
                ValueError,
                "extra_jax_mappings",
            ),
            (
                tiny_model(
                    string_spec(), extra_torch_mappings={lambda x: x: lambda x: x}
                ),
                ValueError,
                "extra_torch_mappings",
            ),
            (
                tiny_model(string_spec(), binary_operators=["+"]),
                ValueError,
                "binary_operators",
            ),
        )
        for model, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    model.fit(X, y)
                self.assertFalse(hasattr(model, "_type_spec_runtime_definition_"))

    def test_custom_expression_spec_requires_runtime_source(self):
        class MissingSourceSpec(ExpressionSpec):
            pass

        X, y = string_data()
        model = tiny_model(
            string_spec(name="MissingExpressionSourceValue"),
            expression_spec=MissingSourceSpec(),
        )

        with self.assertRaisesRegex(ValueError, "does not provide Julia source"):
            model.fit(X, y)
        self.assertFalse(hasattr(model, "_type_spec_runtime_definition_"))

    def test_operator_names_drive_constraint_keys(self):
        type_name = "NamedOperatorValue"
        operator_name = "named_identity"
        X, y = string_data()
        model = tiny_model(
            string_spec(name=type_name),
            operators={1: [f"{operator_name}(x::{type_name})::{type_name} = x"]},
            constraints={operator_name: 1},
        )

        model.fit(X, y)

        runtime = model._load_type_spec_runtime()
        self.assertEqual(runtime.operator_names, {1: (operator_name,)})

    def test_template_source_hook_runs_once_per_fit(self):
        sources = []

        class SourceExpressionSpec(AbstractExpressionSpec):
            @property
            def supports_type_spec(self):
                return True

            def julia_expression_spec(self):
                return ExpressionSpec().julia_expression_spec()

            def create_exports(self, *args, **kwargs):
                return ExpressionSpec().create_exports(*args, **kwargs)

            def _julia_expression_spec_source(self, *, prototype=None):
                sources.append(prototype)
                return f"""
                    @template_spec(
                        expressions=(f,), prototype={prototype}
                    ) do x
                        f(x)
                    end
                """

            def _julia_expression_spec_function_selector(self):
                return "spec -> spec.structure.combine"

        X, y = string_data()
        model = tiny_model(
            string_spec(name="SingleTemplateSourceValue"),
            expression_spec=SourceExpressionSpec(),
        )
        model.fit(X, y)

        self.assertEqual(
            sources,
            ["SymbolicRegression.init_value(SingleTemplateSourceValue)"],
        )
        np.testing.assert_array_equal(model.predict(X), y)

    def test_template_type_spec_parallelism_matrix(self):
        for parallelism in ("serial", "multithreading", "multiprocessing"):
            suffix = parallelism.title()
            type_name = f"Template{suffix}Value"
            X, y = string_data(constant=True)
            model = tiny_model(
                string_spec(
                    name=type_name,
                    sample=f'rng -> {type_name}("a")',
                    mutate="(rng, value, temperature) -> value",
                ),
                expression_spec=identity_template(),
                parallelism=parallelism,
                procs=2 if parallelism == "multiprocessing" else None,
            )

            with self.subTest(parallelism=parallelism):
                model.fit(X, y)
                np.testing.assert_array_equal(model.predict(X), y)

    def test_same_runtime_warm_start_continues(self):
        X, y = string_data()
        model = tiny_model(
            string_spec(name="WarmStartValue"),
            expression_spec=identity_template(),
        )
        model.fit(X, y)
        fingerprint = model._type_spec_runtime_definition_.fingerprint
        model.set_params(warm_start=True)

        model.fit(X, y)

        self.assertEqual(
            model._type_spec_runtime_definition_.fingerprint,
            fingerprint,
        )
        np.testing.assert_array_equal(model.predict(X), y)

    def test_malformed_hooks_fail_with_hook_names(self):
        cases = (
            (
                vector_spec(
                    name="BadScalarConstantsValue",
                    scalar_constants="value -> 1.0",
                ),
                "scalar_constants.*AbstractVector",
            ),
            (
                string_spec(
                    name="BadPredicateValue",
                    is_valid="value -> 1",
                ),
                "is_valid.*Bool",
            ),
            (
                string_spec(name="BadInitValue", init='() -> ""'),
                "init.*BadInitValue",
            ),
            (
                vector_spec(
                    name="BadReconstructionValue",
                    with_scalar_constants="(value, constants) -> constants",
                ),
                "with_scalar_constants.*BadReconstructionValue",
            ),
            (
                vector_spec(
                    name="BadScalarConstantElementValue",
                    scalar_constants="value -> [1, 2]",
                    with_scalar_constants=(
                        "(value, constants) -> "
                        "BadScalarConstantElementValue(collect(constants))"
                    ),
                ),
                "scalar_constants.*concrete `AbstractFloat`",
            ),
            (
                vector_spec(
                    name="NonRoundTripValue",
                    with_scalar_constants=(
                        "(value, constants) -> "
                        "NonRoundTripValue(reverse(collect(constants)))"
                    ),
                ),
                "with_scalar_constants.*preserve",
            ),
            (
                vector_spec(name="BadStringValue", string="value -> 1"),
                "string.*AbstractString",
            ),
            (
                string_spec(
                    name="BadSampleArityValue",
                    sample='() -> BadSampleArityValue("a")',
                ),
                "sample",
            ),
            (
                string_spec(
                    name="RejectedSampleValue",
                    is_valid="value -> false",
                ),
                "sample.*invalid value",
            ),
        )
        for spec, message in cases:
            with self.subTest(spec=spec.name):
                runtime = load_type_spec_runtime(compile_type_spec(spec))
                with self.assertRaisesRegex(ValueError, message):
                    validate_type_spec_runtime(runtime)

    def test_first_fit_validates_type_spec_hooks(self):
        X, y = string_data()
        model = tiny_model(string_spec(name="FitValidationValue", init='() -> ""'))

        with self.assertRaisesRegex(ValueError, "init.*FitValidationValue"):
            model.fit(X, y)
        self.assertFalse(hasattr(model, "_type_spec_runtime_definition_"))

    def test_operator_and_elementwise_loss_are_invoked_during_validation(self):
        type_name = "RuntimeInvocationValue"
        cases = {
            "operator": {
                "operator": (
                    f"function exploding_operator(x::{type_name})::{type_name}\n"
                    ' x.data == "b" && error("operator was invoked")\n'
                    " return x\nend"
                ),
                "loss": (
                    f"identity_loss(x::{type_name}, y::{type_name})::Float64 = 0.0"
                ),
                "message": "operator was invoked",
            },
            "elementwise loss": {
                "operator": (f"identity_runtime(x::{type_name})::{type_name} = x"),
                "loss": (
                    f"function exploding_loss(x::{type_name}, y::{type_name})::Float64\n"
                    ' x.data == y.data && error("loss was invoked")\n'
                    " return 0.0\nend"
                ),
                "message": "loss was invoked",
            },
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                definition = compile_type_spec_runtime(
                    string_spec(name=type_name),
                    {1: [case["operator"]]},
                    elementwise_loss=case["loss"],
                    loss_function=None,
                    loss_function_expression=None,
                    complexity_mapping=None,
                    early_stop_condition=None,
                )
                runtime = load_type_spec_runtime(definition)
                with self.assertRaisesRegex(ValueError, case["message"]):
                    validate_type_spec_runtime(runtime)

    def test_module_loading_has_no_behavioral_hook_side_effects(self):
        suffix = uuid.uuid4().hex
        init_count = f"_pysr_init_count_{suffix}"
        mutate_count = f"_pysr_mutate_count_{suffix}"
        type_name = f"SideEffectValue_{suffix}"
        jl.seval(f"global {init_count} = 0; global {mutate_count} = 0")
        spec = string_spec(
            name=type_name,
            init=(f"() -> begin Main.{init_count} += 1; " f'{type_name}("a") end'),
            mutate=(
                f"(rng, value, temperature) -> begin Main.{mutate_count} += 1; "
                "value end"
            ),
        )

        load_type_spec_runtime(compile_type_spec(spec))
        self.assertEqual(int(jl.seval(init_count)), 0)
        self.assertEqual(int(jl.seval(mutate_count)), 0)

        definition = compile_type_spec_runtime(
            spec,
            {1: [f"side_effect_identity(x::{type_name}) = x"]},
            elementwise_loss=(
                f"side_effect_loss(x::{type_name}, y::{type_name})::Float64 = 0.0"
            ),
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
        )
        runtime = load_type_spec_runtime(definition)
        validate_type_spec_options(
            runtime,
            runtime.operator_functions,
            runtime.elementwise_loss,
        )
        self.assertEqual(int(jl.seval(init_count)), 0)
        self.assertEqual(int(jl.seval(mutate_count)), 0)
        validate_type_spec_runtime(runtime)
        self.assertEqual(int(jl.seval(init_count)), 1)
        self.assertEqual(int(jl.seval(mutate_count)), 1)

    def test_object_array_shapes_and_logical_conversion(self):
        one_dimensional = object_array_1d(value for value in ([1, 2], [3, 4]))
        self.assertEqual(one_dimensional.shape, (2,))
        self.assertEqual(one_dimensional.tolist(), [[1, 2], [3, 4]])
        two_dimensional = object_array_2d(
            row for row in (([1, 2], [3, 4]), ([5, 6], [7, 8]))
        )
        self.assertEqual(two_dimensional.shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "2D array"):
            object_array_2d(["a", "b"])
        with self.assertRaisesRegex(ValueError, "same number"):
            object_array_2d([[1], [2, 3]])

        spec = TypeSpec(
            "PairValue",
            fields={"number": "Float64", "label": "String"},
            sample='rng -> PairValue(0.0, "")',
            mutate="(rng, value, temperature) -> value",
        )
        runtime = load_type_spec_runtime(compile_type_spec(spec))
        values = object_array_1d([(1.0, "one"), (2.0, "two")])
        converted = type_spec_to_julia_array(runtime, values)
        self.assertEqual(
            type_spec_to_python_array(runtime, converted).tolist(), values.tolist()
        )

    def test_derived_scalar_constant_packing_honors_offsets(self):
        runtime = load_type_spec_runtime(
            compile_type_spec(vector_spec(name="OffsetVectorValue"))
        )
        first = runtime.module._convert_value([1.0, 2.0])
        second = runtime.module._convert_value([3.0, 4.0])
        packed, next_index, unpacked = jl.seval("""
            function (first, second)
                DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
                packed = fill(-99.0, 8)
                idx = DE.pack_scalar_constants!(packed, 3, first)
                idx = DE.pack_scalar_constants!(packed, idx, second)
                idx, rebuilt_first =
                    DE.unpack_scalar_constants(packed, 3, first)
                idx, rebuilt_second =
                    DE.unpack_scalar_constants(packed, idx, second)
                return packed, idx, (rebuilt_first, rebuilt_second)
            end
            """)(first, second)
        self.assertEqual(
            list(packed),
            [-99.0, -99.0, 1.0, 2.0, 3.0, 4.0, -99.0, -99.0],
        )
        self.assertEqual(int(next_index), 7)
        self.assertEqual(
            [list(value.data) for value in unpacked],
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_union_payload_scalar_constant_hooks(self):
        spec = TypeSpec(
            "TensorValue",
            fields={"data": "Union{Float64, Vector{Float64}, Matrix{Float64}}"},
            sample="rng -> TensorValue(randn(rng, 2, 2))",
            scalar_constants="""
                function scalar_constants(value)
                    return value.data isa Float64 ? [value.data] : vec(value.data)
                end
            """,
            with_scalar_constants="""
                function with_scalar_constants(value, scalar_constants)
                    data = value.data isa Float64 ? scalar_constants[1] :
                        reshape(collect(scalar_constants), size(value.data))
                    return TensorValue(data)
                end
            """,
        )
        runtime = load_type_spec_runtime(compile_type_spec(spec))
        value = runtime.module._convert_value(np.eye(2))
        self.assertEqual(
            list(runtime.module._scalar_constants(value)),
            [1.0, 0.0, 0.0, 1.0],
        )

    def test_immutable_scalar_constant_view_validates(self):
        spec = vector_spec(
            name="ImmutableScalarConstantsValue",
            scalar_constants=(
                "value -> range(value.data[1], value.data[end]; "
                "length=length(value.data))"
            ),
            with_scalar_constants=(
                "(value, scalar_constants) -> "
                "ImmutableScalarConstantsValue(collect(scalar_constants))"
            ),
        )
        model = tiny_model(spec)
        runtime = model._load_type_spec_runtime(
            for_fit=True,
            operators=model._operators_from_params(),
        )
        self.assertEqual(runtime.spec.name, "ImmutableScalarConstantsValue")

    def test_variable_length_vector_constants(self):
        spec = TypeSpec(
            "VariableVector",
            fields={"data": "Vector{Float64}"},
            sample="rng -> VariableVector(randn(rng, rand(rng, 1:5)))",
            mutate="""
                function mutate_vector(rng, value, temperature)
                    if rand(rng) < 0.2
                        return VariableVector(randn(rng, rand(rng, 1:5)))
                    end
                    data = copy(value.data)
                    data[rand(rng, eachindex(data))] += temperature * randn(rng)
                    return VariableVector(data)
                end
            """,
            scalar_constants="value -> value.data",
            with_scalar_constants=(
                "(value, scalar_constants) -> "
                "VariableVector(collect(scalar_constants))"
            ),
        )
        rng = np.random.default_rng(0)
        values = [rng.normal(size=2) for _ in range(64)]
        X = pd.DataFrame({"x": values})
        prefix = np.array([1.5])
        suffix = np.array([-0.2, -3.0, 0.1])
        y = np.empty(len(values), dtype=object)
        y[:] = [np.concatenate((prefix, value, suffix)) for value in values]
        model = tiny_model(
            spec,
            expression_spec=identity_template(),
            operators={
                2: ["concat_vectors(a, b) = " "VariableVector(vcat(a.data, b.data))"]
            },
            elementwise_loss="""
                function vector_loss(a, b)::Float64
                    return length(a.data) == length(b.data) ?
                        sum(abs2, a.data - b.data) : 1.0e6
                end
            """,
            niterations=60,
            ncycles_per_iteration=100,
            populations=4,
            population_size=50,
            tournament_selection_n=10,
            maxsize=7,
            early_stop_condition=(
                "(loss, complexity) -> loss < 1.0e-8 && complexity == 5"
            ),
            should_optimize_constants=True,
        )

        model.fit(X, y)

        exact = model.equations_.query("complexity == 5").sort_values("loss").iloc[0]
        self.assertLess(exact.loss, 1.0e-8)
        constants = jl.seval("""
            function (expression)
                DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
                tree = DE.get_tree(expression)
                nodes = DE.filter_map(
                    node -> node.degree == 0 && node.constant,
                    identity,
                    tree,
                    typeof(tree),
                )
                return map(node -> node.val.data, nodes)
            end
            """)(exact.julia_expression)
        self.assertEqual([len(value) for value in constants], [1, 3])
        np.testing.assert_allclose(constants[0], prefix, atol=1.0e-6)
        np.testing.assert_allclose(constants[1], suffix, atol=1.0e-6)

    def test_multi_field_fit_predicts_tuples(self):
        spec = TypeSpec(
            "FitPairValue",
            fields={"number": "Float64", "label": "String"},
            sample='rng -> FitPairValue(0.0, "")',
            mutate="(rng, value, temperature) -> value",
        )
        pairs = [(1.0, "one"), (2.0, "two"), (3.0, "three"), (4.0, "four")]
        X = pd.DataFrame({"x": pairs})
        y = pd.Series(pairs, dtype=object)
        model = tiny_model(spec)

        model.fit(X, y)

        self.assertEqual(model.predict(X).tolist(), pairs)

    def test_invalid_expression_evaluation_reports_the_rejected_value(self):
        type_name = "InvalidEvaluationValue"
        spec = TypeSpec(
            type_name,
            fields={"data": "Float64"},
            sample=f"rng -> {type_name}(1.0)",
            scalar_constants="value -> (value.data,)",
            with_scalar_constants=f"(value, constants) -> {type_name}(constants[1])",
        )
        definition = compile_type_spec_runtime(
            spec,
            {1: [f"invalid_operator(x::{type_name})::{type_name} = {type_name}(Inf)"]},
            elementwise_loss="(prediction, target) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
        )
        runtime = load_type_spec_runtime(definition)
        expression = jl.seval("""
            (value_type, operator) -> begin
                operators = OperatorEnum(1 => (operator,))
                node = Node{value_type}(op=1, l=Node{value_type}(feature=1))
                Expression(node; operators, variable_names=["x1"])
            end
            """)(runtime.value_type, runtime.operator_functions[1][0])
        X = object_array_2d([[1.0], [2.0]])

        with self.assertRaisesRegex(ValueError, "`is_valid` rejected"):
            CallableJuliaExpression(expression, runtime)(X)

    def test_type_spec_variable_name_count_is_validated(self):
        X, y = string_data()
        model = tiny_model(string_spec(name="WrongVariableCountValue"))

        with self.assertRaisesRegex(
            ValueError, "variable_names.*one name per TypeSpec feature"
        ):
            model.fit(X, y, variable_names=["first", "extra"])

    def test_rejects_empty_feature_axis(self):
        model = tiny_model(string_spec())

        with self.assertRaisesRegex(ValueError, "at least one feature"):
            model.fit(np.empty((2, 0), dtype=object), np.array(["a", "b"]))

    def test_csv_only_loading_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n1,0.0,x0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint.pkl"):
                PySRRegressor.from_file(
                    run_directory=directory,
                    type_spec=string_spec(),
                    operators={1: ["identity_value(x::StringValue) = x"]},
                    n_features_in=1,
                )

    def test_guess_seeds_the_search_with_a_custom_constant(self):
        type_name = "GuessVectorValue"
        model = tiny_model(
            vector_spec(name=type_name),
            operators={
                2: [
                    f"add_vectors(a::{type_name}, b::{type_name}) = "
                    f"{type_name}(a.data + b.data)"
                ]
            },
            elementwise_loss=(
                f"vector_loss(a::{type_name}, b::{type_name})::Float64 = "
                "sum(abs2, a.data - b.data)"
            ),
            guesses=[f"add_vectors(x0, {type_name}([1.5, -2.5]))"],
        )
        X = np.empty((4, 1), dtype=object)
        X[:, 0] = [np.array([1.0, 2.0])] * 4
        y = np.empty(4, dtype=object)
        y[:] = [np.array([2.5, -0.5])] * 4

        model.fit(X, y)

        self.assertEqual(model.equations_.loss.min(), 0.0)
        self.assertIn("[1.5, -2.5]", " ".join(model.equations_.equation))

    def test_template_guess_seeds_each_expression(self):
        type_name = "GuessTemplateValue"
        model = tiny_model(
            string_spec(name=type_name),
            expression_spec=TemplateExpressionSpec(
                combine="f(x)",
                expressions=["f"],
                variable_names=["x"],
            ),
            operators={
                2: [
                    f"concat(a::{type_name}, b::{type_name}) = "
                    f"{type_name}(a.data * b.data)"
                ]
            },
            elementwise_loss=(
                f"string_loss(a::{type_name}, b::{type_name})::Float64 = "
                "a.data == b.data ? 0.0 : 1.0"
            ),
            guesses=[{"f": 'concat(#1, "!")'}],
        )
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a!", "b!", "a!", "b!"], dtype=object)

        model.fit(X, y, variable_names=["x"])

        self.assertEqual(model.equations_.loss.min(), 0.0)
        self.assertIn('concat(#1, "!")', " ".join(model.equations_.equation))

    def test_guess_rejects_a_constant_that_is_not_the_custom_type(self):
        type_name = "GuessConversionValue"
        model = tiny_model(
            vector_spec(name=type_name),
            guesses=[f"identity_{type_name}(2.0)"],
        )
        X = np.empty((4, 1), dtype=object)
        X[:, 0] = [np.array([1.0, 2.0])] * 4
        y = np.empty(4, dtype=object)
        y[:] = [np.array([1.0, 2.0])] * 4

        with self.assertRaisesRegex(ValueError, "constructor syntax"):
            model.fit(X, y)

    def test_template_custom_combiner_infers_num_features(self):
        type_name = "TemplateVectorValue"
        model = tiny_model(
            vector_spec(name=type_name),
            expression_spec=TemplateExpressionSpec(
                combine="add_vectors(f(x1), g(x2))",
                expressions=["f", "g"],
                variable_names=["x1", "x2"],
            ),
            operators={
                2: [
                    f"""
                    add_vectors(a::{type_name}, b::{type_name}) =
                        {type_name}(a.data + b.data)
                    add_vectors(a::ValidVector, b::ValidVector) =
                        ValidVector(map(add_vectors, a.x, b.x), a.valid && b.valid)
                    """
                ]
            },
            elementwise_loss=(
                f"vector_loss(a::{type_name}, b::{type_name})::Float64 = "
                "sum(abs2, a.data - b.data)"
            ),
        )
        X = np.empty((4, 2), dtype=object)
        X[:, 0] = [np.array([1.0, 2.0])] * 4
        X[:, 1] = [np.array([3.0, 4.0])] * 4
        y = np.empty(4, dtype=object)
        y[:] = [np.array([4.0, 6.0])] * 4

        model.fit(X, y, variable_names=["x1", "x2"])

        structure = model.julia_options_.expression_options.structure
        self.assertEqual(int(structure.num_features.f), 1)
        self.assertEqual(int(structure.num_features.g), 1)

    def test_template_type_spec_parameters(self):
        type_name = "TemplateParameterValue"
        X, y = string_data(constant=True)
        expression_spec = TemplateExpressionSpec(
            combine="choose_parameter(p[1], f(x))",
            expressions=["f"],
            variable_names=["x"],
            parameters={"p": 1},
        )
        model = tiny_model(
            string_spec(
                name=type_name,
                sample=f'rng -> {type_name}("a")',
                mutate="(rng, value, temperature) -> value",
            ),
            expression_spec=expression_spec,
            operators={
                1: [f"identity_{type_name}(x::{type_name}) = x"],
                2: [
                    f"""
                    choose_parameter(a::{type_name}, b::{type_name}) = a
                    choose_parameter(a::{type_name}, b::ValidVector) =
                        ValidVector(map(_ -> a, b.x), b.valid)
                    """
                ],
            },
            parallelism="multiprocessing",
            procs=2,
        )

        model.fit(X, y)
        fingerprint = model._type_spec_runtime_definition_.fingerprint
        model.set_params(warm_start=True)

        model.fit(X, y)

        self.assertEqual(model._type_spec_runtime_definition_.fingerprint, fingerprint)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_template_type_spec_parameters_are_mutated(self):
        type_name = "MutatedParameterValue"
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.full(len(X), "b", dtype=object)
        model = tiny_model(
            string_spec(
                name=type_name,
                sample=f'rng -> {type_name}("a")',
                mutate=f'(rng, value, temperature) -> {type_name}("b")',
            ),
            expression_spec=TemplateExpressionSpec(
                combine="choose_parameter(p[1], f(x))",
                expressions=["f"],
                variable_names=["x"],
                parameters={"p": 1},
            ),
            operators={
                1: [f"identity_{type_name}(x::{type_name}) = x"],
                2: [
                    f"""
                    choose_parameter(a::{type_name}, b::{type_name}) = a
                    choose_parameter(a::{type_name}, b::ValidVector) =
                        ValidVector(map(_ -> a, b.x), b.valid)
                    """
                ],
            },
            niterations=10,
            ncycles_per_iteration=10,
            populations=2,
            population_size=20,
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertEqual(model.get_best()["loss"], 0.0)

    def test_numeric_path_remains_independent(self):
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            unary_operators=[],
            binary_operators=["+"],
            niterations=1,
            ncycles_per_iteration=2,
            populations=1,
            population_size=8,
            tournament_selection_n=3,
            maxsize=7,
            parallelism="serial",
            deterministic=True,
            random_state=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            should_optimize_constants=False,
        )
        model.fit(X, y)
        prediction = model.predict(X)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.isfinite(prediction).all())

        model.set_params(
            type_spec=string_spec(),
            operators={1: ["identity_value(x::StringValue) = x"]},
        )

        np.testing.assert_array_equal(model.predict(X), prediction)
        self.assertTrue(model._supports_export("sympy"))

    def test_configured_full_objective_and_module_local_helpers(self):
        X, y = string_data()
        type_name = "ConfiguredObjectiveValue"
        called = "CONFIGURED_OBJECTIVE_SAW_BATCH"
        model = tiny_model(
            string_spec(
                name=type_name,
                preamble=(
                    f"const {called} = Ref(false)\n" "const PrivateLoss = Float64"
                ),
                loss_type="PrivateLoss",
            ),
            operators={1: [f"configured_identity(x::{type_name}) = x"]},
            elementwise_loss=None,
            loss_function="""
                function configured_objective(
                    tree, dataset::Dataset, options, idx=nothing
                )::PrivateLoss
                    CONFIGURED_OBJECTIVE_SAW_BATCH[] |= idx !== nothing
                    _, complete = eval_tree_array(tree, dataset.X, options)
                    return complete ? 0.0 : Inf
                end
            """,
            batching=True,
            batch_size=2,
            complexity_mapping=f"expression -> {type_name} <: Any ? 1 : 1",
            early_stop_condition="(loss, complexity) -> false",
        )
        model.fit(X, y)
        runtime = model._load_type_spec_runtime()

        self.assertTrue(
            bool(
                jl.getindex(
                    jl.getproperty(runtime.configuration_module, jl.Symbol(called))
                )
            )
        )
        self.assertEqual(model.predict(X).shape, y.shape)

    def test_custom_expression_specs_are_deferred(self):
        class CustomExpressionSpec(AbstractExpressionSpec):
            def julia_expression_spec(self):
                return ExpressionSpec().julia_expression_spec()

            def create_exports(self, *args, **kwargs):
                return ExpressionSpec().create_exports(*args, **kwargs)

        X, y = string_data()
        model = tiny_model(
            string_spec(name="DeferredExpressionValue"),
            expression_spec=CustomExpressionSpec(),
        )
        with self.assertRaisesRegex(
            ValueError, "CustomExpressionSpec does not support TypeSpec"
        ):
            model.fit(X, y)

    def test_rejects_invalid_specs(self):
        cases = [
            (dict(name="not an id"), "not an identifier"),
            (dict(fields={}), "non-empty ordered mapping"),
            (dict(fields={"not an id": "String"}), "is not an identifier"),
            (dict(fields={"data": " "}), "requires a Julia type"),
            (dict(sample=" "), "must contain Julia source"),
            (dict(scalar_constants="value -> [1.0]"), "must be provided together"),
            (dict(mutate=None), "requires an explicit `mutate`"),
            (dict(string=" "), "cannot be empty"),
        ]
        for overrides, message in cases:
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, message):
                    string_spec(**overrides)

    def test_rejects_invalid_configurations(self):
        no_loss = dict(
            elementwise_loss=None, loss_function=None, loss_function_expression=None
        )
        operators = {1: ["x -> x"]}
        cases = [
            (string_spec(), operators, no_loss, "exactly one of"),
            (
                string_spec(loss_type="Float64"),
                operators,
                {**no_loss, "elementwise_loss": "(p, t) -> 0.0"},
                "return type is inferred",
            ),
            (
                string_spec(),
                operators,
                {**no_loss, "loss_function": "f(tree, dataset, options) = 0.0"},
                "explicit `loss_type`",
            ),
            (string_spec(), None, no_loss, "requires explicit"),
            (string_spec(), {0: ["x -> x"]}, no_loss, "positive integers"),
            (string_spec(), {1: []}, no_loss, "cannot be empty"),
            (string_spec(), {1: [" "]}, no_loss, "must contain Julia source"),
        ]
        for spec, operator_table, losses, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_type_spec_configuration(spec, operator_table, **losses)

    def test_fit_data_validation(self):
        X = np.array([["a"], ["b"]], dtype=object)
        y = X[:, 0].copy()

        def prepare(
            model=None,
            X=X,
            y=y,
            Xresampled=None,
            weights=None,
            variable_names=None,
            X_units=None,
            y_units=None,
        ):
            return prepare_type_spec_fit_data(
                model or PySRRegressor(),
                X,
                y,
                Xresampled,
                weights,
                variable_names,
                None,
                X_units,
                y_units,
            )

        with self.assertRaisesRegex(NotImplementedError, "denoising"):
            prepare(model=PySRRegressor(denoise=True))
        with self.assertRaisesRegex(ValueError, "2D array"):
            prepare(X=np.empty((1, 1, 1), dtype=object))
        with self.assertRaisesRegex(NotImplementedError, "one output"):
            prepare(y=np.empty((2, 2), dtype=object))
        with self.assertRaisesRegex(ValueError, "inconsistent numbers of samples"):
            prepare(y=y[:1])
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            prepare(X=np.empty((0, 1), dtype=object), y=np.empty(0, dtype=object))
        with self.assertRaisesRegex(NotImplementedError, "weights"):
            prepare(weights=np.ones(2))
        with self.assertRaisesRegex(NotImplementedError, "units"):
            prepare(X_units=["m"])
        with self.assertWarnsRegex(UserWarning, "reset to `None`"):
            prepare(X=pd.DataFrame({"a": ["a", "b"]}), variable_names=["a"])
        model = PySRRegressor()
        with self.assertWarnsRegex(UserWarning, "Spaces"):
            prepare(model=model, y=y.reshape(-1, 1), variable_names=["a b"])
        self.assertEqual(list(model.feature_names_in_), ["a_b"])

    def test_prediction_data_validation(self):
        model = PySRRegressor()
        model.n_features_in_ = 1
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(["a_b"])
        with self.assertRaisesRegex(ValueError, "2D array"):
            prepare_type_spec_prediction_data(model, np.empty((1, 1, 1), dtype=object))
        with self.assertRaisesRegex(ValueError, "different number of features"):
            prepare_type_spec_prediction_data(
                model, np.array([["a", "b"]], dtype=object)
            )
        with self.assertRaisesRegex(ValueError, "missing features"):
            prepare_type_spec_prediction_data(model, pd.DataFrame({"z": ["a"]}))
        with self.assertWarnsRegex(UserWarning, "Spaces"):
            out = prepare_type_spec_prediction_data(model, pd.DataFrame({"a b": ["a"]}))
        self.assertEqual(out.shape, (1, 1))

        model.n_features_in_ = 2
        model.selection_mask_ = np.array([True, False])
        out = prepare_type_spec_prediction_data(
            model, np.array([["a", "b"]], dtype=object)
        )
        np.testing.assert_array_equal(out, np.array([["a"]], dtype=object))

    def test_julia_array_conversion_errors(self):
        runtime = load_type_spec_runtime(
            compile_type_spec(string_spec(name="ConversionErrorValue"))
        )
        with self.assertRaisesRegex(ValueError, "1D or 2D"):
            type_spec_to_julia_array(runtime, np.empty((1, 1, 1), dtype=object))
        with self.assertRaises(ValueError):
            type_spec_to_julia_array(runtime, np.array([1.0, 2.0]))

    def test_runtime_load_rejects_inconsistent_definition_values(self):
        definition = compile_type_spec_runtime(
            string_spec(name="InconsistentLoadValue"),
            {1: ["x -> x"]},
            elementwise_loss="(prediction, target) -> 0.0",
            loss_function=None,
            loss_function_expression=None,
            complexity_mapping=None,
            early_stop_condition=None,
        )
        load_type_spec_runtime(definition)
        with patch("pysr.type_specs._runtime_sources", return_value=["x -> x"] * 99):
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                load_type_spec_runtime(definition)

    def test_fitted_model_guards(self):
        X, y = string_data()
        spec = string_spec(name="GuardedValue")
        model = tiny_model(spec)
        model.fit(X, y)

        with self.assertRaises(NotImplementedError):
            model.score(X, y)
        with self.assertRaises(ValueError):
            model.predict(np.array([[1.0]]))

        model.set_params(warm_start=True, type_spec=None)
        with self.assertRaisesRegex(ValueError, "Cannot warm-start"):
            model.fit(X, y)

        model.set_params(warm_start=False, type_spec=spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(NotImplementedError, "weights"):
                model.fit(X, y, weights=np.ones(len(y)))
        self.assertFalse(hasattr(model, "_type_spec_runtime_definition_"))


if __name__ == "__main__":
    unittest.main()
