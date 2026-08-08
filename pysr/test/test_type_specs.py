import unittest
import uuid

import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec, jl


class TestTypeSpecs(unittest.TestCase):
    def test_type_spec_instantiates_compact_global_interface(self):
        name = f"PySRTestValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Float64"},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}(1.0)",
            mutate_value=f"(rng, value, temperature) -> {name}(value.data + temperature)",
            count_scalar_constants=1,
            can_optimize=False,
        )

        value_type = spec.instantiate()
        options = jl.nothing
        jl.seval("using Random")
        rng = jl.Random.Xoshiro(0)

        self.assertEqual(jl.SymbolicRegression.init_value(value_type).data, 0.0)
        self.assertEqual(
            jl.SymbolicRegression.sample_value(rng, value_type, options).data, 1.0
        )
        self.assertEqual(
            jl.SymbolicRegression.mutate_value(
                rng, jl.SymbolicRegression.init_value(value_type), 0.5, options
            ).data,
            0.5,
        )
        self.assertEqual(
            jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants(
                jl.SymbolicRegression.init_value(value_type)
            ),
            1,
        )
        self.assertFalse(
            jl.SymbolicRegression.ConstantOptimizationModule.can_optimize(
                value_type, options
            )
        )

    def test_type_spec_rejects_wrong_callback_arity(self):
        name = f"InvalidTypeSpec_{uuid.uuid4().hex}"
        with self.assertRaisesRegex(ValueError, "sample_value must accept"):
            TypeSpec(
                name, fields={"data": "String"}, sample_value='() -> ""'
            ).instantiate()

    def test_type_spec_rejects_incompatible_or_invalid_definitions(self):
        name = f"UniqueTypeSpec_{uuid.uuid4().hex}"
        TypeSpec(name, fields={"data": "Float64"}).instantiate()
        with self.assertRaisesRegex(ValueError, "different TypeSpec"):
            TypeSpec(name, fields={"data": "String"}).instantiate()
        with self.assertRaisesRegex(ValueError, "simple type name"):
            TypeSpec("Base.Invalid", fields={"data": "Float64"}).instantiate()
        with self.assertRaisesRegex(ValueError, "concrete Julia type"):
            TypeSpec("nothing").instantiate()

    def test_type_spec_converts_values_and_callback_constants(self):
        float_spec = TypeSpec("Float64")
        values = float_spec.to_julia_array([1.0, 2.0])
        np.testing.assert_array_equal(np.asarray(list(values)), [1.0, 2.0])
        transposed = float_spec.to_julia_array([[1.0, 2.0]], transpose=True)
        self.assertEqual(tuple(transposed.shape), (2, 1))
        with self.assertRaisesRegex(ValueError, "1D or 2D"):
            TypeSpec("Float64").to_julia_array(np.zeros((1, 1, 1)))

        name = f"CountingTypeSpec_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Float64"},
            count_scalar_constants="value -> 2",
        )
        spec.instantiate()
        self.assertEqual(spec.instantiate(), jl.seval(name))
        self.assertEqual(spec.to_julia_array([1.0])[0].data, 1.0)
        value = jl.seval(f"{name}(1.0)")
        self.assertEqual(
            jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants(
                value
            ),
            2,
        )
        with self.assertRaises(NotImplementedError):
            TypeSpec(
                f"TwoFieldTypeSpec_{uuid.uuid4().hex}",
                fields={"x": "Float64", "y": "Float64"},
            ).to_julia_array([[1.0, 2.0]])

    @staticmethod
    def _tiny_model(type_spec, operator, loss, **kwargs):
        params = dict(
            type_spec=type_spec,
            operators={1: [operator]},
            elementwise_loss=loss,
            loss_type="Float64",
            niterations=1,
            ncycles_per_iteration=5,
            populations=1,
            population_size=10,
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
        params.update(kwargs)
        return PySRRegressor(
            **params,
        )

    def test_string_type_spec_fit_and_predict(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
        )
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = self._tiny_model(
            spec,
            "identity_string(x::String) = x",
            "string_loss(x::String, y::String) = x == y ? 0.0 : 1.0",
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)

    def test_type_spec_supports_full_loss_function(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
        )
        X = np.array([["a"], ["b"]], dtype=object)
        y = np.array(["a", "b"], dtype=object)
        for loss_kwarg, loss in (
            ("loss_function", "full_string_loss(tree, dataset, options) = 0.0"),
            (
                "loss_function_expression",
                "full_string_expression_loss(expression, dataset, options) = 0.0",
            ),
        ):
            with self.subTest(loss_kwarg=loss_kwarg):
                model = PySRRegressor(
                    type_spec=spec,
                    operators={1: ["identity_string_full_loss(x::String) = x"]},
                    loss_type="Float64",
                    niterations=1,
                    ncycles_per_iteration=1,
                    populations=1,
                    population_size=5,
                    tournament_selection_n=3,
                    maxsize=7,
                    parallelism="serial",
                    deterministic=True,
                    random_state=0,
                    progress=False,
                    verbosity=0,
                    temp_equation_file=True,
                    should_optimize_constants=False,
                    **{loss_kwarg: loss},
                )

                model.fit(X, y)

    def test_struct_type_spec_fit_and_predict(self):
        name = f"RASPValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Union{Float64, Vector{Float64}}"},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}(randn(rng))",
            mutate_value=(
                f"(rng, value, temperature) -> {name}(value.data isa Vector "
                "? value.data : value.data + temperature * randn(rng))"
            ),
            count_scalar_constants=1,
            can_optimize=False,
        )
        sequences = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        X = pd.DataFrame({"x": sequences})
        y = pd.Series(sequences, dtype=object)
        model = self._tiny_model(
            spec,
            f"identity_rasp(x::{name}) = x",
            f"rasp_loss(x::{name}, y::{name}) = x.data == y.data ? 0.0 : 1.0",
        )

        model.fit(X, y)

        prediction = model.predict(X, index=model.equations_["loss"].idxmin())
        self.assertEqual([list(value.data) for value in prediction], y.tolist())

    def test_type_spec_supports_multithreading(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
        )
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = self._tiny_model(
            spec,
            "identity_string_threaded(x::String) = x",
            "string_loss_threaded(x::String, y::String) = x == y ? 0.0 : 1.0",
            parallelism="multithreading",
            deterministic=False,
            random_state=None,
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
