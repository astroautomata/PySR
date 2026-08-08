from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .julia_import import AnyValue, jl


@dataclass(frozen=True)
class TypeSpec:
    """Runtime definition of a value type used by SymbolicRegression.jl."""

    julia_type: str
    fields: dict[str, str] | None = None
    init_value: str | None = None
    sample_value: str | None = None
    mutate_value: str | None = None
    count_scalar_constants: int | str | None = None
    can_optimize: bool | None = None

    _instantiated: ClassVar[dict[str, tuple[tuple[Any, ...], AnyValue]]] = {}

    def instantiate(self) -> AnyValue:
        """Define the type and its global SymbolicRegression.jl interface methods."""
        key = (
            self.julia_type,
            tuple(self.fields.items()) if self.fields is not None else None,
            self.init_value,
            self.sample_value,
            self.mutate_value,
            self.count_scalar_constants,
            self.can_optimize,
        )
        instantiated = self._instantiated.get(self.julia_type)
        if instantiated is not None:
            if instantiated[0] != key:
                raise ValueError(
                    f"A different TypeSpec for `{self.julia_type}` is already instantiated."
                )
            return instantiated[1]

        jl.seval("using Random: AbstractRNG")
        if self.fields is not None:
            if not self.julia_type.isidentifier():
                raise ValueError("A TypeSpec with fields requires a simple type name.")
            fields = "\n".join(
                f"    {name}::{type_}" for name, type_ in self.fields.items()
            )
            jl.seval(f"struct {self.julia_type}\n{fields}\nend")

        value_type = jl.seval(self.julia_type)
        if not jl.seval("T -> T isa Type")(value_type):
            raise ValueError(f"`{self.julia_type}` is not a concrete Julia type.")

        if self.init_value is not None:
            self._instantiate("init", self.init_value)
        if self.sample_value is not None:
            self._instantiate("sample", self.sample_value)
        if self.mutate_value is not None:
            self._instantiate("mutate", self.mutate_value)
        if self.count_scalar_constants is not None:
            if isinstance(self.count_scalar_constants, int):
                jl.seval(
                    "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"count_scalar_constants(value::{self.julia_type}) = {self.count_scalar_constants}"
                )
            else:
                self._instantiate("count", self.count_scalar_constants)
        if self.can_optimize is not None:
            jl.seval(
                "SymbolicRegression.ConstantOptimizationModule."
                f"can_optimize(::Type{{{self.julia_type}}}, _) = {str(self.can_optimize).lower()}"
            )
        self._instantiated[self.julia_type] = (key, value_type)
        return value_type

    def to_julia_array(self, values: Any, *, transpose: bool = False) -> AnyValue:
        """Convert Python logical values into a concrete Julia array."""
        array = np.asarray(values, dtype=object)
        if transpose:
            array = array.T
        if array.ndim not in (1, 2):
            raise ValueError("TypeSpec data must be a 1D or 2D array.")

        value_type = self.instantiate()
        if self.fields is None:
            convert = jl.seval("(T, x) -> PythonCall.pyconvert(T, x)")
        elif len(self.fields) == 1:
            convert = jl.seval("(T, x) -> T(PythonCall.pyconvert(fieldtype(T, 1), x))")
        else:
            raise NotImplementedError(
                "Automatic conversion currently supports one-field structs."
            )

        converted = [convert(value_type, value) for value in array.ravel(order="F")]
        return jl.seval("(T, xs, dims) -> reshape(T[x for x in xs], Tuple(dims))")(
            value_type, converted, array.shape
        )

    def _instantiate(self, kind: str, source: str) -> None:
        function = jl.seval(source)
        arity = jl.seval("f -> only(methods(f)).nargs - 1")(function)
        expected = {"init": (0,), "sample": (1, 2), "mutate": (3, 4), "count": (1,)}[
            kind
        ]
        if arity not in expected:
            raise ValueError(
                f"{kind}_value must accept {expected}; got {arity} arguments."
            )

        arguments = {
            "sample": {1: "rng", 2: "rng, options"},
            "mutate": {
                3: "rng, value, temperature",
                4: "rng, value, temperature, options",
            },
        }
        if kind == "init":
            definition = f"SymbolicRegression.init_value(::Type{{{self.julia_type}}}) = ({source})()"
        elif kind == "sample":
            definition = (
                "SymbolicRegression.sample_value("
                f"rng::AbstractRNG, ::Type{{{self.julia_type}}}, options) = "
                f"({source})({arguments['sample'][arity]})"
            )
        elif kind == "mutate":
            definition = (
                "SymbolicRegression.mutate_value("
                f"rng::AbstractRNG, value::{self.julia_type}, temperature, options) = "
                f"({source})({arguments['mutate'][arity]})"
            )
        else:
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"count_scalar_constants(value::{self.julia_type}) = ({source})(value)"
            )
        jl.seval(definition)
