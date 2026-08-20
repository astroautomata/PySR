from __future__ import annotations

import copy
import hashlib
import json
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
from juliacall import JuliaError  # type: ignore

from .julia_helpers import jl_array, jl_named_tuple
from .julia_import import AnyValue, SymbolicRegression, jl

_TYPE_MODULE_INSTALLED = "_PYSR_TYPE_SPEC_INSTALLED"


def object_array_1d(values: Any) -> np.ndarray:
    """Build a 1D object array whose cells are the logical values themselves."""
    if isinstance(values, np.ndarray):
        return values if values.dtype == object else values.astype(object)
    values = list(values)
    array = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        array[i] = value
    return array


def object_array_2d(values: Any) -> np.ndarray:
    """Build a 2D object array whose cells are the logical values themselves."""
    if isinstance(values, np.ndarray):
        return values if values.dtype == object else values.astype(object)
    try:
        values = list(values)
        if any(isinstance(row, (str, bytes)) for row in values):
            raise TypeError
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("TypeSpec X must be a 2D array of logical values.")
    n_columns = len(rows[0]) if rows else 0
    array = np.empty((len(rows), n_columns), dtype=object)
    for i, row in enumerate(rows):
        if len(row) != n_columns:
            raise ValueError("All rows of X must have the same number of features.")
        for j, value in enumerate(row):
            array[i, j] = value
    return array


@dataclass
class TypeSpec:
    """Definition of a custom value type for symbolic regression.

    Parameters
    ----------
    name : str
        Name of the generated Julia type. Use this name in operator and hook
        definitions.
    fields : dict[str, str]
        Ordered mapping from field names to Julia field types.
    sample : str
        Julia callable with signature ``rng -> (...)::{name}``.
    scalar_constants : str, optional
        Julia callable with signature ``value::{name} -> (...)::Vector{Float64}``. Provide
        together with ``with_scalar_constants`` to enable continuous constant
        optimization.
    with_scalar_constants : str, optional
        Julia callable with signature ``(value::{name}, scalar_constants::Vector{Float64}) -> (...)::{name}``.
    init : str, optional
        Julia callable with signature ``() -> value::{name}``. The default repurposes `sample`
        using a deterministic local random-number generator.
    mutate : str, optional
        Julia callable with signature ``(rng, value::{name}, temperature::Float64) -> value::{name}``.
        Use this to mutate a value of the type.
        The `temperature` (from simulated annealing) is between 0 and 1. High temperatures
        generally should be used for more aggressive mutations, and low temperatures for
        more conservative mutations.
        The default mutation repurposes `scalar_constants` and `with_scalar_constants`
        to mutate the set with SymbolicRegression.jl's default constant mutation,
        and put it back into a new value.
    is_valid : str, optional
        Julia callable with signature ``value::{name} -> (...)::Bool``.
        Use this to check if your type is in a valid state. For example, if your type
        has been processed in a way that invalidates it (like a NaN, but specific to your type),
        you may use this function to check for that. This will be used to quit evaluation early.
        The default checks that every scalar constant is finite, or accepts every value for a
        non-optimizable type.
    string : str, optional
        Julia callable with signature ``value::{name} -> (...)::AbstractString`` used to print
        values.
    preamble : str, optional
        Julia source evaluated once before the generated type definition. Types
        and functions it defines are visible to the hooks and to operator and
        objective sources; imports are not, so any source needing a package
        beyond SymbolicRegression must import it itself.
    loss_type : str, optional
        Concrete Julia ``AbstractFloat`` type returned by a custom full
        objective. Elementwise loss return types are inferred.
    """

    name: str
    fields: dict[str, str]
    sample: str
    scalar_constants: str | None = None
    with_scalar_constants: str | None = None
    init: str | None = None
    mutate: str | None = None
    is_valid: str | None = None
    string: str | None = None
    preamble: str | None = None
    loss_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"TypeSpec name {self.name!r} is not an identifier.")
        if not isinstance(self.fields, dict) or not self.fields:
            raise ValueError("`fields` must be a non-empty ordered mapping.")
        for name, field_type in self.fields.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"TypeSpec field name {name!r} is not an identifier.")
            if not isinstance(field_type, str) or not field_type.strip():
                raise ValueError(f"TypeSpec field `{name}` requires a Julia type.")
        if not isinstance(self.sample, str) or not self.sample.strip():
            raise ValueError("`sample` must contain Julia source.")

        if (self.scalar_constants is None) != (self.with_scalar_constants is None):
            raise ValueError(
                "`scalar_constants` and `with_scalar_constants` must be provided "
                "together."
            )
        if not self.can_optimize and self.mutate is None:
            raise ValueError(
                "A non-optimizable TypeSpec requires an explicit `mutate` callable."
            )

        for name in (
            "scalar_constants",
            "with_scalar_constants",
            "init",
            "mutate",
            "is_valid",
            "string",
            "preamble",
            "loss_type",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` cannot be empty.")

    @property
    def can_optimize(self) -> bool:
        return self.scalar_constants is not None


@dataclass(frozen=True)
class _TypeSpecDefinition:
    spec: TypeSpec
    fingerprint: str
    source: str

    @property
    def module_name(self) -> str:
        return f"_PySRTypeSpec_{self.fingerprint[:20]}"


@dataclass(frozen=True)
class _TypeSpecRuntimeDefinition:
    type_spec: _TypeSpecDefinition
    operators: tuple[tuple[int, tuple[str, ...]], ...]
    elementwise_loss: str | None
    loss_function: str | None
    loss_function_expression: str | None
    complexity_mapping: str | None
    early_stop_condition: str | None
    expression_spec: str
    expression_spec_function_selector: str | None
    fingerprint: str
    source: str

    @property
    def module_name(self) -> str:
        return f"_PySRConfig_{self.fingerprint[:20]}"


@dataclass(frozen=True)
class _TypeSpecRuntime:
    definition: _TypeSpecDefinition | _TypeSpecRuntimeDefinition
    module: AnyValue
    value_type: AnyValue
    configuration_module: AnyValue
    operator_functions: dict[int, tuple[AnyValue, ...]]
    operator_names: dict[int, tuple[str, ...]]
    elementwise_loss: AnyValue | None = None
    loss_function: AnyValue | None = None
    loss_function_expression: AnyValue | None = None
    complexity_mapping: AnyValue | None = None
    early_stop_condition: float | AnyValue | None = None
    expression_spec: AnyValue | None = None

    @property
    def type_definition(self) -> _TypeSpecDefinition:
        if isinstance(self.definition, _TypeSpecRuntimeDefinition):
            return self.definition.type_spec
        return self.definition

    @property
    def spec(self) -> TypeSpec:
        return self.type_definition.spec


def _quoted(source: str) -> str:
    return json.dumps(source, ensure_ascii=False).replace("$", r"\$")


def _block(source: str) -> str:
    return dedent(source).strip()


def _runtime_sources(
    operators: tuple[tuple[int, tuple[str, ...]], ...],
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
    complexity_mapping: str | None,
    early_stop_condition: str | None,
    expression_spec: str,
) -> tuple[str, ...]:
    sources = tuple(
        source for _, operator_sources in operators for source in operator_sources
    )
    optional_sources = (
        elementwise_loss,
        loss_function,
        loss_function_expression,
        complexity_mapping,
        early_stop_condition,
    )
    return (
        *sources,
        *(source for source in optional_sources if source is not None),
        expression_spec,
    )


def _runtime_install_source(
    type_definition: _TypeSpecDefinition,
    runtime_module_name: str,
    sources: tuple[str, ...],
    expression_spec_function_selector: str | None,
) -> str:
    source_literals = ",\n".join(f"            {_quoted(source)}" for source in sources)
    selector = _optional_source(expression_spec_function_selector)
    return _block(f"""
        let type_module_name = Symbol({_quoted(type_definition.module_name)})
            installed =
                isdefined(Main, type_module_name) &&
                isdefined(
                    Base.invokelatest(getproperty, Main, type_module_name),
                    Symbol({_quoted(_TYPE_MODULE_INSTALLED)}),
                )
            if !installed
                Base.include_string(
                    Main,
                    {_quoted(type_definition.source)},
                    {_quoted("PySR." + type_definition.module_name)},
                )
            end
        end
        let
            parent = getproperty(
                Main, Symbol({_quoted(type_definition.module_name)})
            )
            module_name = Symbol({_quoted(runtime_module_name)})
            needs_install = !isdefined(parent, module_name)
            if !needs_install
                existing = Base.invokelatest(getproperty, parent, module_name)
                needs_install = !isdefined(existing, :_definition_values)
            end
            if needs_install
                Core.eval(parent, Expr(:module, true, module_name, Expr(:block)))
                module_ = Base.invokelatest(getproperty, parent, module_name)
                Core.eval(module_, :(using SymbolicRegression))
                for name in getproperty(parent, :_PYSR_PARENT_BINDING_NAMES)
                    isdefined(module_, name) && continue
                    Core.eval(
                        module_,
                        Expr(
                            :const,
                            Expr(
                                :(=),
                                name,
                                QuoteNode(getproperty(parent, name)),
                            ),
                        ),
                    )
                end
                sources = String[
        {source_literals}
                ]
                values = Any[]
                for (index, source) in enumerate(sources)
                    push!(
                        values,
                        Base.include_string(
                            module_,
                            source,
                            "PySR TypeSpec configuration $index",
                        ),
                    )
                end
                selector_source = {selector}
                if selector_source !== nothing
                    selector_value = Base.include_string(
                        module_,
                        selector_source,
                        "PySR TypeSpec expression function selector",
                    )
                    push!(values, Base.invokelatest(selector_value, values[end]))
                end
                Core.eval(
                    module_,
                    Expr(
                        :const,
                        Expr(:(=), :_definition_values, QuoteNode(Tuple(values))),
                    ),
                )
            end
        end
        """) + "\n"


def compile_type_spec_runtime(
    spec: TypeSpec,
    operators: dict[int, list[str]],
    *,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
    complexity_mapping: str | None,
    early_stop_condition: str | None,
    expression_spec: str | None = None,
    expression_spec_function_selector: str | None = None,
) -> _TypeSpecRuntimeDefinition:
    validate_type_spec_configuration(
        spec,
        operators,
        elementwise_loss=elementwise_loss,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
    )
    type_definition = compile_type_spec(spec)
    normalized_operators = tuple(
        (arity, tuple(sources)) for arity, sources in _normalize_operators(operators)
    )
    if expression_spec is None:
        expression_spec = _block(f"""
            SymbolicRegression.ExpressionSpec(
                node_type=SymbolicRegression.InterfaceDynamicExpressionsModule.DE.Node{{
                    {spec.name}
                }},
            )
            """)
    payload = {
        "type_spec": type_definition.fingerprint,
        "operators": normalized_operators,
        "elementwise_loss": elementwise_loss,
        "loss_function": loss_function,
        "loss_function_expression": loss_function_expression,
        "complexity_mapping": complexity_mapping,
        "early_stop_condition": early_stop_condition,
        "expression_spec": expression_spec,
        "expression_spec_function_selector": expression_spec_function_selector,
        "loss_type": spec.loss_type,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    runtime_module_name = f"_PySRConfig_{fingerprint[:20]}"
    sources = _runtime_sources(
        normalized_operators,
        elementwise_loss,
        loss_function,
        loss_function_expression,
        complexity_mapping,
        early_stop_condition,
        expression_spec,
    )
    source = _runtime_install_source(
        type_definition,
        runtime_module_name,
        sources,
        expression_spec_function_selector,
    )
    return _TypeSpecRuntimeDefinition(
        type_spec=type_definition,
        operators=normalized_operators,
        elementwise_loss=elementwise_loss,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
        complexity_mapping=complexity_mapping,
        early_stop_condition=early_stop_condition,
        expression_spec=expression_spec,
        expression_spec_function_selector=expression_spec_function_selector,
        fingerprint=fingerprint,
        source=source,
    )


def compile_type_spec_runtime_for_model(
    model: Any,
    operators: dict[int, list[str]],
) -> _TypeSpecRuntimeDefinition:
    """Compile one model configuration without evaluating its Julia sources."""
    from .expression_specs import ExpressionSpec

    spec = model.type_spec
    assert spec is not None
    expression_spec = model.expression_spec_
    expression_spec_source = expression_spec._julia_expression_spec_source(
        prototype=f"SymbolicRegression.init_value({spec.name})"
    )
    if expression_spec_source is None and type(expression_spec) is not ExpressionSpec:
        raise ValueError(
            f"{type(expression_spec).__name__} declares TypeSpec support "
            "but does not provide Julia source."
        )
    return compile_type_spec_runtime(
        spec,
        operators,
        elementwise_loss=model.elementwise_loss,
        loss_function=model.loss_function,
        loss_function_expression=model.loss_function_expression,
        complexity_mapping=model.complexity_mapping,
        early_stop_condition=(
            None
            if model.early_stop_condition is None
            else str(model.early_stop_condition)
        ),
        expression_spec=expression_spec_source,
        expression_spec_function_selector=(
            expression_spec._julia_expression_spec_function_selector()
        ),
    )


_TYPE_SPEC_MODULE = _block(r"""
    import SymbolicRegression: init_value, mutate_value, sample_value
    import SymbolicRegression.ConstantOptimizationModule: can_optimize
    import SymbolicRegression.InterfaceDynamicExpressionsModule: string_constant
    import SymbolicRegression.InterfaceDynamicExpressionsModule.DE:
        count_scalar_constants, get_number_type, is_valid,
        pack_scalar_constants!, unpack_scalar_constants
    import SymbolicRegression.InterfaceDynamicExpressionsModule.DE.StringsModule:
        needs_brackets

    const _config = __TYPE_SPEC_CONFIG__
    abstract type _TypeSpecValue end

    macro _define_type_spec(config_expression)
        config = Core.eval(__module__, config_expression)
        fields = map(config.fields) do (name, type)
            :($(Symbol(name))::$(Meta.parse(type)))
        end
        type_definition = Expr(
            :struct,
            false,
            Expr(:<:, Symbol(config.name), :_TypeSpecValue),
            Expr(:block, fields...),
        )
        return esc(type_definition)
    end

    _include(source, label) = Base.include_string(@__MODULE__, source, label)
    _config.preamble === nothing ||
        _include(_config.preamble, "TypeSpec.preamble")
    @_define_type_spec _config
    const _value_type = getfield(@__MODULE__, Symbol(_config.name))

    _fields(value) = ntuple(i -> getfield(value, i), fieldcount(_value_type))
    Base.:(==)(a::_TypeSpecValue, b::_TypeSpecValue) = _fields(a) == _fields(b)
    Base.isequal(a::_TypeSpecValue, b::_TypeSpecValue) =
        isequal(_fields(a), _fields(b))
    Base.hash(value::_TypeSpecValue, h::UInt) = hash(_fields(value), h)

    const _sample = _include(_config.sample, "TypeSpec.sample")
    const _init = _config.init === nothing ?
        () -> _sample(Random.Xoshiro(0)) :
        _include(_config.init, "TypeSpec.init")
    init_value(::Type{<:_TypeSpecValue}) = _init()
    sample_value(rng::AbstractRNG, ::Type{<:_TypeSpecValue}, options) = _sample(rng)
    can_optimize(::Type{<:_TypeSpecValue}, _) = _config.optimizable

    if _config.optimizable
        const _scalar_constants =
            _include(_config.scalar_constants, "TypeSpec.scalar_constants")
        const _with_scalar_constants = _include(
            _config.with_scalar_constants,
            "TypeSpec.with_scalar_constants",
        )
        get_number_type(::Type{<:_TypeSpecValue}) =
            eltype(_scalar_constants(init_value(_value_type)))
    end

    count_scalar_constants(value::_TypeSpecValue) =
        _config.optimizable ? length(_scalar_constants(value)) : 0

    if _config.optimizable
        function pack_scalar_constants!(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        )
            scalar_constants = _scalar_constants(value)
            copyto!(
                buffer,
                idx,
                scalar_constants,
                firstindex(scalar_constants),
                length(scalar_constants),
            )
            return idx + length(scalar_constants)
        end
        function unpack_scalar_constants(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        )
            count = length(_scalar_constants(value))
            scalar_constants = @view buffer[idx:(idx + count - 1)]
            return idx + count, _with_scalar_constants(value, scalar_constants)
        end
    end

    const _mutate = if _config.mutate === nothing
        function (rng, value, temperature, mutation)
            scalar_constants = collect(_scalar_constants(value))
            isempty(scalar_constants) && return _sample(rng)
            i = rand(rng, eachindex(scalar_constants))
            scalar_constants[i] =
                SymbolicRegression.MutationFunctionsModule.mutate_value(
                    rng, scalar_constants[i], temperature, mutation
                )
            return _with_scalar_constants(value, scalar_constants)
        end
    else
        mutate = _include(_config.mutate, "TypeSpec.mutate")
        (rng, value, temperature, _) -> mutate(rng, value, temperature)
    end
    mutate_value(
        rng::AbstractRNG,
        value::_TypeSpecValue,
        temperature,
        mutation::SymbolicRegression.ConstantMutation,
    ) = _mutate(rng, value, temperature, mutation)

    const _is_valid = if _config.is_valid !== nothing
        _include(_config.is_valid, "TypeSpec.is_valid")
    elseif _config.optimizable
        value -> all(isfinite, _scalar_constants(value))
    else
        _ -> true
    end
    is_valid(value::_TypeSpecValue) = _is_valid(value)

    const _string = if _config.string === nothing
        function (value)
            fields = map(1:fieldcount(_value_type)) do i
                sprint(show, getfield(value, i); context=:compact => true)
            end
            return fieldcount(_value_type) == 1 ? only(fields) :
                string(_config.name, "(", join(fields, ", "), ")")
        end
    else
        _include(_config.string, "TypeSpec.string")
    end
    Base.show(io::IO, value::_TypeSpecValue) = print(io, _string(value))
    needs_brackets(::_TypeSpecValue) = false
    string_constant(
        value::_TypeSpecValue, ::Val{precision}, unit
    ) where {precision} = _string(value) * unit

    function _convert_value(x)
        x isa _value_type && return x
        x = PythonCall.Py(x)
        fieldcount(_value_type) == 1 ||
            PythonCall.pyhasattr(x, "__len__") &&
                PythonCall.pylen(x) == fieldcount(_value_type) ||
            throw(ArgumentError(
                "TypeSpec values require exactly $(fieldcount(_value_type)) fields."
            ))
        values = ntuple(fieldcount(_value_type)) do i
            source = fieldcount(_value_type) == 1 ? x : x[i - 1]
            PythonCall.pyconvert(fieldtype(_value_type, i), source)
        end
        return _value_type(values...)
    end
    _convert_array(values, dims) = reshape(
        _value_type[_convert_value(value) for value in values], Tuple(dims)
    )
    """)


def _optional_source(source: str | None) -> str:
    return "nothing" if source is None else _quoted(source)


def validate_type_spec_loss_configuration(
    spec: TypeSpec,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> str:
    configured = [
        ("elementwise_loss", elementwise_loss),
        ("loss_function", loss_function),
        ("loss_function_expression", loss_function_expression),
    ]
    selected = [mode for mode, source in configured if source is not None]
    if len(selected) != 1:
        raise ValueError(
            "TypeSpec requires exactly one of `elementwise_loss`, `loss_function`, "
            "and `loss_function_expression`."
        )
    mode = selected[0]
    if mode == "elementwise_loss" and spec.loss_type is not None:
        raise ValueError(
            "Do not set `loss_type` with `elementwise_loss`; its return type is inferred."
        )
    if mode != "elementwise_loss" and spec.loss_type is None:
        raise ValueError("TypeSpec full objectives require an explicit `loss_type`.")
    return mode


def _normalize_operators(
    operators: dict[int, list[str]] | None,
) -> list[tuple[int, list[str]]]:
    if not operators:
        raise ValueError("TypeSpec requires explicit `operators={...}`.")
    normalized = []
    for arity, sources in operators.items():
        if type(arity) is not int or arity < 1:
            raise ValueError("TypeSpec operator arities must be positive integers.")
        if not sources:
            raise ValueError(f"TypeSpec operator arity {arity} cannot be empty.")
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("Every TypeSpec operator must contain Julia source.")
        normalized.append((arity, list(sources)))
    return sorted(normalized)


def validate_type_spec_configuration(
    spec: TypeSpec,
    operators: dict[int, list[str]] | None,
    *,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> None:
    _normalize_operators(operators)
    validate_type_spec_loss_configuration(
        spec, elementwise_loss, loss_function, loss_function_expression
    )


def compile_type_spec(spec: TypeSpec) -> _TypeSpecDefinition:
    """Create deterministic Julia source without evaluating user code."""
    field_sources = ",\n".join(
        f"    {_quoted(name)} => {_quoted(field_type)}"
        for name, field_type in spec.fields.items()
    )
    config = _block(f"""
        (
            name = {_quoted(spec.name)},
            fields = (
            {field_sources},
            ),
            sample = {_quoted(spec.sample)},
            scalar_constants = {_optional_source(spec.scalar_constants)},
            with_scalar_constants = {_optional_source(spec.with_scalar_constants)},
            init = {_optional_source(spec.init)},
            mutate = {_optional_source(spec.mutate)},
            is_valid = {_optional_source(spec.is_valid)},
            string = {_optional_source(spec.string)},
            preamble = {_optional_source(spec.preamble)},
            optimizable = {str(spec.can_optimize).lower()},
        )
        """)
    body = _TYPE_SPEC_MODULE.replace("__TYPE_SPEC_CONFIG__", config, 1)
    fingerprint = hashlib.sha256(body.encode()).hexdigest()
    module_name = f"_PySRTypeSpec_{fingerprint[:20]}"
    source = _block(f"""
        module {module_name}
        using Random
        using SymbolicRegression
        using PythonCall

        {body}

        const _PYSR_PARENT_BINDING_NAMES = Tuple(sort!(
            filter(
                name -> isdefined(@__MODULE__, name) &&
                    !startswith(String(name), '#'),
                collect(names(@__MODULE__; all=true, imported=true)),
            );
            by=String,
        ))
        const {_TYPE_MODULE_INSTALLED} = true
        end
        import .{module_name}: {spec.name}
        """) + "\n"
    return _TypeSpecDefinition(
        spec=copy.deepcopy(spec),
        fingerprint=fingerprint,
        source=source,
    )


def load_type_spec_runtime(
    definition: _TypeSpecDefinition | _TypeSpecRuntimeDefinition,
) -> _TypeSpecRuntime:
    """Load deterministic TypeSpec modules and return their Julia objects."""
    type_definition = (
        definition.type_spec
        if isinstance(definition, _TypeSpecRuntimeDefinition)
        else definition
    )
    if isinstance(definition, _TypeSpecRuntimeDefinition):
        install_source = definition.source
        filename = "PySR." + definition.module_name
    else:
        install_source = type_definition.source
        filename = "PySR." + type_definition.module_name
    module_symbol = jl.Symbol(type_definition.module_name)
    runtime_module_symbol = (
        jl.Symbol(definition.module_name)
        if isinstance(definition, _TypeSpecRuntimeDefinition)
        else None
    )
    needs_install = not bool(jl.isdefined(jl.Main, module_symbol))
    if not needs_install:
        module = jl.getproperty(jl.Main, module_symbol)
        needs_install = not bool(
            jl.isdefined(module, jl.Symbol(_TYPE_MODULE_INSTALLED))
        )
        if not needs_install and runtime_module_symbol is not None:
            needs_install = not bool(jl.isdefined(module, runtime_module_symbol))
            if not needs_install:
                configuration_module = jl.getproperty(module, runtime_module_symbol)
                needs_install = not bool(
                    jl.isdefined(configuration_module, jl.Symbol("_definition_values"))
                )
    if needs_install:
        jl.Base.include_string(jl.Main, install_source, filename)
    module = jl.getproperty(jl.Main, module_symbol)
    value_type = jl.getproperty(module, jl.Symbol(type_definition.spec.name))

    if isinstance(definition, _TypeSpecDefinition):
        runtime = _TypeSpecRuntime(
            definition=definition,
            module=module,
            value_type=value_type,
            configuration_module=module,
            operator_names={},
            operator_functions={},
        )
    else:
        assert runtime_module_symbol is not None
        configuration_module = jl.getproperty(module, runtime_module_symbol)
        raw_values = list(
            jl.getproperty(configuration_module, jl.Symbol("_definition_values"))
        )
        source_count = len(
            _runtime_sources(
                definition.operators,
                definition.elementwise_loss,
                definition.loss_function,
                definition.loss_function_expression,
                definition.complexity_mapping,
                definition.early_stop_condition,
                definition.expression_spec,
            )
        )
        expected_count = source_count + (
            definition.expression_spec_function_selector is not None
        )
        if len(raw_values) != expected_count:
            raise RuntimeError(
                "TypeSpec runtime source/value ordering is inconsistent."
            )
        values = iter(raw_values[:source_count])
        operator_functions = {
            arity: tuple(next(values) for _ in operator_sources)
            for arity, operator_sources in definition.operators
        }
        optional_sources = (
            definition.elementwise_loss,
            definition.loss_function,
            definition.loss_function_expression,
            definition.complexity_mapping,
            definition.early_stop_condition,
        )
        optional_values = [
            next(values) if source is not None else None for source in optional_sources
        ]
        runtime = _TypeSpecRuntime(
            definition=definition,
            module=module,
            value_type=value_type,
            configuration_module=configuration_module,
            operator_functions=operator_functions,
            operator_names={
                arity: tuple(str(jl.Base.nameof(function)) for function in functions)
                for arity, functions in operator_functions.items()
            },
            elementwise_loss=optional_values[0],
            loss_function=optional_values[1],
            loss_function_expression=optional_values[2],
            complexity_mapping=optional_values[3],
            early_stop_condition=optional_values[4],
            expression_spec=next(values),
        )
    return runtime


@cache
def _type_spec_worker_addprocs_factory() -> AnyValue:
    return jl.seval(r"""
        begin
            import Distributed
            function (
                addprocs_function,
                source,
                filename,
                worker_imports,
                type_module_name,
                runtime_module_name,
            )
                addprocs = something(addprocs_function, Distributed.addprocs)
                expression = Meta.parseall(source)
                type_module_symbol = Symbol(type_module_name)
                runtime_module_symbol = Symbol(runtime_module_name)
                return function (numprocs; kwargs...)
                    procs = addprocs(numprocs; kwargs...)
                    try
                        SymbolicRegression.import_module_on_workers(
                            procs,
                            pathof(SymbolicRegression),
                            worker_imports,
                            0,
                        )
                        Distributed.remotecall_eval(Main, procs, expression)
                        head_module = getproperty(
                            getproperty(Main, type_module_symbol),
                            runtime_module_symbol,
                        )
                        module_expression = :(
                            getproperty(
                                getproperty(
                                    Main,
                                    $(QuoteNode(type_module_symbol)),
                                ),
                                $(QuoteNode(runtime_module_symbol)),
                            )
                        )
                        for proc in procs
                            worker_module = Distributed.remotecall_fetch(
                                Core.eval,
                                proc,
                                Main,
                                module_expression,
                            )
                            worker_module === head_module || error(
                                "TypeSpec runtime identity differs on worker $proc."
                            )
                        end
                    catch
                        Distributed.rmprocs(procs)
                        rethrow()
                    end
                    return procs
                end
            end
        end
        """)


def create_type_spec_addprocs_function(
    definition: _TypeSpecRuntimeDefinition,
    addprocs_function: AnyValue | None,
    worker_imports: AnyValue | None,
) -> AnyValue:
    """Wrap worker creation so every new process installs the exact runtime source."""
    return _type_spec_worker_addprocs_factory()(
        addprocs_function,
        definition.source,
        "PySR." + definition.module_name,
        worker_imports,
        definition.type_spec.module_name,
        definition.module_name,
    )


@cache
def _type_spec_validator() -> AnyValue:
    return jl.seval(r"""
        function (module_, T, type_name, optimizable)
            DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
            fail(hook, message) = throw(ArgumentError("TypeSpec `$hook` $message"))
            function call(hook, f, args...)
                try
                    return f(args...)
                catch error
                    fail(hook, "failed: $(sprint(showerror, error))")
                end
            end
            function check_value(hook, value)
                value isa T || fail(hook, "must return `$type_name`.")
                valid = call("is_valid", DE.is_valid, value)
                valid isa Bool || fail("is_valid", "must return `Bool`.")
                valid || fail(hook, "returned an invalid value.")
                return value
            end
            function scalar_constant_count(value)
                count = call(
                    "count_scalar_constants", DE.count_scalar_constants, value
                )
                count isa Int && count >= 0 ||
                    fail("count_scalar_constants", "must return a nonnegative `Int`.")
                return count
            end
            function check_optimization(value, count)
                scalar_constants =
                    call("scalar_constants", module_._scalar_constants, value)
                scalar_constants isa AbstractVector ||
                    fail("scalar_constants", "must return an `AbstractVector`.")
                length(scalar_constants) == count ||
                    fail("count_scalar_constants", "disagrees with `scalar_constants`.")
                number_type = DE.get_number_type(T)
                isconcretetype(number_type) && number_type <: AbstractFloat ||
                    fail("scalar_constants", "must return a vector with a concrete `AbstractFloat` element type.")

                offset = 3
                packed = fill(number_type(NaN), count + 4)
                next_idx = call(
                    "pack_scalar_constants!",
                    DE.pack_scalar_constants!,
                    packed,
                    offset,
                    value,
                )
                next_idx == offset + count ||
                    fail("pack_scalar_constants!", "returned the wrong next index.")
                all(isnan, packed[1:(offset - 1)]) &&
                    all(isnan, packed[(offset + count):end]) ||
                    fail("pack_scalar_constants!", "wrote outside its scalar-constant range.")
                isequal(packed[offset:(offset + count - 1)], scalar_constants) ||
                    fail("pack_scalar_constants!", "disagrees with `scalar_constants`.")

                rebuilt = call(
                    "with_scalar_constants",
                    module_._with_scalar_constants,
                    value,
                    scalar_constants,
                )
                rebuilt isa T ||
                    fail("with_scalar_constants", "must return `$type_name`.")
                isequal(rebuilt, value) ||
                    fail("with_scalar_constants(value, scalar_constants(value))", "must preserve the value.")

                result = call(
                    "unpack_scalar_constants",
                    DE.unpack_scalar_constants,
                    packed,
                    offset,
                    value,
                )
                result isa Tuple && length(result) == 2 ||
                    fail("unpack_scalar_constants", "must return `(next_idx, $type_name)`.")
                unpacked_idx, unpacked = result
                unpacked_idx == offset + count ||
                    fail("unpack_scalar_constants", "returned the wrong next index.")
                unpacked isa T ||
                    fail("unpack_scalar_constants", "must return `$type_name`.")

                repacked = fill(number_type(NaN), count + 4)
                repacked_idx = call(
                    "pack_scalar_constants!",
                    DE.pack_scalar_constants!,
                    repacked,
                    offset,
                    unpacked,
                )
                repacked_idx == offset + count && isequal(packed, repacked) ||
                    fail("scalar-constant optimization hooks", "must preserve the packed scalar representation.")
            end

            rng = module_.Random.Xoshiro(0)
            sampled = check_value("sample", call("sample", SymbolicRegression.sample_value, rng, T, nothing))
            initial = check_value("init", call("init", SymbolicRegression.init_value, T))
            for value in (initial, sampled)
                count = scalar_constant_count(value)
                optimizable && check_optimization(value, count)
            end

            mutated = check_value(
                "mutate",
                call("mutate", SymbolicRegression.mutate_value, rng, sampled, 1.0, SymbolicRegression.ConstantMutation()),
            )
            count = scalar_constant_count(mutated)
            optimizable && check_optimization(mutated, count)

            call("string", module_._string, sampled) isa AbstractString ||
                fail("string", "must return an `AbstractString`.")
            return sampled
        end
        """)


def validate_type_spec_runtime(runtime: _TypeSpecRuntime) -> None:
    """Invoke TypeSpec hooks once before searching a new configuration."""
    try:
        sampled = _type_spec_validator()(
            runtime.module,
            runtime.value_type,
            runtime.spec.name,
            runtime.spec.can_optimize,
        )
    except JuliaError as error:
        raise ValueError(str(jl.sprint(jl.showerror, error.args[0]))) from error
    validate_type_spec_options(
        runtime,
        runtime.operator_functions,
        runtime.elementwise_loss,
        probe_value=sampled,
    )


@cache
def _type_spec_operator_validator() -> AnyValue:
    return jl.seval(r"""
        function (T, type_name, operator, arity, probe_value)
            inferred = Base.promote_op(operator, ntuple(_ -> T, arity)...)
            inferred === T || throw(ArgumentError(
                "TypeSpec operator `$(nameof(operator))` must be type-stable and " *
                "infer `$type_name` as its return type; inferred `$inferred`. " *
                "Add an explicit `::$type_name` return annotation if needed."
            ))
            if probe_value !== nothing
                args = ntuple(_ -> probe_value, arity)
                result = Base.invokelatest(operator, args...)
                result isa T || throw(ArgumentError(
                    "TypeSpec operator `$(nameof(operator))` must return `$type_name`."
                ))
            end
            return nothing
        end
        """)


@cache
def _type_spec_loss_validator() -> AnyValue:
    return jl.seval(r"""
        function (T, elementwise_loss, configured_loss_type, probe_value)
            loss_type = if elementwise_loss === nothing
                configured_loss_type
            else
                Base.promote_op(elementwise_loss, T, T)
            end
            isconcretetype(loss_type) && loss_type <: AbstractFloat ||
                throw(ArgumentError(
                    "The TypeSpec loss must return a concrete subtype of " *
                    "`AbstractFloat`; got `$loss_type`. Add a concrete Julia " *
                    "return type annotation."
                ))
            if probe_value !== nothing && elementwise_loss !== nothing
                loss = Base.invokelatest(
                    elementwise_loss, probe_value, probe_value
                )
                loss isa loss_type || throw(ArgumentError(
                    "The TypeSpec elementwise loss must return `$loss_type`."
                ))
            end
            return loss_type
        end
        """)


def validate_type_spec_options(
    runtime: _TypeSpecRuntime,
    operators: dict[int, tuple[AnyValue, ...]],
    elementwise_loss: AnyValue | None,
    *,
    probe_value: AnyValue | None = None,
) -> AnyValue:
    """Validate ordinary Julia options against a loaded TypeSpec."""
    try:
        for arity, functions in operators.items():
            for function in functions:
                _type_spec_operator_validator()(
                    runtime.value_type,
                    runtime.spec.name,
                    function,
                    arity,
                    probe_value,
                )
        configured_loss_type = (
            None
            if runtime.spec.loss_type is None
            else jl.Base.include_string(
                runtime.module,
                runtime.spec.loss_type,
                "PySR TypeSpec loss_type",
            )
        )
        return _type_spec_loss_validator()(
            runtime.value_type,
            elementwise_loss,
            configured_loss_type,
            probe_value,
        )
    except JuliaError as error:
        raise ValueError(str(jl.sprint(jl.showerror, error.args[0]))) from error


@cache
def _type_spec_guess_parser() -> AnyValue:
    return jl.seval(r"""
        begin
            import Random
            function (module_, options, T, variable_names, guess)
                DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
                operators = options.operators
                is_operator(name) =
                    name isa Symbol &&
                    any(ops -> any(op -> nameof(op) === name, ops), operators.ops)
                uses_variable(ex, names) =
                    ex isa Symbol ? string(ex) in names :
                    ex isa Expr ? any(arg -> uses_variable(arg, names), ex.args) : false
                function constant(value)
                    value isa T && return value
                    try
                        return value isa Tuple ? T(value...) : T(value)
                    catch
                        throw(ArgumentError(
                            "Cannot use `$value` as a `$T` constant. " *
                            "Write constants with `$T` constructor syntax."
                        ))
                    end
                end
                # Operator calls stay symbolic; every other sub-expression is a
                # constant, evaluated where the type and its helpers are defined.
                function fold(ex, names)
                    if ex isa Expr && ex.head === :call && is_operator(ex.args[1])
                        children = map(arg -> fold(arg, names), ex.args[2:end])
                        return Expr(:call, ex.args[1], children...)
                    elseif uses_variable(ex, names)
                        return ex
                    else
                        return constant(Core.eval(module_, ex))
                    end
                end
                guess isa NamedTuple || return fold(Meta.parse(guess), variable_names)

                # Template guesses: `#i` is the i-th argument of each expression.
                eval_context =
                    if SymbolicRegression.InterfaceDynamicExpressionsModule.takes_eval_context(
                        operators
                    )
                        (; eval_context=SymbolicRegression.EvalContext(;
                            options.turbo, options.bumper
                        ))
                    else
                        NamedTuple()
                    end
                contents = map(guess) do source
                    count = maximum(
                        (parse(Int, m[1]) for m in eachmatch(r"#(\d+)", source)); init=0
                    )
                    arguments = ["__arg_$i" for i in 1:count]
                    tree = DE.parse_expression(
                        fold(
                            Meta.parse(replace(source, r"#(\d+)" => s"__arg_\1")),
                            arguments,
                        );
                        operators,
                        variable_names=arguments,
                        expression_type=DE.Expression,
                        node_type=DE.with_type_parameters(options.node_type, T),
                    ).tree
                    SymbolicRegression.ComposableExpression(
                        tree; operators, variable_names=nothing, eval_context...
                    )
                end
                structure = options.expression_options.structure
                parameters = if isempty(structure.num_parameters)
                    NamedTuple()
                else
                    (;
                        parameters=SymbolicRegression.TemplateExpressionModule._initialize_template_parameters(
                            Random.default_rng(),
                            T,
                            structure.num_parameters,
                            get(options.expression_options, :parameter_initializer, nothing),
                            options,
                        ),
                    )
                end
                return SymbolicRegression.TemplateExpression(
                    contents;
                    structure,
                    operators,
                    variable_names=nothing,
                    parameters...,
                )
            end
        end
        """)


def create_type_spec_guess_parser(
    runtime: _TypeSpecRuntime, options: AnyValue, variable_names: Iterable[Any]
) -> Callable[[Any], AnyValue]:
    """Parse guesses, evaluating custom constants where the type is defined."""
    parser = _type_spec_guess_parser()
    names = jl_array([str(name) for name in variable_names])

    def parse_guess(guess: Any) -> AnyValue:
        try:
            return parser(
                runtime.configuration_module,
                options,
                runtime.value_type,
                names,
                jl_named_tuple(guess) if isinstance(guess, dict) else guess,
            )
        except JuliaError as error:
            raise ValueError(
                f"Failed to parse TypeSpec guess {guess!r}: "
                + str(jl.sprint(jl.showerror, error.args[0]))
            ) from error

    return parse_guess


def type_spec_to_julia_array(
    runtime: _TypeSpecRuntime, values: Any, *, transpose: bool = False
) -> AnyValue:
    """Convert logical Python values to an array of the generated Julia type."""
    array = (
        values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    )
    if array.dtype != object:
        array = array.astype(object)
    if transpose:
        array = array.T
    if array.ndim not in (1, 2):
        raise ValueError("TypeSpec data must be a 1D or 2D array.")

    try:
        return runtime.module._convert_array(array.ravel(order="F"), array.shape)
    except JuliaError as error:
        raise ValueError(str(jl.sprint(jl.showerror, error.args[0]))) from error


def type_spec_to_python_array(runtime: _TypeSpecRuntime, values: Any) -> np.ndarray:
    """Unwrap generated values into their logical one-field or tuple payloads."""
    field_names = tuple(runtime.spec.fields)
    output = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if len(field_names) == 1:
            output[i] = getattr(value, field_names[0])
        else:
            output[i] = tuple(getattr(value, name) for name in field_names)
    return output


class CallableJuliaExpression:
    def __init__(self, expression: AnyValue, runtime: _TypeSpecRuntime):
        self.expression = expression
        self.runtime = runtime

    def __call__(self, X: np.ndarray, *args):
        jl_X = type_spec_to_julia_array(
            self.runtime,
            object_array_2d(X),
            transpose=True,
        )
        raw_output, completed = SymbolicRegression.eval_tree_array(
            self.expression, jl_X, *args
        )
        if not bool(completed):
            raise ValueError(
                "The expression could not be evaluated over the given input: an "
                "operator returned a value that the spec's `is_valid` rejected."
            )
        return type_spec_to_python_array(self.runtime, raw_output)


def create_type_spec_exports(
    runtime: _TypeSpecRuntime,
    equations: pd.DataFrame,
    search_output: tuple[AnyValue, AnyValue],
    output_index: int | None,
) -> pd.DataFrame:
    equations = copy.deepcopy(equations)
    _, all_out_hof = search_output
    out_hof = all_out_hof[output_index] if output_index is not None else all_out_hof
    expressions = []
    callables = []
    for _, row in equations.iterrows():
        expression = out_hof.members[row["complexity"] - 1].tree
        expressions.append(expression)
        callables.append(CallableJuliaExpression(expression, runtime))
    return pd.DataFrame(
        {"julia_expression": expressions, "lambda_format": callables},
        index=equations.index,
    )


def prepare_type_spec_fit_data(
    model: Any,
    X: Any,
    y: Any,
    Xresampled: Any,
    weights: Any,
    variable_names: Any,
    complexity_of_variables: Any,
    X_units: Any,
    y_units: Any,
) -> tuple[np.ndarray, np.ndarray, None, Any, np.ndarray, Any, Any, Any]:
    if Xresampled is not None or model.denoise or model.select_k_features:
        raise NotImplementedError(
            "TypeSpec does not support resampling, denoising, or feature selection."
        )
    if isinstance(X, pd.DataFrame):
        if variable_names is not None:
            warnings.warn(
                "`variable_names` has been reset to `None` as `X` is a DataFrame."
            )
        variable_names = X.columns.astype(str).to_numpy()
        X = X.to_numpy(dtype=object)
    else:
        X = object_array_2d(X)
    if X.ndim != 2:
        raise ValueError("TypeSpec X must be a 2D array of logical values.")
    if X.shape[1] == 0:
        raise ValueError("TypeSpec X must contain at least one feature.")
    if variable_names is not None and len(variable_names) != X.shape[1]:
        raise ValueError("`variable_names` must provide one name per TypeSpec feature.")
    if variable_names is not None and any(" " in name for name in variable_names):
        variable_names = [name.replace(" ", "_") for name in variable_names]
        warnings.warn(
            "Spaces in variable names are not supported. "
            "Spaces have been replaced with underscores. \n"
            "Please use valid names instead."
        )

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy(dtype=object)
    else:
        y = object_array_1d(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise NotImplementedError("TypeSpec currently supports one output.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have inconsistent numbers of samples.")
    if X.shape[0] == 0:
        raise ValueError("X and y must contain at least one sample.")
    if weights is not None:
        raise NotImplementedError("TypeSpec does not currently support weights.")
    if X_units is not None or y_units is not None:
        raise NotImplementedError("TypeSpec does not currently support units.")

    model.n_features_in_ = X.shape[1]
    if variable_names is None:
        variable_names = np.array([f"x{i}" for i in range(X.shape[1])])
    model.feature_names_in_ = np.asarray(variable_names, dtype=str)
    model.display_feature_names_in_ = model.feature_names_in_
    model.nout_ = 1
    model.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)
    model.X_units_ = copy.deepcopy(X_units)
    model.y_units_ = copy.deepcopy(y_units)
    return (
        X,
        y,
        None,
        weights,
        model.feature_names_in_,
        complexity_of_variables,
        X_units,
        y_units,
    )


def prepare_type_spec_prediction_data(model: Any, X: Any) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        X = object_array_2d(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != model.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        X = pd.DataFrame(X)
    if isinstance(X.columns, pd.RangeIndex):
        if model.selection_mask_ is not None:
            X = X[X.columns[model.selection_mask_]]
        X.columns = model.feature_names_in_
    columns = X.columns.astype(str)
    if columns.str.contains(" ").any():
        X = X.copy()
        X.columns = columns.str.replace(" ", "_")
        warnings.warn(
            "Spaces in DataFrame column names are not supported. "
            "Spaces have been replaced with underscores. \n"
            "Please rename the columns to valid names."
        )
    X = X.rename(columns=str)
    missing_features = set(model.feature_names_in_) - set(X.columns)
    if missing_features:
        raise ValueError(f"X is missing features: {sorted(missing_features)}")
    return np.asarray(X.reindex(columns=model.feature_names_in_).to_numpy(dtype=object))


def validate_type_spec_model_configuration(model: Any) -> None:
    if model.binary_operators is not None or model.unary_operators is not None:
        raise ValueError(
            "TypeSpec requires `operators={...}` and does not accept "
            "`binary_operators` or `unary_operators`."
        )
    validate_type_spec_configuration(
        model.type_spec,
        model.operators,
        elementwise_loss=model.elementwise_loss,
        loss_function=model.loss_function,
        loss_function_expression=model.loss_function_expression,
    )
    unsupported = {
        "turbo": model.turbo,
        "bumper": model.bumper,
        "autodiff_backend": model.autodiff_backend is not None,
        "output_jax_format": model.output_jax_format,
        "output_torch_format": model.output_torch_format,
        "extra_sympy_mappings": model.extra_sympy_mappings is not None,
        "extra_jax_mappings": model.extra_jax_mappings is not None,
        "extra_torch_mappings": model.extra_torch_mappings is not None,
    }
    configured = [name for name, enabled in unsupported.items() if enabled]
    if configured:
        raise ValueError(
            "TypeSpec does not support "
            + ", ".join(f"`{name}`" for name in configured)
            + "."
        )
    expression_spec = model.expression_spec_
    if not expression_spec.supports_type_spec:
        raise ValueError(f"{type(expression_spec).__name__} does not support TypeSpec.")
    expression_spec._validate_type_spec()
