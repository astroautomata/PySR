"""Translate PySR settings to the optional Rust search engine."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from inspect import signature
from random import choices
from string import ascii_lowercase, ascii_uppercase, digits

import numpy as np
import pandas as pd

from .expression_specs import ExpressionSpec, TemplateExpressionSpec
from .mutations import _LEGACY_MUTATION_PARAMETERS
from .utils import _subscriptify

_DIRECT_OPTIONS = (
    "maxsize maxdepth niterations ncycles_per_iteration parsimony warmup_maxsize_by "
    "adaptive_parsimony_scaling use_frequency use_frequency_in_tournament alpha annealing "
    "tournament_selection_n tournament_selection_p topn fraction_replaced fraction_replaced_hof "
    "fraction_replaced_guesses migration hof_migration crossover_probability perturbation_factor "
    "probability_negate_constant skip_mutation_failures should_simplify should_optimize_constants "
    "optimizer_algorithm optimizer_nrestarts optimizer_iterations optimizer_f_calls_limit "
    "timeout_in_seconds max_evals print_precision deterministic use_tracing "
    "dimensional_constraint_penalty dimensionless_constants_only complexity_of_constants "
    "complexity_of_operators batching batch_size loss_scale "
    "nested_constraints early_stop_condition elementwise_loss populations population_size "
    "verbosity"
).split()

_JULIA_ONLY_OPTIONS = (
    "loss_function loss_function_expression complexity_mapping turbo bumper autodiff_backend "
    "logger_spec cluster_manager procs heap_size_hint_in_bytes worker_timeout worker_imports "
    "fast_cycle type_spec update"
).split()


def generate_run_id():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(choices(digits + ascii_lowercase + ascii_uppercase, k=6))
    return f"{date_str}_{suffix}"


def validate_rust_params(model):
    defaults = signature(type(model).__init__).parameters
    for name in _JULIA_ONLY_OPTIONS:
        if getattr(model, name) != defaults[name].default:
            raise ValueError(f"`{name}` is not supported with backend='rust'.")
    if model.precision not in (32, 64):
        raise ValueError("`precision` must be 32 or 64 with backend='rust'.")
    if model.parallelism == "multiprocessing":
        raise ValueError(
            "`parallelism='multiprocessing'` is not supported with backend='rust'."
        )
    if type(model.expression_spec_) not in (ExpressionSpec, TemplateExpressionSpec):
        raise ValueError(
            "`expression_spec` must be ExpressionSpec or TemplateExpressionSpec with backend='rust'."
        )
    if isinstance(model.expression_spec_, TemplateExpressionSpec):
        for name in ("output_torch_format", "output_jax_format"):
            if getattr(model, name):
                raise ValueError(
                    f"`{name}` is not supported for Rust template expressions."
                )


def _configuration(value, parameter, module):
    if not is_dataclass(value) or type(value).__module__ != module:
        raise ValueError(
            f"`{parameter}` contains a custom Julia configuration unsupported with backend='rust'."
        )
    return {"type": type(value).__name__, **asdict(value)}


def run_rust_search(model, X, y, runtime_params, weights, seed, parallelism):
    from .sr import _process_constraints, _resolve_input_stream

    try:
        from symbolic_regression_rs import equation_search  # type: ignore
    except ImportError as error:
        raise ImportError(
            "backend='rust' requires the Rust engine. "
            'Install it with `pip install "pysr[rust]"`. '
            "See https://pysr.ai/backend#rust-backend."
        ) from error

    options = {}
    for name in _DIRECT_OPTIONS:
        value = getattr(runtime_params, name, getattr(model, name))
        if value is not None:
            options[name] = value
    if model.constraints is not None:
        constrained_operators = {
            arity: [op for op in operators if op in model.constraints]
            for arity, operators in runtime_params.operators.items()
        }
        options["constraints"] = _process_constraints(
            constrained_operators, model.constraints
        )
    options.update(
        operators=runtime_params.operators,
        optimizer_probability=model.optimize_probability,
        seed=seed,
        parallelism=parallelism,
        output_directory=str(model.output_directory_),
        run_id=model.run_id_,
        tracing_file=str(model.tracing_file),
        input_stream=_resolve_input_stream(model.input_stream),
        progress=runtime_params.progress and model.verbosity > 0 and y.ndim == 1,
    )
    if model.complexity_of_variables_ is not None:
        options["complexity_of_variables"] = model.complexity_of_variables_
    mutation_weights = {
        name.removeprefix("weight_"): getattr(model, name)
        for name in _LEGACY_MUTATION_PARAMETERS
        if getattr(model, name) is not None
    }
    if mutation_weights:
        options["mutation_weights"] = mutation_weights
    for name in ("mutations", "default_mutations"):
        values = getattr(model, name)
        if values is not None:
            options[name] = [
                (_configuration(value, name, "pysr.mutations"), weight)
                for value, weight in values.items()
            ]
    for name in ("plugins", "default_plugins"):
        values = getattr(model, name)
        if values is not None:
            options[name] = [
                _configuration(value, name, "pysr.plugins") for value in values
            ]
    if isinstance(model.expression_spec_, TemplateExpressionSpec):
        options["expression_spec"] = {
            name: getattr(model.expression_spec_, name)
            for name in ("combine", "expressions", "variable_names", "parameters")
        }
    saved_state = (
        model.rust_state_
        if model.warm_start and model.rust_state_stream_ is not None
        else None
    )
    dtype = model._get_precision_mapped_dtype(X)
    state = equation_search(
        np.asarray(X, dtype=dtype),
        np.asarray(y, dtype=dtype),
        weights=None if weights is None else np.asarray(weights, dtype=dtype),
        variable_names=list(map(str, model.feature_names_in_)),
        display_variable_names=list(map(str, model.display_feature_names_in_)),
        y_variable_names=(
            [f"y{_subscriptify(i)}" for i in range(model.nout_)]
            if model.nout_ > 1
            else None
        ),
        X_units=None if model.X_units_ is None else list(model.X_units_),
        y_units=(
            model.y_units_[0]
            if model.nout_ == 1 and isinstance(model.y_units_, list)
            else model.y_units_
        ),
        guesses=model.guesses,
        saved_state=saved_state,
        **options,
    )
    model.rust_state_stream_ = pickle.dumps(state)
    model.run_id_ = state.run_id
    for message in state.warnings:
        warnings.warn(message)
    model.equations_ = model.get_hof()
    return model


@dataclass
class CallableRustExpression:
    state: object
    output: int
    complexity: int

    def __call__(self, X):
        return np.asarray(
            self.state.evaluate(
                np.asarray(X), output=self.output, complexity=self.complexity
            )
        )


def create_rust_template_exports(model, equations, output_index):
    if getattr(model, "rust_state_stream_", None) is None:
        raise ValueError(
            "Rust template expressions require the original checkpoint.pkl; CSV-only reconstruction cannot evaluate templates."
        )
    state = model.rust_state_
    return pd.DataFrame(
        {
            "lambda_format": [
                CallableRustExpression(state, output_index, int(complexity))
                for complexity in equations["complexity"]
            ]
        },
        index=equations.index,
    )
