"""Adapter for the optional Rust symbolic regression backend."""

from __future__ import annotations

from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .base import BackendSearchResult

if TYPE_CHECKING:
    from numpy import ndarray

    from pysr.sr import _DynamicallySetParams


_RUST_OPERATOR_ALIASES = {
    "-": "sub",
    "square": "abs2",
}

_RUST_BUILTIN_OPERATORS = {
    "+",
    "*",
    "/",
    "sub",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "abs",
    "abs2",
}


def _rust_only_error(param_name: str) -> NotImplementedError:
    return NotImplementedError(
        f"`{param_name}` is not supported by `backend='rust'` yet. "
        "Use `backend='julia'` for the full PySR feature set."
    )


def _normalize_operator_name(operator: str) -> str:
    return _RUST_OPERATOR_ALIASES.get(operator, operator)


def _is_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and float(value).is_integer()
        and float(value) >= 0
    )


def _validate_rust_backend_request(model: Any, weights, category) -> None:
    from pysr.expression_specs import ExpressionSpec

    if model.nout_ != 1:
        raise _rust_only_error("multi-output regression")
    if weights is not None:
        raise _rust_only_error("weights")
    if not isinstance(model.expression_spec_, ExpressionSpec):
        raise _rust_only_error("expression_spec")
    if category is not None:
        raise _rust_only_error("category")
    if model.loss_function is not None:
        raise _rust_only_error("loss_function")
    if model.loss_function_expression is not None:
        raise _rust_only_error("loss_function_expression")
    if model.elementwise_loss not in (None, "L2DistLoss()"):
        raise _rust_only_error("elementwise_loss")
    if model.constraints is not None:
        raise _rust_only_error("constraints")
    if model.nested_constraints is not None:
        raise _rust_only_error("nested_constraints")
    if model.X_units_ is not None or model.y_units_ is not None:
        raise _rust_only_error("units")
    if model.fast_cycle:
        raise _rust_only_error("fast_cycle")
    if model.turbo:
        raise _rust_only_error("turbo")
    if model.bumper:
        raise _rust_only_error("bumper")
    if model.autodiff_backend is not None:
        raise _rust_only_error("autodiff_backend")
    if model.cluster_manager is not None:
        raise _rust_only_error("cluster_manager")
    if model.worker_imports is not None:
        raise _rust_only_error("worker_imports")
    if model.logger_spec is not None:
        raise _rust_only_error("logger_spec")
    if model.warm_start:
        raise _rust_only_error("warm_start")
    if model.guesses is not None:
        raise _rust_only_error("guesses")
    if model.optimizer_algorithm != "BFGS":
        raise _rust_only_error("optimizer_algorithm")
    if model.precision not in (32, 64):
        raise _rust_only_error("precision=16")
    if model.complexity_mapping is not None:
        raise _rust_only_error("complexity_mapping")
    if model.complexity_of_operators is not None:
        raise _rust_only_error("complexity_of_operators")
    if model.complexity_of_constants is not None and not _is_nonnegative_integer(
        model.complexity_of_constants
    ):
        raise _rust_only_error("complexity_of_constants")
    if model.complexity_of_variables_ is not None and not _is_nonnegative_integer(
        model.complexity_of_variables_
    ):
        raise _rust_only_error("complexity_of_variables")
    if model.dimensional_constraint_penalty is not None:
        raise _rust_only_error("dimensional_constraint_penalty")
    if model.dimensionless_constants_only:
        raise _rust_only_error("dimensionless_constants_only")
    if isinstance(model.early_stop_condition, str):
        raise _rust_only_error("string early_stop_condition")
    if model.parallelism not in (None, "serial", "multithreading"):
        raise _rust_only_error("parallelism")
    if model.procs is not None:
        raise _rust_only_error("procs")
    if model.heap_size_hint_in_bytes is not None:
        raise _rust_only_error("heap_size_hint_in_bytes")
    if model.worker_timeout is not None:
        raise _rust_only_error("worker_timeout")


def _load_rust_module():
    try:
        return import_module("symbolic_regression_rs")
    except ImportError as exc:
        raise ImportError(
            "The Rust backend requires the optional `symbolic_regression_rs` "
            "module from the `pysr-rust-backend` package. Install PySR with "
            "`pip install 'pysr[rust]'`, or use `backend='julia'`."
        ) from exc


def _build_rust_operators(operators: dict[int, list[str]]) -> dict[int, list[str]]:
    rust_operators: dict[int, list[str]] = {}
    for arity, op_list in operators.items():
        if arity not in (1, 2):
            raise _rust_only_error(f"operators with arity {arity}")
        rust_op_list = []
        for op in op_list:
            if "(" in op:
                raise _rust_only_error("inline custom operators")
            rust_op = _normalize_operator_name(op)
            if rust_op not in _RUST_BUILTIN_OPERATORS:
                raise ValueError(
                    f"`backend='rust'` does not recognize operator {op!r}. "
                    "Use a Rust builtin operator or `backend='julia'` for "
                    "custom operators."
                )
            rust_op_list.append(rust_op)
        rust_operators[arity] = rust_op_list
    return rust_operators


def _build_rust_options(model: Any, runtime_params: Any, seed: int, X: np.ndarray):
    from pysr.sr import _get_batch_size

    batching = model.batching is True or (model.batching == "auto" and len(X) > 1000)
    batch_size = _get_batch_size(len(X), runtime_params.batch_size)

    options: dict[str, Any] = {
        "seed": int(seed),
        "niterations": int(model.niterations),
        "populations": int(model.populations),
        "population_size": int(model.population_size),
        "ncycles_per_iteration": int(model.ncycles_per_iteration),
        "batch_size": int(batch_size),
        "maxsize": int(model.maxsize),
        "maxdepth": int(runtime_params.maxdepth),
        "warmup_maxsize_by": float(runtime_params.warmup_maxsize_by),
        "parsimony": float(model.parsimony),
        "adaptive_parsimony_scaling": float(model.adaptive_parsimony_scaling),
        "crossover_probability": float(model.crossover_probability),
        "perturbation_factor": float(model.perturbation_factor),
        "probability_negate_constant": float(model.probability_negate_constant),
        "tournament_selection_n": int(model.tournament_selection_n),
        "tournament_selection_p": float(model.tournament_selection_p),
        "alpha": float(model.alpha),
        "optimizer_nrestarts": int(model.optimizer_nrestarts),
        "optimizer_probability": float(model.optimize_probability),
        "optimizer_iterations": int(model.optimizer_iterations),
        "optimizer_f_calls_limit": int(model.optimizer_f_calls_limit or 10_000),
        "fraction_replaced": float(model.fraction_replaced),
        "fraction_replaced_hof": float(model.fraction_replaced_hof),
        "topn": int(model.topn),
        "print_precision": int(model.print_precision),
        "max_evals": int(model.max_evals or 0),
        "timeout_in_seconds": float(model.timeout_in_seconds or 0.0),
        "use_frequency": bool(model.use_frequency),
        "use_frequency_in_tournament": bool(model.use_frequency_in_tournament),
        "skip_mutation_failures": bool(model.skip_mutation_failures),
        "annealing": bool(model.annealing),
        "should_optimize_constants": bool(model.should_optimize_constants),
        "migration": bool(model.migration),
        "hof_migration": bool(model.hof_migration),
        "should_simplify": bool(model.should_simplify),
        "batching": bool(batching),
        "deterministic": bool(model.deterministic),
        "parallelism": model.parallelism or "multithreading",
        "progress": bool(runtime_params.progress and model.verbosity > 0),
        "mutation_weights": {
            "mutate_constant": float(model.weight_mutate_constant),
            "mutate_operator": float(model.weight_mutate_operator),
            "mutate_feature": float(model.weight_mutate_feature),
            "swap_operands": float(model.weight_swap_operands),
            "rotate_tree": float(model.weight_rotate_tree),
            "add_node": float(model.weight_add_node),
            "insert_node": float(model.weight_insert_node),
            "delete_node": float(model.weight_delete_node),
            "simplify": float(model.weight_simplify),
            "randomize": float(model.weight_randomize),
            "do_nothing": float(model.weight_do_nothing),
            "optimize": float(model.weight_optimize),
        },
    }
    if model.complexity_of_constants is not None:
        options["complexity_of_constants"] = int(model.complexity_of_constants)
    if model.complexity_of_variables_ is not None:
        options["complexity_of_variables"] = int(model.complexity_of_variables_)
    if model.early_stop_condition is not None:
        options["early_stop_condition"] = float(model.early_stop_condition)
    return options


def _normalize_hall_of_fame(raw_result: dict[str, Any]) -> BackendSearchResult:
    rows = raw_result.get("hall_of_fame")
    if not rows:
        raise RuntimeError("Rust backend did not return any hall-of-fame equations.")

    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "Complexity": "complexity",
            "Loss": "loss",
            "Equation": "equation",
        }
    )
    required = {"complexity", "loss", "equation"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Rust backend hall-of-fame output is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.loc[:, ["complexity", "loss", "equation"]].copy()
    df["complexity"] = df["complexity"].astype(int)
    df["loss"] = df["loss"].astype(float)
    df["equation"] = df["equation"].astype(str)
    df = df.sort_values(["complexity", "loss"], kind="mergesort")
    df = df.drop_duplicates(subset=["complexity"], keep="first")
    df = df.reset_index(drop=True)
    return BackendSearchResult(
        hall_of_fame=df,
        backend_version=raw_result.get("backend_version"),
    )


def _write_hall_of_fame(model: Any, hall_of_fame: pd.DataFrame) -> None:
    equation_file = model.get_equation_file()
    Path(equation_file).parent.mkdir(parents=True, exist_ok=True)
    hall_of_fame.to_csv(equation_file, index=False)


def run_rust_backend(
    model: Any,
    X: "ndarray",
    y: "ndarray",
    runtime_params: "_DynamicallySetParams",
    *,
    weights,
    category,
    seed: int,
):
    _validate_rust_backend_request(model, weights, category)
    rust_operators = _build_rust_operators(runtime_params.operators)
    rust = _load_rust_module()

    if np.issubdtype(np.asarray(X).dtype, np.complexfloating):
        raise _rust_only_error("complex input data")

    np_dtype = model._get_precision_mapped_dtype(np.asarray(X))
    if np_dtype not in (np.float32, np.float64):
        raise _rust_only_error("non-real precision")

    X_rust = np.ascontiguousarray(np.asarray(X, dtype=np_dtype))
    y_rust = np.ascontiguousarray(np.asarray(y, dtype=np_dtype))
    options = _build_rust_options(model, runtime_params, seed, X_rust)

    raw_result = rust.search(
        X_rust,
        y_rust,
        options=options,
        operators=rust_operators,
        variable_names=[str(v) for v in model.feature_names_in_],
    )
    result = _normalize_hall_of_fame(raw_result)

    model.rust_state_ = raw_result
    model.rust_backend_version_ = result.backend_version
    model.equation_file_contents_ = [result.hall_of_fame]
    _write_hall_of_fame(model, result.hall_of_fame)
    model.equations_ = model.get_hof()
    return model
