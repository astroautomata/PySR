"""Define the PySRRegressor scikit-learn interface."""

from __future__ import annotations

import copy
import logging
import os
import pickle as pkl
import re
import sys
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, fields
from io import StringIO
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils.validation import _check_feature_names_in  # type: ignore
from sklearn.utils.validation import check_is_fitted

from .denoising import denoise, multi_denoise
from .deprecated import DEPRECATED_KWARGS
from .export_latex import (
    sympy2latex,
    sympy2latextable,
    sympy2multilatextable,
    with_preamble,
)
from .export_sympy import assert_valid_sympy_symbol
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    ParametricExpressionSpec,
    parametric_expression_deprecation_warning,
)
from .feature_selection import run_feature_selection
from .julia_extensions import load_required_packages
from .julia_helpers import (
    _escape_filename,
    _load_cluster_manager,
    jl_array,
    jl_deserialize,
    jl_is_function,
    jl_named_tuple,
    jl_serialize,
)
from .julia_import import AnyValue, SymbolicRegression, VectorValue, jl
from .logger_specs import AbstractLoggerSpec
from .utils import (
    ArrayLike,
    PathLike,
    _preprocess_julia_floats,
    _safe_check_feature_names_in,
    _subscriptify,
    _suggest_keywords,
)

try:
    from sklearn.utils.validation import validate_data

    OLD_SKLEARN = False
except ImportError:
    OLD_SKLEARN = True

try:
    from typing import List
except ImportError:
    from typing_extensions import List


ALREADY_RAN = False

pysr_logger = logging.getLogger(__name__)


def _process_constraints(
    operators: dict[int, list[str]],
    constraints: dict[str, int | tuple[int, ...]],
) -> dict[str, int | tuple[int, ...]]:
    constraints = constraints.copy()

    for arity, op_list in operators.items():
        for op in op_list:
            if op not in constraints:
                if arity == 1:
                    # Unary operators get complexity -1
                    constraints[op] = -1
                else:
                    # Multi-arity operators (arity >= 2)
                    if op in ["^", "pow"]:
                        # Warn user that they should set up constraints
                        warnings.warn(
                            "You are using the `^` operator, but have not set up `constraints` for it. "
                            "This may lead to overly complex expressions. "
                            "One typical constraint is to use `constraints={..., '^': (-1, 1)}`, which "
                            "will allow arbitrary-complexity base (-1) but only powers such as "
                            "a constant or variable (1). "
                            "For more tips, please see https://ai.damtp.cam.ac.uk/pysr/tuning/"
                        )
                    # Create default constraint tuple with -1 for each argument
                    constraints[op] = tuple([-1] * arity)

            # Apply arity-specific validation for existing constraints
            if isinstance(constraints[op], tuple):
                constraint_tuple = cast(Tuple[int, ...], constraints[op])
                # Validate that constraint tuple length matches operator arity
                if len(constraint_tuple) != arity:
                    raise ValueError(
                        f"Operator '{op}' has arity {arity} but constraint tuple has "
                        f"length {len(constraint_tuple)}. Expected tuple of length {arity}."
                    )

                # Apply operator-specific rules (only for binary operators for now)
                if arity == 2:
                    if op in ["plus", "sub", "+", "-"]:
                        if constraint_tuple[0] != constraint_tuple[1]:
                            raise NotImplementedError(
                                "You need equal constraints on both sides for - and +, "
                                "due to simplification strategies."
                            )
                    elif op in ["mult", "*"]:
                        # Make sure the complex expression is in the left side.
                        if constraint_tuple[0] == -1:
                            continue
                        if (
                            constraint_tuple[1] == -1
                            or constraint_tuple[0] < constraint_tuple[1]
                        ):
                            constraints[op] = (constraint_tuple[1], constraint_tuple[0])
    return constraints


def _maybe_create_inline_operators(
    operators: dict[int, list[str]],
    extra_sympy_mappings: dict[str, Callable] | None,
    expression_spec: AbstractExpressionSpec,
) -> dict[int, list[str]]:
    operators = {arity: op_list.copy() for arity, op_list in operators.items()}

    for arity, op_list in operators.items():
        for i, op in enumerate(op_list):
            is_user_defined_operator = "(" in op

            if is_user_defined_operator:
                jl.seval(op)
                # Cut off from the first non-alphanumeric char:
                first_non_char = [j for j, char in enumerate(op) if char == "("][0]
                function_name = op[:first_non_char]
                # Assert that function_name only contains
                # alphabetical characters, numbers,
                # and underscores:
                if not re.match(r"^[a-zA-Z0-9_]+$", function_name):
                    raise ValueError(
                        f"Invalid function name {function_name}. "
                        "Only alphanumeric characters, numbers, "
                        "and underscores are allowed."
                    )
                missing_sympy_mapping = (
                    extra_sympy_mappings is None
                    or function_name not in extra_sympy_mappings
                )
                if missing_sympy_mapping and expression_spec.supports_sympy:
                    raise ValueError(
                        f"Custom function {function_name} is not defined in `extra_sympy_mappings`. "
                        "You can define it with, "
                        "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                        "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                        "You can also define these at initialization time."
                    )
                op_list[i] = function_name
    return operators


def _check_assertions(
    X,
    use_custom_variable_names,
    variable_names,
    complexity_of_variables,
    weights,
    y,
    X_units,
    y_units,
):
    # Check for potential errors before they happen
    assert len(X.shape) == 2
    assert len(y.shape) in [1, 2]
    assert X.shape[0] == y.shape[0]
    if weights is not None:
        assert weights.shape == y.shape
        assert X.shape[0] == weights.shape[0]
    if use_custom_variable_names:
        assert len(variable_names) == X.shape[1]
        # Check none of the variable names are function names:
        for var_name in variable_names:
            # Check if alphanumeric only:
            if not re.match(r"^[₀₁₂₃₄₅₆₇₈₉a-zA-Z0-9_]+$", var_name):
                raise ValueError(
                    f"Invalid variable name {var_name}. "
                    "Only alphanumeric characters, numbers, "
                    "and underscores are allowed."
                )
            assert_valid_sympy_symbol(var_name)
    if (
        isinstance(complexity_of_variables, list)
        and len(complexity_of_variables) != X.shape[1]
    ):
        raise ValueError(
            "The number of elements in `complexity_of_variables` must equal the number of features in `X`."
        )
    if X_units is not None and len(X_units) != X.shape[1]:
        raise ValueError(
            "The number of units in `X_units` must equal the number of features in `X`."
        )
    if y_units is not None:
        good_y_units = False
        if isinstance(y_units, list):
            if len(y.shape) == 1:
                good_y_units = len(y_units) == 1
            else:
                good_y_units = len(y_units) == y.shape[1]
        else:
            good_y_units = len(y.shape) == 1 or y.shape[1] == 1

        if not good_y_units:
            raise ValueError(
                "The number of units in `y_units` must equal the number of output features in `y`."
            )


def _validate_elementwise_loss(
    custom_loss, *, has_weights: bool, probe_value: Any = 1.0
) -> None:
    """Check whether a Julia `elementwise_loss` accepts the expected inputs.

    The function probes the loss with two or three arguments, depending on
    whether weights are present, using the same dtype that fitting will use.
    If the probe fails, it raises a `ValueError` describing the expected
    signature.
    """

    # This can be either a LossFunctions.jl object (e.g. `L2DistLoss()`) or a Julia function.
    # Only validate arity when the evaluated object is actually a function.
    if not jl_is_function(custom_loss):
        return

    probe_args = (
        (probe_value, probe_value, probe_value)
        if has_weights
        else (probe_value, probe_value)
    )
    ok = bool(jl.applicable(custom_loss, *probe_args))
    if has_weights:
        if not ok:
            raise ValueError(
                "`elementwise_loss` must accept (prediction, target, weight) when `weights` is passed to `fit`."
            )
    else:
        if not ok:
            raise ValueError(
                "`elementwise_loss` must accept (prediction, target). If you intended a full objective, use "
                "`loss_function` or `loss_function_expression`."
            )


def _validate_custom_objective(
    custom_objective,
    *,
    knob,
    signature,
    other_alternative=None,
) -> None:
    if not jl_is_function(custom_objective):
        raise ValueError(f"`{knob}` must evaluate to a callable Julia function.")

    methods = jl.collect(jl.methods(custom_objective))

    def _accepts_npos(m, npos: int) -> bool:
        required_npos = int(m.nargs) - 1
        if bool(m.isva):
            return required_npos <= npos
        return required_npos == npos

    accepts_three_args = any(_accepts_npos(m, 3) for m in methods)
    accepts_two_args = any(_accepts_npos(m, 2) for m in methods)

    if not accepts_three_args and accepts_two_args:
        msg = (
            f"`{knob}` must have signature like {signature}. "
            "If you intended an elementwise loss, use `elementwise_loss`."
        )
        if other_alternative is not None:
            msg += f" If you intended the other full-objective mode, use `{other_alternative}`."
        raise ValueError(msg)

    if not accepts_three_args:
        raise ValueError(f"`{knob}` must have signature {signature}.")


def _validate_custom_full_objective(custom_full_objective) -> None:
    _validate_custom_objective(
        custom_full_objective,
        knob="loss_function",
        signature="(tree, dataset, options)",
        other_alternative="loss_function_expression",
    )


def _validate_custom_expression_objective(custom_loss_expression) -> None:
    _validate_custom_objective(
        custom_loss_expression,
        knob="loss_function_expression",
        signature="(expression, dataset, options)",
    )


def _validate_export_mappings(extra_jax_mappings, extra_torch_mappings):
    # It is expected extra_jax/torch_mappings will be updated after fit.
    # Thus, validation is performed here instead of in _validate_init_params
    if extra_jax_mappings is not None:
        for value in extra_jax_mappings.values():
            if not isinstance(value, str):
                raise ValueError(
                    "extra_jax_mappings must have keys that are strings! "
                    "e.g., {sympy.sqrt: 'jnp.sqrt'}."
                )
    if extra_torch_mappings is not None:
        for value in extra_torch_mappings.values():
            if not callable(value):
                raise ValueError(
                    "extra_torch_mappings must be callable functions! "
                    "e.g., {sympy.sqrt: torch.sqrt}."
                )


# Class validation constants
VALID_OPTIMIZER_ALGORITHMS = ["BFGS", "NelderMead"]


@dataclass
class _DynamicallySetParams:
    """Defines some parameters that are set at runtime."""

    operators: dict[int, list[str]]
    maxdepth: int
    constraints: dict[str, int | tuple[int, ...]]
    batch_size: int | None
    update_verbosity: int
    progress: bool
    warmup_maxsize_by: float


class PySRRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    High-performance symbolic regression algorithm.

    This is the scikit-learn interface for SymbolicRegression.jl.
    This model will automatically search for equations which fit
    a given dataset subject to a particular loss and set of
    constraints.

    Most default parameters have been tuned over several example equations,
    but you should adjust `niterations`, `binary_operators`, `unary_operators`
    to your requirements. You can view more detailed explanations of the options
    on the [options page](https://ai.damtp.cam.ac.uk/pysr/options) of the
    documentation.

    Parameters
    ----------
    model_selection : str
        Model selection criterion when selecting a final expression from
        the list of best expression at each complexity.
        Can be `'accuracy'`, `'best'`, or `'score'`. Default is `'best'`.
        `'accuracy'` selects the candidate model with the lowest loss
        (highest accuracy).
        `'score'` selects the candidate model with the highest score.
        Score is defined as the negated derivative of the log-loss with
        respect to complexity - if an expression has a much better
        loss at a slightly higher complexity, it is preferred.
        `'best'` selects the candidate model with the highest score
        among expressions with a loss better than at least 1.5x the
        most accurate model.
    binary_operators : list[str]
        List of strings for binary operators used in the search.
        See the [operators page](https://ai.damtp.cam.ac.uk/pysr/operators/)
        for more details.
        Default is `["+", "-", "*", "/"]`.
    unary_operators : list[str]
        Operators which only take a single scalar as input.
        For example, `"cos"` or `"exp"`.
        Default is `None`.
    operators : dict[int, list[str]]
        Generic operators by arity (number of arguments). Keys are integers
        representing arity, values are lists of operator strings.
        Example: `{1: ["sin", "cos"], 2: ["+", "-", "*"], 3: ["muladd"]}`.
        Cannot be used with `binary_operators` or `unary_operators`.
        Default is `None`.
    expression_spec : AbstractExpressionSpec
        The type of expression to search for. By default,
        this is just `ExpressionSpec()`. You can also use
        `TemplateExpressionSpec(...)` which allows you to specify
        a custom template for the expressions.
        Default is `ExpressionSpec()`.
    niterations : int
        Number of iterations of the algorithm to run. The best
        equations are printed and migrate between populations at the
        end of each iteration.
        Default is `100`.
    populations : int
        Number of populations running.
        Default is `31`.
    population_size : int
        Number of individuals in each population.
        Default is `27`.
    max_evals : int
        Limits the total number of evaluations of expressions to
        this number.  Default is `None`.
    maxsize : int
        Max complexity of an equation.  Default is `30`.
    maxdepth : int
        Max depth of an equation. You can use both `maxsize` and
        `maxdepth`. `maxdepth` is by default not used.
        Default is `None`.
    warmup_maxsize_by : float
        Whether to slowly increase max size from a small number up to
        the maxsize (if greater than 0).  If greater than 0, says the
        fraction of training time at which the current maxsize will
        reach the user-passed maxsize.
        Default is `0.0`.
    timeout_in_seconds : float
        Make the search return early once this many seconds have passed.
        Default is `None`.
    constraints : dict[str, int | tuple[int,...]]
        Dictionary of int (unary) or tuples (multi-arity), this enforces
        maxsize constraints on the individual arguments of operators.
        E.g., `'pow': (-1, 1)` says that power laws can have any
        complexity left argument, but only 1 complexity in the right
        argument. For arity-3 operators like muladd, use 3-tuples like
        `'muladd': (-1, -1, 1)` to constrain each argument's complexity.
        Use this to force more interpretable solutions.
        Default is `None`.
    nested_constraints : dict[str, dict]
        Specifies how many times a combination of operators can be
        nested. For example, `{"sin": {"cos": 0}}, "cos": {"cos": 2}}`
        specifies that `cos` may never appear within a `sin`, but `sin`
        can be nested with itself an unlimited number of times. The
        second term specifies that `cos` can be nested up to 2 times
        within a `cos`, so that `cos(cos(cos(x)))` is allowed
        (as well as any combination of `+` or `-` within it), but
        `cos(cos(cos(cos(x))))` is not allowed. When an operator is not
        specified, it is assumed that it can be nested an unlimited
        number of times. This requires that there is no operator which
        is used both in the unary operators and the binary operators
        (e.g., `-` could be both subtract, and negation). For binary
        operators, you only need to provide a single number: both
        arguments are treated the same way, and the max of each
        argument is constrained.
        Default is `None`.
    elementwise_loss : str
        String of Julia code specifying an elementwise loss function.
        Can either be a loss from LossFunctions.jl, or your own loss
        written as a function. Examples of custom written losses include:
        `myloss(x, y) = abs(x-y)` for non-weighted, or
        `myloss(x, y, w) = w*abs(x-y)` for weighted.
        The included losses include:
        Regression: `LPDistLoss{P}()`, `L1DistLoss()`,
        `L2DistLoss()` (mean square), `LogitDistLoss()`,
        `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`,
        `PeriodicLoss(c)`, `QuantileLoss(τ)`.
        Classification: `ZeroOneLoss()`, `PerceptronLoss()`,
        `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`,
        `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`,
        `SigmoidLoss()`, `DWDMarginLoss(q)`.
        Default is `"L2DistLoss()"`.
    loss_function : str
        Alternatively, you can specify the full objective function as
        a snippet of Julia code, including any sort of custom evaluation
        (including symbolic manipulations beforehand), and any sort
        of loss function or regularizations. The default `loss_function`
        used in SymbolicRegression.jl is roughly equal to:
        ```julia
        function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            return sum((prediction .- dataset.y) .^ 2) / dataset.n
        end
        ```
        where the example elementwise loss is mean-squared error.
        You may pass a function with the same arguments as this (note
        that the name of the function doesn't matter). Here,
        both `prediction` and `dataset.y` are 1D arrays of length `dataset.n`.
        Default is `None`.
    loss_function_expression : str
        Similar to `loss_function`, but takes as input the full
        expression object as the first argument, rather than
        the innermost `AbstractExpressionNode`. This is useful
        for specifying custom loss functions on `TemplateExpressionSpec`.
        Default is `None`.
    loss_scale : Literal["log", "linear"]
        Determines how loss values are scaled when computing scores.
        "log" (default) uses logarithmic scaling of loss ratios; this mode
        requires non-negative loss values and is ideal for traditional
        loss functions that are always non-negative.
        "linear" uses direct
        differences between losses; this mode handles any loss values
        (including negative) and is useful for custom loss functions,
        especially those based on likelihoods.
        Default is "log".
    complexity_of_operators : dict[str, int | float]
        If you would like to use a complexity other than 1 for an
        operator, specify the complexity here. For example,
        `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
        of the `sin` operator, and a complexity of 1 for each use of
        the `+` operator (which is the default). You may specify real
        numbers for a complexity, and the total complexity of a tree
        will be rounded to the nearest integer after computing.
        Default is `None`.
    complexity_of_constants : int | float
        Complexity of constants. Default is `1`.
    complexity_of_variables : int | float | list[int | float]
        Global complexity of variables. To set different complexities for
        different variables, pass a list of complexities to the `fit` method
        with keyword `complexity_of_variables`. You cannot use both.
        Default is `1`.
    complexity_mapping : str
        Alternatively, you can pass a function (a string of Julia code) that
        takes the expression as input and returns the complexity. Make sure that
        this operates on `AbstractExpression` (and unpacks to `AbstractExpressionNode`),
        and returns an integer.
        Default is `None`.
    parsimony : float
        Multiplicative factor for how much to punish complexity.
        Default is `0.0`.
    dimensional_constraint_penalty : float
        Additive penalty for if dimensional analysis of an expression fails.
        By default, this is `1000.0`.
    dimensionless_constants_only : bool
        Whether to only search for dimensionless constants, if using units.
        Default is `False`.
    use_frequency : bool
        Whether to measure the frequency of complexities, and use that
        instead of parsimony to explore equation space. Will naturally
        find equations of all complexities.
        Default is `True`.
    use_frequency_in_tournament : bool
        Whether to use the frequency mentioned above in the tournament,
        rather than just the simulated annealing.
        Default is `True`.
    adaptive_parsimony_scaling : float
        If the adaptive parsimony strategy (`use_frequency` and
        `use_frequency_in_tournament`), this is how much to (exponentially)
        weight the contribution. If you find that the search is only optimizing
        the most complex expressions while the simpler expressions remain stagnant,
        you should increase this value.
        Default is `1040.0`.
    alpha : float
        Initial temperature for simulated annealing
        (requires `annealing` to be `True`).
        Default is `3.17`.
    annealing : bool
        Whether to use annealing.  Default is `False`.
    early_stop_condition : float | str
        Stop the search early if this loss is reached. You may also
        pass a string containing a Julia function which
        takes a loss and complexity as input, for example:
        `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.
        Default is `None`.
    ncycles_per_iteration : int
        Number of total mutations to run, per 10 samples of the
        population, per iteration.
        Default is `380`.
    fraction_replaced : float
        How much of population to replace with migrating equations from
        other populations.
        Default is `0.00036`.
    fraction_replaced_hof : float
        How much of population to replace with migrating equations from
        hall of fame. Default is `0.0614`.
    fraction_replaced_guesses : float
        How much of the population to replace with migrating equations from
        guesses. Default is `0.001`.
    weight_add_node : float
        Relative likelihood for mutation to add a node.
        Default is `2.47`.
    weight_insert_node : float
        Relative likelihood for mutation to insert a node.
        Default is `0.0112`.
    weight_delete_node : float
        Relative likelihood for mutation to delete a node.
        Default is `0.870`.
    weight_do_nothing : float
        Relative likelihood for mutation to leave the individual.
        Default is `0.273`.
    weight_mutate_constant : float
        Relative likelihood for mutation to change the constant slightly
        in a random direction.
        Default is `0.0346`.
    weight_mutate_operator : float
        Relative likelihood for mutation to swap an operator.
        Default is `0.293`.
    weight_mutate_feature : float
        Relative likelihood for mutation to change which feature a variable node references.
        Default is `0.1`.
    weight_swap_operands : float
        Relative likehood for swapping operands in binary operators.
        Default is `0.198`.
    weight_rotate_tree : float
        How often to perform a tree rotation at a random node.
        Default is `4.26`.
    weight_randomize : float
        Relative likelihood for mutation to completely delete and then
        randomly generate the equation
        Default is `0.000502`.
    weight_simplify : float
        Relative likelihood for mutation to simplify constant parts by evaluation
        Default is `0.00209`.
    weight_optimize: float
        Constant optimization can also be performed as a mutation, in addition to
        the normal strategy controlled by `optimize_probability` which happens
        every iteration. Using it as a mutation is useful if you want to use
        a large `ncycles_periteration`, and may not optimize very often.
        Default is `0.0`.
    crossover_probability : float
        Absolute probability of crossover-type genetic operation, instead of a mutation.
        Default is `0.0259`.
    skip_mutation_failures : bool
        Whether to skip mutation and crossover failures, rather than
        simply re-sampling the current member.
        Default is `True`.
    migration : bool
        Whether to migrate.  Default is `True`.
    hof_migration : bool
        Whether to have the hall of fame migrate.  Default is `True`.
    topn : int
        How many top individuals migrate from each population.
        Default is `12`.
    should_simplify : bool
        Whether to use algebraic simplification in the search. Note that only
        a few simple rules are implemented. Default is `True`.
    should_optimize_constants : bool
        Whether to numerically optimize constants (Nelder-Mead/Newton)
        at the end of each iteration. Default is `True`.
    optimizer_algorithm : str
        Optimization scheme to use for optimizing constants. Can currently
        be `NelderMead` or `BFGS`.
        Default is `"BFGS"`.
    optimizer_nrestarts : int
        Number of time to restart the constants optimization process with
        different initial conditions.
        Default is `2`.
    optimizer_f_calls_limit : int
        How many function calls to allow during optimization.
        Default is `10_000`.
    optimize_probability : float
        Probability of optimizing the constants during a single iteration of
        the evolutionary algorithm.
        Default is `0.14`.
    optimizer_iterations : int
        Number of iterations that the constants optimizer can take.
        Default is `8`.
    perturbation_factor : float
        Constants are perturbed by a max factor of
        (perturbation_factor*T + 1). Either multiplied by this or
        divided by this.
        Default is `0.129`.
    probability_negate_constant : float
        Probability of negating a constant in the equation when mutating it.
        Default is `0.00743`.
    tournament_selection_n : int
        Number of expressions to consider in each tournament.
        Default is `15`.
    tournament_selection_p : float
        Probability of selecting the best expression in each
        tournament. The probability will decay as p*(1-p)^n for other
        expressions, sorted by loss.
        Default is `0.982`.
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None
        Parallelism to use for the search. Can be `"serial"`, `"multithreading"`, or `"multiprocessing"`.
        Default is `"multithreading"`.
    procs: int | None
        Number of processes to use for parallelism. If `None`, defaults to `cpu_count()`.
        Default is `None`.
    cluster_manager : str
        For distributed computing, this sets the job queue system. Set
        to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
        "htc". If set to one of these, PySR will run in distributed
        mode, and use `procs` to figure out how many processes to launch.
        Default is `None`.
    heap_size_hint_in_bytes : int
        For multiprocessing, this sets the `--heap-size-hint` parameter
        for new Julia processes. This can be configured when using
        multi-node distributed compute, to give a hint to each process
        about how much memory they can use before aggressive garbage
        collection.
    worker_timeout : float | None
        Timeout in seconds for worker processes during multiprocessing to respond.
        If a worker does not respond within this time, it will be restarted.
        Default is `None`.
    worker_imports : list[str] | None
        List of module names as strings to import in worker processes.
        For example, `["MyPackage", "OtherPackage"]` will run `using MyPackage, OtherPackage`
        in each worker process. Default is `None`.
    batching : bool | "auto"
        Whether to compare population members on small batches during
        evolution. Still uses full dataset for comparing against hall
        of fame. "auto" enables batching for N>1000. Default is `"auto"`.
    batch_size : int | None
        The batch size to use if batching. If None, uses the
        full dataset when N<=1000, 128 for N<5000, 256 for N<50000,
        or 512 for N≥50000. Default is `None`.
    fast_cycle : bool
        Batch over population subsamples. This is a slightly different
        algorithm than regularized evolution, but does cycles 15%
        faster. May be algorithmically less efficient.
        Default is `False`.
    turbo: bool
        (Experimental) Whether to use LoopVectorization.jl to speed up the
        search evaluation. Certain operators may not be supported.
        Does not support 16-bit precision floats.
        Default is `False`.
    bumper: bool
        (Experimental) Whether to use Bumper.jl to speed up the search
        evaluation. Does not support 16-bit precision floats.
        Default is `False`.
    precision : int
        What precision to use for the data. By default this is `32`
        (float32), but you can select `64` or `16` as well, giving
        you 64 or 16 bits of floating point precision, respectively.
        If you pass complex data, the corresponding complex precision
        will be used (i.e., `64` for complex128, `32` for complex64).
        Default is `32`.
    autodiff_backend : Literal["Zygote", "Mooncake", "Enzyme"] | None
        Which backend to use for automatic differentiation during constant
        optimization. Currently `"Zygote"`, `"Mooncake"`, and `"Enzyme"` are supported.
        The default, `None`, uses forward-mode or finite difference.
        Default is `None`.
    random_state : int, Numpy RandomState instance or None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
        Default is `None`.
    deterministic : bool
        Make a PySR search give the same result every run.
        To use this, you must turn off parallelism
        (with `parallelism="serial"`),
        and set `random_state` to a fixed seed.
        Default is `False`.
    warm_start : bool
        Tells fit to continue from where the last call to fit finished.
        If false, each call to fit will be fresh, overwriting previous results.
        Default is `False`.
    guesses : list[str] | list[list[str]] | list[dict[str, str]] | list[list[dict[str, str]]] | None
        Initial guesses for expressions to seed the search. Examples:
        `["x0 + x1", "x0^2"]`, `[["x0"], ["x1"]]` (multi-output),
        `[{"f": "#1 + #2"}]` (TemplateExpressionSpec where `#1`, `#2` are
        placeholders for the 1st, 2nd arguments of expression `f`).
        Default is `None`.
    verbosity : int
        What verbosity level to use. 0 means minimal print statements.
        Default is `1`.
    update_verbosity : int
        What verbosity level to use for package updates.
        Will take value of `verbosity` if not given.
        Default is `None`.
    print_precision : int
        How many significant digits to print for floats. Default is `5`.
    progress : bool
        Whether to use a progress bar instead of printing to stdout.
        Default is `True`.
    logger_spec: AbstractLoggerSpec | None
        Logger specification for the Julia backend. See, for example,
        `TensorBoardLoggerSpec`.
        Default is `None`.
    input_stream : str
        The stream to read user input from. By default, this is `"stdin"`.
        If you encounter issues with reading from `stdin`, like a hang,
        you can simply pass `"devnull"` to this argument. You can also
        reference an arbitrary Julia object in the `Main` namespace.
        Default is `"stdin"`.
    run_id : str
        A unique identifier for the run. Will be generated using the
        current date and time if not provided.
        Default is `None`.
    output_directory : str
        The base directory to save output files to. Files
        will be saved in a subdirectory according to the run ID.
        Will be set to `outputs/` if not provided.
        Default is `None`.
    temp_equation_file : bool
        Whether to put the hall of fame file in the temp directory.
        Deletion is then controlled with the `delete_tempfiles`
        parameter.
        Default is `False`.
    tempdir : str
        directory for the temporary files. Default is `None`.
    delete_tempfiles : bool
        Whether to delete the temporary files after finishing.
        Default is `True`.
    update: bool
        Whether to automatically update Julia packages when `fit` is called.
        You should make sure that PySR is up-to-date itself first, as
        the packaged Julia packages may not necessarily include all
        updated dependencies.
        Default is `False`.
    output_jax_format : bool
        Whether to create a 'jax_format' column in the output,
        containing jax-callable functions and the default parameters in
        a jax array.
        Default is `False`.
    output_torch_format : bool
        Whether to create a 'torch_format' column in the output,
        containing a torch module with trainable parameters.
        Default is `False`.
    extra_sympy_mappings : dict[str, Callable]
        Provides mappings between custom `binary_operators` or
        `unary_operators` defined in julia strings, to those same
        operators defined in sympy.
        E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
        model to be export to sympy, `extra_sympy_mappings`
        would be `{"inv": lambda x: 1/x}`.
        Default is `None`.
    extra_jax_mappings : dict[Callable, str]
        Similar to `extra_sympy_mappings` but for model export
        to jax. The dictionary maps sympy functions to jax functions.
        For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
        the `sympy.sin` function to the equivalent jax expression `jnp.sin`.
        Default is `None`.
    extra_torch_mappings : dict[Callable, Callable]
        The same as `extra_jax_mappings` but for model export
        to pytorch. Note that the dictionary keys should be callable
        pytorch expressions.
        For example: `extra_torch_mappings={sympy.sin: torch.sin}`.
        Default is `None`.
    denoise : bool
        Whether to use a Gaussian Process to denoise the data before
        inputting to PySR. Can help PySR fit noisy data.
        Default is `False`.
    select_k_features : int
        Whether to run feature selection in Python using random forests,
        before passing to the symbolic regression code. None means no
        feature selection; an int means select that many features.
        Default is `None`.
    **kwargs : dict
        Supports deprecated keyword arguments. Other arguments will
        result in an error.
    Attributes
    ----------
    equations_ : pandas.DataFrame | list[pandas.DataFrame]
        Processed DataFrame containing the results of model fitting.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    display_feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Pretty names of features, used only during printing.
    X_units_ : list[str] of length n_features
        Units of each variable in the training dataset, `X`.
    y_units_ : str | list[str] of length n_out
        Units of each variable in the training dataset, `y`.
    nout_ : int
        Number of output dimensions.
    selection_mask_ : ndarray of shape (`n_features_in_`,)
        Mask of which features of `X` to use when `select_k_features` is set.
    tempdir_ : Path | None
        Path to the temporary equations directory.
    julia_state_stream_ : ndarray
        The serialized state for the julia SymbolicRegression.jl backend (after fitting),
        stored as an array of uint8, produced by Julia's Serialization.serialize function.
    julia_options_stream_ : ndarray
        The serialized julia options, stored as an array of uint8,
    logger_ : AnyValue | None
        The logger instance used for this fit, if any.
    expression_spec_ : AbstractExpressionSpec
        The expression specification used for this fit. This is equal to
        `self.expression_spec` if provided, or `ExpressionSpec()` otherwise.
    equation_file_contents_ : list[pandas.DataFrame]
        Contents of the equation file output by the Julia backend.
    show_pickle_warnings_ : bool
        Whether to show warnings about what attributes can be pickled.

    Examples
    --------
    ```python
    >>> import numpy as np
    >>> from pysr import PySRRegressor
    >>> randstate = np.random.RandomState(0)
    >>> X = 2 * randstate.randn(100, 5)
    >>> # y = 2.5382 * cos(x_3) + x_0 - 0.5
    >>> y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    >>> model = PySRRegressor(
    ...     niterations=40,
    ...     binary_operators=["+", "*"],
    ...     unary_operators=[
    ...         "cos",
    ...         "exp",
    ...         "sin",
    ...         "inv(x) = 1/x",  # Custom operator (julia syntax)
    ...     ],
    ...     model_selection="best",
    ...     elementwise_loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
    ... )
    >>> model.fit(X, y)
    >>> model
    PySRRegressor.equations_ = [
    0         0.000000                                          3.8552167  3.360272e+01           1
    1         1.189847                                          (x0 * x0)  3.110905e+00           3
    2         0.010626                          ((x0 * x0) + -0.25573406)  3.045491e+00           5
    3         0.896632                              (cos(x3) + (x0 * x0))  1.242382e+00           6
    4         0.811362                ((x0 * x0) + (cos(x3) * 2.4384754))  2.451971e-01           8
    5  >>>>  13.733371          (((cos(x3) * 2.5382) + (x0 * x0)) + -0.5)  2.889755e-13          10
    6         0.194695  ((x0 * x0) + (((cos(x3) + -0.063180044) * 2.53...  1.957723e-13          12
    7         0.006988  ((x0 * x0) + (((cos(x3) + -0.32505524) * 1.538...  1.944089e-13          13
    8         0.000955  (((((x0 * x0) + cos(x3)) + -0.8251649) + (cos(...  1.940381e-13          15
    ]
    >>> model.score(X, y)
    1.0
    >>> model.predict(np.array([1,2,3,4,5]))
    array([-1.15907818, -1.15907818, -1.15907818, -1.15907818, -1.15907818])
    ```
    """

    equations_: pd.DataFrame | list[pd.DataFrame] | None
    n_features_in_: int
    feature_names_in_: ArrayLike[str]
    display_feature_names_in_: ArrayLike[str]
    complexity_of_variables_: int | float | list[int | float] | None
    X_units_: ArrayLike[str] | None
    y_units_: str | ArrayLike[str] | None
    nout_: int
    selection_mask_: NDArray[np.bool_] | None
    run_id_: str
    output_directory_: str
    julia_state_stream_: NDArray[np.uint8] | None
    julia_options_stream_: NDArray[np.uint8] | None
    logger_: AnyValue | None
    equation_file_contents_: list[pd.DataFrame] | None
    show_pickle_warnings_: bool

    def __init__(
        self,
        model_selection: Literal["best", "accuracy", "score"] = "best",
        *,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        operators: dict[int, list[str]] | None = None,
        expression_spec: AbstractExpressionSpec | None = None,
        niterations: int = 100,
        populations: int = 31,
        population_size: int = 27,
        max_evals: int | None = None,
        maxsize: int = 30,
        maxdepth: int | None = None,
        warmup_maxsize_by: float | None = None,
        timeout_in_seconds: float | None = None,
        constraints: dict[str, int | tuple[int, ...]] | None = None,
        nested_constraints: dict[str, dict[str, int]] | None = None,
        elementwise_loss: str | None = None,
        loss_function: str | None = None,
        loss_function_expression: str | None = None,
        loss_scale: Literal["log", "linear"] = "log",
        complexity_of_operators: dict[str, int | float] | None = None,
        complexity_of_constants: int | float | None = None,
        complexity_of_variables: int | float | list[int | float] | None = None,
        complexity_mapping: str | None = None,
        parsimony: float = 0.0,
        dimensional_constraint_penalty: float | None = None,
        dimensionless_constants_only: bool = False,
        use_frequency: bool = True,
        use_frequency_in_tournament: bool = True,
        adaptive_parsimony_scaling: float = 1040.0,
        alpha: float = 3.17,
        annealing: bool = False,
        early_stop_condition: float | str | None = None,
        ncycles_per_iteration: int = 380,
        fraction_replaced: float = 0.00036,
        fraction_replaced_hof: float = 0.0614,
        fraction_replaced_guesses: float = 0.001,
        weight_add_node: float = 2.47,
        weight_insert_node: float = 0.0112,
        weight_delete_node: float = 0.870,
        weight_do_nothing: float = 0.273,
        weight_mutate_constant: float = 0.0346,
        weight_mutate_operator: float = 0.293,
        weight_mutate_feature: float = 0.1,
        weight_swap_operands: float = 0.198,
        weight_rotate_tree: float = 4.26,
        weight_randomize: float = 0.000502,
        weight_simplify: float = 0.00209,
        weight_optimize: float = 0.0,
        crossover_probability: float = 0.0259,
        skip_mutation_failures: bool = True,
        migration: bool = True,
        hof_migration: bool = True,
        topn: int = 12,
        should_simplify: bool = True,
        should_optimize_constants: bool = True,
        optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS",
        optimizer_nrestarts: int = 2,
        optimizer_f_calls_limit: int | None = None,
        optimize_probability: float = 0.14,
        optimizer_iterations: int = 8,
        perturbation_factor: float = 0.129,
        probability_negate_constant: float = 0.00743,
        tournament_selection_n: int = 15,
        tournament_selection_p: float = 0.982,
        parallelism: (
            Literal["serial", "multithreading", "multiprocessing"] | None
        ) = None,
        procs: int | None = None,
        cluster_manager: (
            Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"] | None
        ) = None,
        heap_size_hint_in_bytes: int | None = None,
        worker_timeout: float | None = None,
        worker_imports: list[str] | None = None,
        batching: bool | Literal["auto"] = "auto",
        batch_size: int | None = None,
        fast_cycle: bool = False,
        turbo: bool = False,
        bumper: bool = False,
        precision: Literal[16, 32, 64] = 32,
        autodiff_backend: Literal["Zygote", "Mooncake", "Enzyme"] | None = None,
        random_state: int | np.random.RandomState | None = None,
        deterministic: bool = False,
        warm_start: bool = False,
        guesses: (
            list[str]
            | list[list[str]]
            | list[dict[str, str]]
            | list[list[dict[str, str]]]
            | None
        ) = None,
        verbosity: int = 1,
        update_verbosity: int | None = None,
        print_precision: int = 5,
        progress: bool = True,
        logger_spec: AbstractLoggerSpec | None = None,
        input_stream: str = "stdin",
        run_id: str | None = None,
        output_directory: str | None = None,
        temp_equation_file: bool = False,
        tempdir: str | None = None,
        delete_tempfiles: bool = True,
        update: bool = False,
        output_jax_format: bool = False,
        output_torch_format: bool = False,
        extra_sympy_mappings: dict[str, Callable] | None = None,
        extra_torch_mappings: dict[Callable, Callable] | None = None,
        extra_jax_mappings: dict[Callable, str] | None = None,
        denoise: bool = False,
        select_k_features: int | None = None,
        **kwargs,
    ):
        # Hyperparameters
        # - Model search parameters
        self.model_selection = model_selection
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.operators = operators
        self.expression_spec = expression_spec
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.ncycles_per_iteration = ncycles_per_iteration
        # - Equation Constraints
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.warmup_maxsize_by = warmup_maxsize_by
        self.should_simplify = should_simplify
        # - Early exit conditions:
        self.max_evals = max_evals
        self.timeout_in_seconds = timeout_in_seconds
        self.early_stop_condition = early_stop_condition
        # - Loss parameters
        self.elementwise_loss = elementwise_loss
        self.loss_function = loss_function
        self.loss_function_expression = loss_function_expression
        self.loss_scale = loss_scale
        self.complexity_of_operators = complexity_of_operators
        self.complexity_of_constants = complexity_of_constants
        self.complexity_of_variables = complexity_of_variables
        self.complexity_mapping = complexity_mapping
        self.parsimony = parsimony
        self.dimensional_constraint_penalty = dimensional_constraint_penalty
        self.dimensionless_constants_only = dimensionless_constants_only
        self.use_frequency = use_frequency
        self.use_frequency_in_tournament = use_frequency_in_tournament
        self.adaptive_parsimony_scaling = adaptive_parsimony_scaling
        self.alpha = alpha
        self.annealing = annealing
        # - Evolutionary search parameters
        # -- Mutation parameters
        self.weight_add_node = weight_add_node
        self.weight_insert_node = weight_insert_node
        self.weight_delete_node = weight_delete_node
        self.weight_do_nothing = weight_do_nothing
        self.weight_mutate_constant = weight_mutate_constant
        self.weight_mutate_operator = weight_mutate_operator
        self.weight_mutate_feature = weight_mutate_feature
        self.weight_swap_operands = weight_swap_operands
        self.weight_rotate_tree = weight_rotate_tree
        self.weight_randomize = weight_randomize
        self.weight_simplify = weight_simplify
        self.weight_optimize = weight_optimize
        self.crossover_probability = crossover_probability
        self.skip_mutation_failures = skip_mutation_failures
        # -- Migration parameters
        self.migration = migration
        self.hof_migration = hof_migration
        self.fraction_replaced = fraction_replaced
        self.fraction_replaced_hof = fraction_replaced_hof
        self.fraction_replaced_guesses = fraction_replaced_guesses
        self.topn = topn
        # -- Constants parameters
        self.should_optimize_constants = should_optimize_constants
        self.optimizer_algorithm = optimizer_algorithm
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimizer_f_calls_limit = optimizer_f_calls_limit
        self.optimize_probability = optimize_probability
        self.optimizer_iterations = optimizer_iterations
        self.perturbation_factor = perturbation_factor
        self.probability_negate_constant = probability_negate_constant
        # -- Selection parameters
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        # -- Performance parameters
        self.parallelism = parallelism
        self.procs = procs
        self.cluster_manager = cluster_manager
        self.heap_size_hint_in_bytes = heap_size_hint_in_bytes
        self.worker_timeout = worker_timeout
        self.worker_imports = worker_imports
        self.batching = batching
        self.batch_size = batch_size
        self.fast_cycle = fast_cycle
        self.turbo = turbo
        self.bumper = bumper
        self.precision = precision
        self.autodiff_backend = autodiff_backend
        self.random_state = random_state
        self.deterministic = deterministic
        self.warm_start = warm_start
        self.guesses = guesses
        # Additional runtime parameters
        # - Runtime user interface
        self.verbosity = verbosity
        self.update_verbosity = update_verbosity
        self.print_precision = print_precision
        self.progress = progress
        self.logger_spec = logger_spec
        self.input_stream = input_stream
        # - Project management
        self.run_id = run_id
        self.output_directory = output_directory
        self.temp_equation_file = temp_equation_file
        self.tempdir = tempdir
        self.delete_tempfiles = delete_tempfiles
        self.update = update
        self.output_jax_format = output_jax_format
        self.output_torch_format = output_torch_format
        self.extra_sympy_mappings = extra_sympy_mappings
        self.extra_jax_mappings = extra_jax_mappings
        self.extra_torch_mappings = extra_torch_mappings
        # Pre-modelling transformation
        self.denoise = denoise
        self.select_k_features = select_k_features

        # Once all valid parameters have been assigned handle the
        # deprecated kwargs
        if len(kwargs) > 0:  # pragma: no cover
            for k, v in kwargs.items():
                # Handle renamed kwargs
                if k in DEPRECATED_KWARGS:
                    updated_kwarg_name = DEPRECATED_KWARGS[k]
                    setattr(self, updated_kwarg_name, v)
                    warnings.warn(
                        f"`{k}` has been renamed to `{updated_kwarg_name}` in PySRRegressor. "
                        "Please use that instead.",
                        FutureWarning,
                    )
                elif k == "multithreading":
                    # Specific advice given in `_map_parallelism_params`
                    self.multithreading: bool | None = v
                # Handle kwargs that have been moved to the fit method
                elif k in ["weights", "variable_names", "Xresampled"]:
                    warnings.warn(
                        f"`{k}` is a data-dependent parameter and should be passed when fit is called. "
                        f"Ignoring parameter; please pass `{k}` during the call to fit instead.",
                        FutureWarning,
                    )
                elif k == "julia_project":
                    warnings.warn(
                        "The `julia_project` parameter has been deprecated. To use a custom "
                        "julia project, please see `https://ai.damtp.cam.ac.uk/pysr/backend`.",
                        FutureWarning,
                    )
                elif k == "julia_kwargs":
                    warnings.warn(
                        "The `julia_kwargs` parameter has been deprecated. To pass custom "
                        "keyword arguments to the julia backend, you should use environment variables. "
                        "See the Julia documentation for more information.",
                        FutureWarning,
                    )
                else:
                    suggested_keywords = _suggest_keywords(PySRRegressor, k)
                    err_msg = (
                        f"`{k}` is not a valid keyword argument for PySRRegressor."
                    )
                    if len(suggested_keywords) > 0:
                        err_msg += f" Did you mean {', '.join(map(lambda s: f'`{s}`', suggested_keywords))}?"
                    raise TypeError(err_msg)

    @classmethod
    def from_file(
        cls,
        equation_file: None = None,  # Deprecated
        *,
        run_directory: PathLike,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        operators: dict[int, list[str]] | None = None,
        n_features_in: int | None = None,
        feature_names_in: ArrayLike[str] | None = None,
        selection_mask: NDArray[np.bool_] | None = None,
        nout: int = 1,
        **pysr_kwargs,
    ) -> "PySRRegressor":
        """
        Create a model from a saved model checkpoint or equation file.

        Parameters
        ----------
        run_directory : str
            The directory containing outputs from a previous run.
            This is of the form `[output_directory]/[run_id]`.
            Default is `None`.
        binary_operators : list[str]
            The same binary operators used when creating the model.
            Not needed if loading from a pickle file.
        unary_operators : list[str]
            The same unary operators used when creating the model.
            Not needed if loading from a pickle file.
        operators : dict[int, list[str]]
            Operator mapping by arity used when creating the model. Provide this if the
            original run relied on the generic `operators` parameter. Not needed if
            loading from a pickle file.
        n_features_in : int
            Number of features passed to the model.
            Not needed if loading from a pickle file.
        feature_names_in : list[str]
            Names of the features passed to the model.
            Not needed if loading from a pickle file.
        selection_mask : NDArray[np.bool_]
            If using `select_k_features`, you must pass `model.selection_mask_` here.
            Not needed if loading from a pickle file.
        nout : int
            Number of outputs of the model.
            Not needed if loading from a pickle file.
            Default is `1`.
        **pysr_kwargs : dict
            Any other keyword arguments to initialize the PySRRegressor object.
            These will overwrite those stored in the pickle file.
            Not needed if loading from a pickle file.

        Returns
        -------
        model : PySRRegressor
            The model with fitted equations.
        """
        if equation_file is not None:
            raise ValueError(
                "Passing `equation_file` is deprecated and no longer compatible with "
                "the most recent versions of PySR's backend. Please pass `run_directory` "
                "instead, which contains all checkpoint files."
            )

        pkl_filename = Path(run_directory) / "checkpoint.pkl"
        if pkl_filename.exists():
            pysr_logger.info(f"Attempting to load model from {pkl_filename}...")
            assert binary_operators is None
            assert unary_operators is None
            assert operators is None
            assert n_features_in is None
            with open(pkl_filename, "rb") as f:
                model = cast("PySRRegressor", pkl.load(f))

            # Update any parameters if necessary, such as
            # extra_sympy_mappings:
            model.set_params(**pysr_kwargs)

            if "equations_" not in model.__dict__ or model.equations_ is None:
                model.refresh()

            if model.expression_spec is not None:
                warnings.warn(
                    "Loading model from checkpoint file with a non-default expression spec "
                    "is not fully supported as it relies on dynamic objects. This may result in unexpected behavior.",
                )

            return model
        else:
            pysr_logger.info(
                f"Checkpoint file {pkl_filename} does not exist. "
                "Attempting to recreate model from scratch..."
            )
            csv_filename = Path(run_directory) / "hall_of_fame.csv"
            csv_filename_bak = Path(run_directory) / "hall_of_fame.csv.bak"
            if not csv_filename.exists() and not csv_filename_bak.exists():
                raise FileNotFoundError(
                    f"Hall of fame file `{csv_filename}` or `{csv_filename_bak}` does not exist. "
                    "Please pass a `run_directory` containing a valid checkpoint file."
                )
            if (
                operators is None
                and binary_operators is None
                and unary_operators is None
            ):
                raise ValueError(
                    "When recreating a model from CSV backups you must provide either "
                    "`operators` or legacy `binary_operators`/`unary_operators`."
                )
            assert n_features_in is not None
            model = cls(
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                operators=operators,
                **pysr_kwargs,
            )
            model.nout_ = nout
            model.n_features_in_ = n_features_in

            if feature_names_in is None:
                model.feature_names_in_ = np.array(
                    [f"x{i}" for i in range(n_features_in)]
                )
                model.display_feature_names_in_ = np.array(
                    [f"x{_subscriptify(i)}" for i in range(n_features_in)]
                )
            else:
                assert len(feature_names_in) == n_features_in
                model.feature_names_in_ = feature_names_in
                model.display_feature_names_in_ = feature_names_in

            if selection_mask is None:
                model.selection_mask_ = np.ones(n_features_in, dtype=np.bool_)
            else:
                model.selection_mask_ = selection_mask

            model.refresh(run_directory=run_directory)

            return model

    def __repr__(self) -> str:
        """
        Print all current equations fitted by the model.

        The string `>>>>` denotes which equation is selected by the
        `model_selection`.
        """
        if not hasattr(self, "equations_") or self.equations_ is None:
            return "PySRRegressor.equations_ = None"

        output = "PySRRegressor.equations_ = [\n"

        equations = self.equations_
        if not isinstance(equations, list):
            all_equations = [equations]
        else:
            all_equations = equations

        for i, equations in enumerate(all_equations):
            selected = pd.Series([""] * len(equations), index=equations.index)
            chosen_row = idx_model_selection(equations, self.model_selection)
            selected[chosen_row] = ">>>>"
            repr_equations = pd.DataFrame(
                dict(
                    pick=selected,
                    **(
                        {"score": equations["score"]}
                        if "score" in equations.columns
                        else {}
                    ),
                    equation=equations["equation"],
                    loss=equations["loss"],
                    complexity=equations["complexity"],
                )
            )

            if len(all_equations) > 1:
                output += "[\n"

            for line in repr_equations.__repr__().split("\n"):
                output += "\t" + line + "\n"

            if len(all_equations) > 1:
                output += "]"

            if i < len(all_equations) - 1:
                output += ", "

        output += "]"
        return output

    def __getstate__(self) -> dict[str, Any]:
        """
        Handle pickle serialization for PySRRegressor.

        The Scikit-learn standard requires estimators to be serializable via
        `pickle.dumps()`. However, some attributes do not support pickling
        and need to be hidden, such as the JAX and Torch representations.
        """
        state = self.__dict__
        show_pickle_warning = not (
            "show_pickle_warnings_" in state and not state["show_pickle_warnings_"]
        )
        state_keys_containing_lambdas = [
            "extra_sympy_mappings",
            "extra_torch_mappings",
            "extra_jax_mappings",
        ]
        for state_key in state_keys_containing_lambdas:
            warn_msg = (
                f"`{state_key}` cannot be pickled and will be removed from the "
                "serialized instance. When loading the model, please redefine "
                f"`{state_key}` at runtime."
            )
            if state[state_key] is not None:
                if show_pickle_warning:
                    warnings.warn(warn_msg)
                else:
                    pysr_logger.debug(warn_msg)
        state_keys_to_clear = state_keys_containing_lambdas
        state_keys_to_clear.append("logger_")
        pickled_state = {
            key: (None if key in state_keys_to_clear else value)
            for key, value in state.items()
        }
        if ("equations_" in pickled_state) and (
            pickled_state["equations_"] is not None
        ):
            pickled_state["output_torch_format"] = False
            pickled_state["output_jax_format"] = False
            if self.nout_ == 1:
                pickled_columns = ~pickled_state["equations_"].columns.isin(
                    ["jax_format", "torch_format"]
                )
                pickled_state["equations_"] = (
                    pickled_state["equations_"].loc[:, pickled_columns].copy()
                )
            else:
                pickled_columns = [
                    ~dataframe.columns.isin(["jax_format", "torch_format"])
                    for dataframe in pickled_state["equations_"]
                ]
                pickled_state["equations_"] = [
                    dataframe.loc[:, signle_pickled_columns]
                    for dataframe, signle_pickled_columns in zip(
                        pickled_state["equations_"], pickled_columns
                    )
                ]
        return pickled_state

    def _checkpoint(self):
        """Save the model's current state to a checkpoint file.

        This should only be used internally by PySRRegressor.
        """
        # Save model state:
        self.show_pickle_warnings_ = False
        with open(self.get_pkl_filename(), "wb") as f:
            try:
                pkl.dump(self, f)
            except Exception as e:
                pysr_logger.warning(
                    f"Error checkpointing model to {self.get_pkl_filename()}: {e}"
                )
        self.show_pickle_warnings_ = True

    def get_pkl_filename(self) -> Path:
        path = Path(self.output_directory_) / self.run_id_ / "checkpoint.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def equations(self):  # pragma: no cover
        warnings.warn(
            "PySRRegressor.equations is now deprecated. "
            "Please use PySRRegressor.equations_ instead.",
            FutureWarning,
        )
        return self.equations_

    @property
    def julia_options_(self):
        """The deserialized julia options."""
        return jl_deserialize(self.julia_options_stream_)

    @property
    def julia_state_(self):
        """The deserialized state."""
        return cast(
            Union[Tuple[VectorValue, AnyValue], None],
            jl_deserialize(self.julia_state_stream_),
        )

    @property
    def raw_julia_state_(self):
        warnings.warn(
            "PySRRegressor.raw_julia_state_ is now deprecated. "
            "Please use PySRRegressor.julia_state_ instead, or julia_state_stream_ "
            "for the raw stream of bytes.",
            FutureWarning,
        )
        return self.julia_state_

    @property
    def expression_spec_(self):
        return self.expression_spec or ExpressionSpec()

    def get_best(
        self, index: int | list[int] | None = None
    ) -> pd.Series | list[pd.Series]:
        """
        Get best equation using `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from `self.equations_`,
            give the row number here. This overrides the `model_selection`
            parameter. If there are multiple output features, then pass
            a list of indices with the order the same as the output feature.

        Returns
        -------
        best_equation : pandas.Series
            Dictionary representing the best expression found.

        Raises
        ------
        NotImplementedError
            Raised when an invalid model selection strategy is provided.
        """
        check_is_fitted(self, attributes=["equations_"])

        if index is not None:
            if isinstance(self.equations_, list):
                assert isinstance(
                    index, list
                ), "With multiple output features, index must be a list."
                return [eq.iloc[i] for eq, i in zip(self.equations_, index)]
            else:
                equations_ = cast(pd.DataFrame, self.equations_)
                return cast(pd.Series, equations_.iloc[index])

        if isinstance(self.equations_, list):
            return [
                cast(pd.Series, eq.loc[idx_model_selection(eq, self.model_selection)])
                for eq in self.equations_
            ]
        else:
            equations_ = cast(pd.DataFrame, self.equations_)
            return cast(
                pd.Series,
                equations_.loc[idx_model_selection(equations_, self.model_selection)],
            )

    @property
    def equation_file_(self):
        raise NotImplementedError(
            "PySRRegressor.equation_file_ is now deprecated. "
            "Please use PySRRegressor.output_directory_ and PySRRegressor.run_id_ "
            "instead. For loading, you should pass `run_directory`."
        )

    def _setup_equation_file(self):
        """Set the pathname of the output directory."""
        if self.warm_start and (
            hasattr(self, "run_id_") or hasattr(self, "output_directory_")
        ):
            assert hasattr(self, "output_directory_")
            assert hasattr(self, "run_id_")
            if self.run_id is not None:
                assert self.run_id_ == self.run_id
            if self.output_directory is not None:
                assert self.output_directory_ == self.output_directory
        else:
            if self.temp_equation_file:
                if self.tempdir is not None:
                    Path(self.tempdir).mkdir(parents=True, exist_ok=True)
                self.output_directory_ = tempfile.mkdtemp(dir=self.tempdir)
            else:
                self.output_directory_ = (
                    "outputs"
                    if self.output_directory is None
                    else self.output_directory
                )
            self.run_id_ = (
                cast(str, SymbolicRegression.SearchUtilsModule.generate_run_id())
                if self.run_id is None
                else self.run_id
            )
            if self.temp_equation_file:
                assert self.output_directory is None

    def _clear_equation_file_contents(self):
        self.equation_file_contents_ = None

    def _validate_and_modify_params(self) -> _DynamicallySetParams:
        """
        Ensure parameters passed at initialization are valid.

        Also returns a dictionary of parameters to update from their
        values given at initialization.

        Returns
        -------
        packed_modified_params : dict
            Dictionary of parameters to modify from their initialized
            values. For example, default parameters are set here
            when a parameter is left set to `None`.
        """
        # Immutable parameter validation
        # Ensure instance parameters are allowable values:

        # Validate operators vs binary_operators/unary_operators mutual exclusion
        if self.operators is not None:
            if self.binary_operators is not None or self.unary_operators is not None:
                raise ValueError(
                    "Cannot use `operators` with `binary_operators` or `unary_operators`. "
                    "Use either the generic `operators` parameter or the specific operator parameters."
                )
        else:
            if self.binary_operators is None and self.unary_operators is None:
                # Neither operators nor binary/unary specified, use defaults
                pass
            # If binary_operators or unary_operators is specified, that's fine
        if self.tournament_selection_n > self.population_size:
            raise ValueError(
                "`tournament_selection_n` parameter must be smaller than `population_size`."
            )

        if self.maxsize > 40:
            warnings.warn(
                "Note: Using a large maxsize for the equation search will be "
                "exponentially slower and use significant memory."
            )
        elif self.maxsize < 7:
            raise ValueError("PySR requires a maxsize of at least 7")

        # NotImplementedError - Values that could be supported at a later time
        if self.optimizer_algorithm not in VALID_OPTIMIZER_ALGORITHMS:
            raise NotImplementedError(
                f"PySR currently only supports the following optimizer algorithms: {VALID_OPTIMIZER_ALGORITHMS}"
            )

        param_container = _DynamicallySetParams(
            operators={2: ["+", "*", "-", "/"]},
            maxdepth=self.maxsize,
            constraints={},
            batch_size=None,
            update_verbosity=int(self.verbosity),
            progress=self.progress,
            warmup_maxsize_by=0.0,
        )

        # Convert binary_operators/unary_operators to operators format if needed
        if self.operators is None:
            # Build operators dict from binary_operators and unary_operators
            operators_dict = {}
            if self.binary_operators is not None:
                operators_dict[2] = self.binary_operators.copy()
            else:
                # Keep default binary operators
                operators_dict[2] = ["+", "*", "-", "/"]
            if self.unary_operators is not None:
                operators_dict[1] = self.unary_operators.copy()
            param_container.operators = operators_dict

        for param_name in map(lambda x: x.name, fields(_DynamicallySetParams)):
            user_param_value = getattr(self, param_name)
            if user_param_value is None and param_name != "batch_size":
                # Leave as the default in DynamicallySetParams
                # (except for batch_size, which we want to keep as None)
                ...
            else:
                # If user has specified it, we will override the default.
                # However, there are some special cases to mutate it:
                new_param_value = _mutate_parameter(param_name, user_param_value)
                setattr(param_container, param_name, new_param_value)
        # TODO: This should just be part of the __init__ of _DynamicallySetParams

        assert param_container.operators and any(
            len(ops) > 0 for ops in param_container.operators.values()
        ), "At least one operator must be provided."

        return param_container

    def _validate_and_set_fit_params(
        self,
        X,
        y,
        Xresampled,
        weights,
        variable_names,
        complexity_of_variables,
        X_units,
        y_units,
    ) -> tuple[
        ndarray,
        ndarray,
        ndarray | None,
        ndarray | None,
        ArrayLike[str],
        int | float | list[int | float] | None,
        ArrayLike[str] | None,
        str | ArrayLike[str] | None,
    ]:
        """
        Validate the parameters passed to the :term`fit` method.

        This method also sets the `nout_` attribute.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape `(n_samples, n_features)`.
        y : ndarray | pandas.DataFrame}
            Target values of shape `(n_samples,)` or `(n_samples, n_targets)`.
            Will be cast to `X`'s dtype if necessary.
        Xresampled : ndarray | pandas.DataFrame
            Resampled training data used for denoising,
            of shape `(n_resampled, n_features)`.
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of y.
        variable_names : ndarray of length n_features
            Names of each feature in the training dataset, `X`.
        complexity_of_variables : int | float | list[int | float]
            Complexity of each feature in the training dataset, `X`.
        X_units : list[str] of length n_features
            Units of each feature in the training dataset, `X`.
        y_units : str | list[str] of length n_out
            Units of each feature in the training dataset, `y`.

        Returns
        -------
        X_validated : ndarray of shape (n_samples, n_features)
            Validated training data.
        y_validated : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Validated target data.
        Xresampled : ndarray of shape (n_resampled, n_features)
            Validated resampled training data used for denoising.
        variable_names_validated : list[str] of length n_features
            Validated list of variable names for each feature in `X`.
        X_units : list[str] of length n_features
            Validated units for `X`.
        y_units : str | list[str] of length n_out
            Validated units for `y`.

        """
        if isinstance(X, pd.DataFrame):
            if variable_names:
                variable_names = None
                warnings.warn(
                    "`variable_names` has been reset to `None` as `X` is a DataFrame. "
                    "Using DataFrame column names instead."
                )

            cols_str = X.columns.astype(str)
            if cols_str.str.contains(" ").any():
                X.columns = cols_str.str.replace(" ", "_")
                warnings.warn(
                    "Spaces in DataFrame column names are not supported. "
                    "Spaces have been replaced with underscores. \n"
                    "Please rename the columns to valid names."
                )
        elif variable_names and any([" " in name for name in variable_names]):
            variable_names = [name.replace(" ", "_") for name in variable_names]
            warnings.warn(
                "Spaces in `variable_names` are not supported. "
                "Spaces have been replaced with underscores. \n"
                "Please use valid names instead."
            )

        if (
            complexity_of_variables is not None
            and self.complexity_of_variables is not None
        ):
            raise ValueError(
                "You cannot set `complexity_of_variables` at both `fit` and `__init__`. "
                "Pass it at `__init__` to set it to global default, OR use `fit` to set it for "
                "each variable individually."
            )
        elif complexity_of_variables is not None:
            complexity_of_variables = complexity_of_variables
        elif self.complexity_of_variables is not None:
            complexity_of_variables = self.complexity_of_variables
        else:
            complexity_of_variables = None

        # Data validation and feature name fetching via sklearn
        # This method sets the n_features_in_ attribute
        if Xresampled is not None:
            Xresampled = check_array(Xresampled)
        if weights is not None:
            weights = check_array(weights, ensure_2d=False)
            check_consistent_length(weights, y)
        X, y = self._validate_data_X_y(X, y)
        self.feature_names_in_ = _safe_check_feature_names_in(
            self, variable_names, generate_names=False
        )

        if self.feature_names_in_ is None:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
            self.display_feature_names_in_ = np.array(
                [f"x{_subscriptify(i)}" for i in range(X.shape[1])]
            )
            variable_names = self.feature_names_in_
        else:
            self.display_feature_names_in_ = self.feature_names_in_
            variable_names = self.feature_names_in_

        # Handle multioutput data
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y = y.reshape(-1)
        elif len(y.shape) == 2:
            self.nout_ = y.shape[1]
        else:
            raise NotImplementedError("y shape not supported!")

        self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)
        self.X_units_ = copy.deepcopy(X_units)
        self.y_units_ = copy.deepcopy(y_units)

        return (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

    def _validate_data_X_y(self, X: Any, y: Any) -> tuple[ndarray, ndarray]:
        if OLD_SKLEARN:
            raw_out = self._validate_data(X=X, y=y, reset=True, multi_output=True)  # type: ignore
        else:
            raw_out = validate_data(self, X=X, y=y, reset=True, multi_output=True)  # type: ignore
        return cast(Tuple[ndarray, ndarray], raw_out)

    def _validate_data_X(self, X: Any) -> ndarray:
        if OLD_SKLEARN:
            raw_out = self._validate_data(X=X, reset=False)  # type: ignore
        else:
            raw_out = validate_data(self, X=X, reset=False)  # type: ignore
        return cast(ndarray, raw_out)

    def _get_precision_mapped_dtype(self, X: np.ndarray) -> type:
        is_complex = np.issubdtype(X.dtype, np.complexfloating)
        is_real = not is_complex
        if is_real:
            return {16: np.float16, 32: np.float32, 64: np.float64}[self.precision]
        else:
            return {32: np.complex64, 64: np.complex128}[self.precision]

    def _pre_transform_training_data(
        self,
        X: ndarray,
        y: ndarray,
        Xresampled: ndarray | None,
        variable_names: ArrayLike[str],
        complexity_of_variables: int | float | list[int | float] | None,
        X_units: ArrayLike[str] | None,
        y_units: ArrayLike[str] | str | None,
        random_state: np.random.RandomState,
    ):
        """
        Transform the training data before fitting the symbolic regressor.

        This method also updates/sets the `selection_mask_` attribute.

        Parameters
        ----------
        X : ndarray
            Training data of shape (n_samples, n_features).
        y : ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).
            Will be cast to X's dtype if necessary.
        Xresampled : ndarray | None
            Resampled training data, of shape `(n_resampled, n_features)`,
            used for denoising.
        variable_names : list[str]
            Names of each variable in the training dataset, `X`.
            Of length `n_features`.
        complexity_of_variables : int | float | list[int | float] | None
            Complexity of each variable in the training dataset, `X`.
        X_units : list[str]
            Units of each variable in the training dataset, `X`.
        y_units : str | list[str]
            Units of each variable in the training dataset, `y`.
        random_state : int | np.RandomState
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`. Default is `None`.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed training data. n_samples will be equal to
            `Xresampled.shape[0]` if `self.denoise` is `True`,
            and `Xresampled is not None`, otherwise it will be
            equal to `X.shape[0]`. n_features will be equal to
            `self.select_k_features` if `self.select_k_features is not None`,
            otherwise it will be equal to `X.shape[1]`
        y_transformed : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Transformed target data. n_samples will be equal to
            `Xresampled.shape[0]` if `self.denoise` is `True`,
            and `Xresampled is not None`, otherwise it will be
            equal to `X.shape[0]`.
        variable_names_transformed : list[str] of length n_features
            Names of each variable in the transformed dataset,
            `X_transformed`.
        X_units_transformed : list[str] of length n_features
            Units of each variable in the transformed dataset.
        y_units_transformed : str | list[str] of length n_out
            Units of each variable in the transformed dataset.
        """
        # Feature selection transformation
        if self.select_k_features:
            selection_mask = run_feature_selection(
                X, y, self.select_k_features, random_state=random_state
            )
            X = X[:, selection_mask]

            if Xresampled is not None:
                Xresampled = Xresampled[:, selection_mask]

            # Reduce variable_names to selection
            variable_names = cast(
                ArrayLike[str],
                [
                    variable_names[i]
                    for i in range(len(variable_names))
                    if selection_mask[i]
                ],
            )

            if isinstance(complexity_of_variables, list):
                complexity_of_variables = [
                    complexity_of_variables[i]
                    for i in range(len(complexity_of_variables))
                    if selection_mask[i]
                ]
                self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)

            if X_units is not None:
                X_units = cast(
                    ArrayLike[str],
                    [X_units[i] for i in range(len(X_units)) if selection_mask[i]],
                )
                self.X_units_ = copy.deepcopy(X_units)

            # Re-perform data validation and feature name updating
            X, y = self._validate_data_X_y(X, y)
            # Update feature names with selected variable names
            self.selection_mask_ = selection_mask
            self.feature_names_in_ = _check_feature_names_in(self, variable_names)
            self.display_feature_names_in_ = self.feature_names_in_
            pysr_logger.info(f"Using features {self.feature_names_in_}")

        # Denoising transformation
        if self.denoise:
            if self.nout_ > 1:
                X, y = multi_denoise(
                    X, y, Xresampled=Xresampled, random_state=random_state
                )
            else:
                X, y = denoise(X, y, Xresampled=Xresampled, random_state=random_state)

        return X, y, variable_names, complexity_of_variables, X_units, y_units

    def _run(
        self,
        X: ndarray,
        y: ndarray,
        runtime_params: _DynamicallySetParams,
        weights: ndarray | None,
        category: ndarray | None,
        seed: int,
    ):
        """
        Run the symbolic regression fitting process on the julia backend.

        Parameters
        ----------
        X : ndarray
            Training data of shape `(n_samples, n_features)`.
        y : ndarray
            Target values of shape `(n_samples,)` or `(n_samples, n_targets)`.
            Will be cast to `X`'s dtype if necessary.
        runtime_params : DynamicallySetParams
            Dynamically set versions of some parameters passed in __init__.
        weights : ndarray | None
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of y.
        category : ndarray | None
            If `expression_spec` is a `ParametricExpressionSpec`, then this
            argument should be a list of integers representing the category
            of each sample in `X`.
        seed : int
            Random seed for julia backend process.

        Returns
        -------
        self : object
            Reference to `self` with fitted attributes.

        Raises
        ------
        ImportError
            Raised when the julia backend fails to import a package.
        """
        # Need to be global as we don't want to recreate/reinstate julia for
        # every new instance of PySRRegressor
        global ALREADY_RAN

        # These are the parameters which may be modified from the ones
        # specified in init, so we define them here locally:
        operators = runtime_params.operators
        constraints = runtime_params.constraints

        nested_constraints = self.nested_constraints
        complexity_of_operators = self.complexity_of_operators
        complexity_of_variables = self.complexity_of_variables_
        cluster_manager = self.cluster_manager

        # Start julia backend processes
        if not ALREADY_RAN and runtime_params.update_verbosity != 0:
            pysr_logger.info("Compiling Julia backend...")

        parallelism, numprocs = _map_parallelism_params(
            self.parallelism, self.procs, getattr(self, "multithreading", None)
        )

        if self.deterministic and parallelism != "serial":
            raise ValueError(
                "To ensure deterministic searches, you must set `parallelism='serial'`. "
                "Additionally, make sure to set `random_state` to a seed."
            )
        if self.random_state is not None and (
            parallelism != "serial" or not self.deterministic
        ):
            warnings.warn(
                "Note: Setting `random_state` without also setting `deterministic=True` "
                "and `parallelism='serial'` will result in non-deterministic searches."
            )

        if cluster_manager is not None:
            if parallelism != "multiprocessing":
                raise ValueError(
                    "To use cluster managers, you must set `parallelism='multiprocessing'`."
                )

        # TODO(mcranmer): These functions should be part of this class.
        operators = _maybe_create_inline_operators(
            operators=operators,
            extra_sympy_mappings=self.extra_sympy_mappings,
            expression_spec=self.expression_spec_,
        )
        if constraints is not None:
            _constraints = _process_constraints(
                operators=operators,
                constraints=constraints,
            )
            # Build constraints for each arity (including empty arities)
            max_arity = max(operators.keys()) if operators else 2
            constraints_by_arity = {}
            for arity in range(1, max_arity + 1):
                if arity in operators and operators[arity]:
                    constraints_by_arity[arity] = [
                        _constraints[op] for op in operators[arity]
                    ]
                else:
                    constraints_by_arity[arity] = []
        else:
            max_arity = max(operators.keys()) if operators else 2
            constraints_by_arity = {arity: None for arity in range(1, max_arity + 1)}

        # Parse dict into Julia Dict for nested constraints::
        if nested_constraints is not None:
            nested_constraints_str = "Dict("
            for outer_k, outer_v in nested_constraints.items():
                nested_constraints_str += f"({outer_k}) => Dict("
                for inner_k, inner_v in outer_v.items():
                    nested_constraints_str += f"({inner_k}) => {inner_v}, "
                nested_constraints_str += "), "
            nested_constraints_str += ")"
            nested_constraints = jl.seval(nested_constraints_str)

        # Parse dict into Julia Dict for complexities:
        if complexity_of_operators is not None:
            complexity_of_operators_str = "Dict("
            for k, v in complexity_of_operators.items():
                complexity_of_operators_str += f"({k}) => {v}, "
            complexity_of_operators_str += ")"
            complexity_of_operators = jl.seval(complexity_of_operators_str)
        # TODO: Refactor this into helper function

        if isinstance(complexity_of_variables, list):
            complexity_of_variables = jl_array(complexity_of_variables)

        np_dtype = self._get_precision_mapped_dtype(np.array(X))

        custom_loss = jl.seval(
            str(self.elementwise_loss)
            if self.elementwise_loss is not None
            else "nothing"
        )
        if self.elementwise_loss is not None:
            _validate_elementwise_loss(
                custom_loss,
                has_weights=weights is not None,
                probe_value=np_dtype(1.0),
            )

        custom_full_objective = jl.seval(
            str(self.loss_function) if self.loss_function is not None else "nothing"
        )
        if self.loss_function is not None:
            _validate_custom_full_objective(custom_full_objective)

        custom_loss_expression = jl.seval(
            str(self.loss_function_expression)
            if self.loss_function_expression is not None
            else "nothing"
        )
        if self.loss_function_expression is not None:
            _validate_custom_expression_objective(custom_loss_expression)

        early_stop_condition = jl.seval(
            str(self.early_stop_condition)
            if self.early_stop_condition is not None
            else "nothing"
        )

        input_stream = jl.seval(self.input_stream)

        load_required_packages(
            turbo=self.turbo,
            bumper=self.bumper,
            autodiff_backend=self.autodiff_backend,
            cluster_manager=cluster_manager,
            logger_spec=self.logger_spec,
        )

        if cluster_manager is not None:
            active_project = jl.seval("Base.active_project()")
            if isinstance(active_project, str) and len(active_project) > 0:
                # `ClusterManagers.addprocs_slurm` launches new Julia workers via `srun`.
                # The project (environment) is propagated via `JULIA_PROJECT` rather
                # than a `--project=...` flag, so ensure it is set.
                os.environ.setdefault(
                    "JULIA_PROJECT", str(Path(active_project).resolve().parent)
                )
            cluster_manager = _load_cluster_manager(cluster_manager)

        if self.autodiff_backend is not None:
            autodiff_backend = jl.Symbol(self.autodiff_backend)
        else:
            autodiff_backend = None

        mutation_weights = SymbolicRegression.MutationWeights(
            mutate_constant=self.weight_mutate_constant,
            mutate_operator=self.weight_mutate_operator,
            mutate_feature=self.weight_mutate_feature,
            swap_operands=self.weight_swap_operands,
            rotate_tree=self.weight_rotate_tree,
            add_node=self.weight_add_node,
            insert_node=self.weight_insert_node,
            delete_node=self.weight_delete_node,
            simplify=self.weight_simplify,
            randomize=self.weight_randomize,
            do_nothing=self.weight_do_nothing,
            optimize=self.weight_optimize,
        )

        # Convert operators dict to Julia format and create OperatorEnum
        # Fill in empty tuples for missing arities up to max arity
        max_arity = max(operators.keys()) if operators else 2
        jl_operators_dict = {}

        for arity in range(1, max_arity + 1):
            if arity in operators:
                jl_op_list = []
                for op in operators[arity]:
                    jl_op = jl.seval(op)
                    if not jl_is_function(jl_op):
                        raise ValueError(
                            f"When building operators for arity {arity}, `'{op}'` did not return a Julia function"
                        )
                    jl_op_list.append(jl_op)
                jl_operators_dict[arity] = tuple(jl_op_list)
            else:
                # Empty tuple for missing arities
                jl_operators_dict[arity] = ()

        complexity_mapping = (
            jl.seval(self.complexity_mapping) if self.complexity_mapping else None
        )

        if hasattr(self, "logger_") and self.logger_ is not None and self.warm_start:
            logger = self.logger_
        else:
            logger = self.logger_spec.create_logger() if self.logger_spec else None

        self.logger_ = logger

        # Use Julia function to create OperatorEnum from Dict{Int,Tuple}
        create_operator_enum = jl.seval(
            "__sr_make_op_enum(ops_dict) = OperatorEnum([k => v for (k, v) in ops_dict]...)"
        )
        jl_operator_enum = create_operator_enum(jl_operators_dict)

        # Build constraints dict with same structure
        jl_constraints_dict = None
        if any(c for c in constraints_by_arity.values() if c is not None):
            constraints_pairs = []
            for arity in range(1, max_arity + 1):
                if constraints_by_arity[arity] is not None:
                    constraints_pairs.append(
                        jl.Pair(arity, jl_array(constraints_by_arity[arity]))
                    )
            if constraints_pairs:
                jl_constraints_dict = jl.Dict(constraints_pairs)

        # Call to Julia backend.
        # See https://github.com/astroautomata/SymbolicRegression.jl/blob/master/src/OptionsStruct.jl
        options = SymbolicRegression.Options(
            operators=jl_operator_enum,
            constraints=jl_constraints_dict,
            complexity_of_operators=complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=complexity_of_variables,
            complexity_mapping=complexity_mapping,
            expression_spec=self.expression_spec_.julia_expression_spec(),
            nested_constraints=nested_constraints,
            elementwise_loss=custom_loss,
            loss_function=custom_full_objective,
            loss_function_expression=custom_loss_expression,
            loss_scale=jl.Symbol(self.loss_scale),
            maxsize=int(self.maxsize),
            output_directory=_escape_filename(self.output_directory_),
            npopulations=int(self.populations),
            # Determine actual batching based on "auto" mode
            batching=(self.batching if self.batching != "auto" else len(X) > 1000),
            batch_size=int(
                _get_batch_size(len(X), runtime_params.batch_size)
                if (
                    self.batching == True or (self.batching == "auto" and len(X) > 1000)
                )
                else len(X)
            ),
            mutation_weights=mutation_weights,
            tournament_selection_p=self.tournament_selection_p,
            tournament_selection_n=self.tournament_selection_n,
            # These have the same name:
            parsimony=self.parsimony,
            dimensional_constraint_penalty=self.dimensional_constraint_penalty,
            dimensionless_constants_only=self.dimensionless_constants_only,
            alpha=self.alpha,
            maxdepth=runtime_params.maxdepth,
            fast_cycle=self.fast_cycle,
            turbo=self.turbo,
            bumper=self.bumper,
            autodiff_backend=autodiff_backend,
            migration=self.migration,
            hof_migration=self.hof_migration,
            fraction_replaced_hof=self.fraction_replaced_hof,
            should_simplify=self.should_simplify,
            should_optimize_constants=self.should_optimize_constants,
            warmup_maxsize_by=runtime_params.warmup_maxsize_by,
            use_frequency=self.use_frequency,
            use_frequency_in_tournament=self.use_frequency_in_tournament,
            adaptive_parsimony_scaling=self.adaptive_parsimony_scaling,
            npop=self.population_size,
            ncycles_per_iteration=self.ncycles_per_iteration,
            fraction_replaced=self.fraction_replaced,
            fraction_replaced_guesses=self.fraction_replaced_guesses,
            topn=self.topn,
            print_precision=self.print_precision,
            optimizer_algorithm=self.optimizer_algorithm,
            optimizer_nrestarts=self.optimizer_nrestarts,
            optimizer_f_calls_limit=self.optimizer_f_calls_limit,
            optimizer_probability=self.optimize_probability,
            optimizer_iterations=self.optimizer_iterations,
            perturbation_factor=self.perturbation_factor,
            probability_negate_constant=self.probability_negate_constant,
            annealing=self.annealing,
            timeout_in_seconds=self.timeout_in_seconds,
            crossover_probability=self.crossover_probability,
            skip_mutation_failures=self.skip_mutation_failures,
            max_evals=self.max_evals,
            input_stream=input_stream,
            early_stop_condition=early_stop_condition,
            seed=seed,
            deterministic=self.deterministic,
            define_helper_functions=False,
        )

        self.julia_options_stream_ = jl_serialize(options)

        # Convert data to desired precision

        # This converts the data into a Julia array:
        jl_X = jl_array(np.array(X, dtype=np_dtype).T)
        if len(y.shape) == 1:
            jl_y = jl_array(np.array(y, dtype=np_dtype))
        else:
            jl_y = jl_array(np.array(y, dtype=np_dtype).T)
        if weights is not None:
            if len(weights.shape) == 1:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype))
            else:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype).T)
        else:
            jl_weights = None

        if category is not None:
            offset_for_julia_indexing = 1
            jl_category = jl_array(
                (category + offset_for_julia_indexing).astype(np.int64)
            )
            jl_extra = jl.seval("NamedTuple{(:class,)}")((jl_category,))
        else:
            jl_extra = jl.NamedTuple()

        if len(y.shape) > 1:
            # We set these manually so that they respect Python's 0 indexing
            # (by default Julia will use y1, y2...)
            jl_y_variable_names = jl_array(
                [f"y{_subscriptify(i)}" for i in range(y.shape[1])]
            )
        else:
            jl_y_variable_names = None

        jl_guesses = _prepare_guesses_for_julia(self.guesses, self.nout_)

        # Convert worker_imports to Julia symbols
        jl_worker_imports = (
            jl_array([jl.Symbol(s) for s in self.worker_imports])
            if self.worker_imports is not None
            else None
        )

        out = SymbolicRegression.equation_search(
            jl_X,
            jl_y,
            weights=jl_weights,
            extra=jl_extra,
            niterations=int(self.niterations),
            variable_names=jl_array([str(v) for v in self.feature_names_in_]),
            display_variable_names=jl_array(
                [str(v) for v in self.display_feature_names_in_]
            ),
            y_variable_names=jl_y_variable_names,
            X_units=jl_array(self.X_units_),
            y_units=(
                jl_array(self.y_units_)
                if isinstance(self.y_units_, list)
                else self.y_units_
            ),
            options=options,
            guesses=jl_guesses,
            numprocs=numprocs,
            parallelism=parallelism,
            saved_state=self.julia_state_,
            return_state=True,
            run_id=self.run_id_,
            addprocs_function=cluster_manager,
            heap_size_hint_in_bytes=self.heap_size_hint_in_bytes,
            worker_timeout=self.worker_timeout,
            worker_imports=jl_worker_imports,
            progress=runtime_params.progress
            and self.verbosity > 0
            and len(y.shape) == 1,
            verbosity=int(self.verbosity),
            logger=logger,
        )
        if self.logger_spec is not None:
            self.logger_spec.write_hparams(logger, self.get_params())
            if not self.warm_start:
                self.logger_spec.close(logger)

        self.julia_state_stream_ = jl_serialize(out)

        # Set attributes
        self.equations_ = self.get_hof(out)

        ALREADY_RAN = True

        return self

    def fit(
        self,
        X,
        y,
        *,
        Xresampled=None,
        weights=None,
        variable_names: ArrayLike[str] | None = None,
        complexity_of_variables: int | float | list[int | float] | None = None,
        X_units: ArrayLike[str] | None = None,
        y_units: str | ArrayLike[str] | None = None,
        category: ndarray | None = None,
    ) -> "PySRRegressor":
        """
        Search for equations to fit the dataset and store them in `self.equations_`.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape (n_samples, n_features).
        y : ndarray | pandas.DataFrame
            Target values of shape (n_samples,) or (n_samples, n_targets).
            Will be cast to X's dtype if necessary.
        Xresampled : ndarray | pandas.DataFrame
            Resampled training data, of shape (n_resampled, n_features),
            to generate a denoised data on. This
            will be used as the training data, rather than `X`.
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of `y`. Alternatively,
            if a custom `loss` was set, it will can be used
            in arbitrary ways.
        variable_names : list[str]
            A list of names for the variables, rather than "x0", "x1", etc.
            If `X` is a pandas dataframe, the column names will be used
            instead of `variable_names`. Cannot contain spaces or special
            characters. Avoid variable names which are also
            function names in `sympy`, such as "N".
        X_units : list[str]
            A list of units for each variable in `X`. Each unit should be
            a string representing a Julia expression. See DynamicQuantities.jl
            https://symbolicml.org/DynamicQuantities.jl/dev/units/ for more
            information.
        y_units : str | list[str]
            Similar to `X_units`, but as a unit for the target variable, `y`.
            If `y` is a matrix, a list of units should be passed. If `X_units`
            is given but `y_units` is not, then `y_units` will be arbitrary.
        category : list[int]
            If `expression_spec` is a `ParametricExpressionSpec`, then this
            argument should be a list of integers representing the category
            of each sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Init attributes that are not specified in BaseEstimator
        if self.warm_start and hasattr(self, "julia_state_stream_"):
            pass
        else:
            if hasattr(self, "julia_state_stream_"):
                warnings.warn(
                    "The discovered expressions are being reset. "
                    "Please set `warm_start=True` if you wish to continue "
                    "to start a search where you left off.",
                )

            self.equations_ = None
            self.nout_ = 1
            self.selection_mask_ = None
            self.julia_state_stream_ = None
            self.julia_options_stream_ = None
            self.complexity_of_variables_ = None
            self.X_units_ = None
            self.y_units_ = None

        self._setup_equation_file()
        self._clear_equation_file_contents()

        runtime_params = self._validate_and_modify_params()

        if category is not None:
            assert Xresampled is None

        if isinstance(self.expression_spec, ParametricExpressionSpec):
            assert category is not None

        # TODO: Put `category` here
        (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        ) = self._validate_and_set_fit_params(
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

        if X.shape[0] > 50000:
            warnings.warn(
                "You are using a dataset with more than 50,000 datapoints. "
                "Symbolic regression rarely benefits from this many points - consider "
                "subsampling to 10,000 points or fewer. If you have high noise, "
                "denoise the data first rather than using more points."
            )

        random_state = check_random_state(self.random_state)  # For np random
        seed = cast(int, random_state.randint(0, 2**31 - 1))  # For julia random

        if isinstance(self.expression_spec, ParametricExpressionSpec):
            parametric_expression_deprecation_warning(
                self.expression_spec.max_parameters, variable_names
            )

        # Pre transformations (feature selection and denoising)
        X, y, variable_names, complexity_of_variables, X_units, y_units = (
            self._pre_transform_training_data(
                X,
                y,
                Xresampled,
                variable_names,
                complexity_of_variables,
                X_units,
                y_units,
                random_state,
            )
        )

        # Assertion checks
        use_custom_variable_names = variable_names is not None
        # TODO: this is always true.

        _check_assertions(
            X,
            use_custom_variable_names,
            variable_names,
            complexity_of_variables,
            weights,
            y,
            X_units,
            y_units,
        )

        # Initially, just save model parameters, so that
        # it can be loaded from an early exit:
        if not self.temp_equation_file:
            self._checkpoint()

        # Perform the search:
        self._run(X, y, runtime_params, weights=weights, seed=seed, category=category)

        # Then, after fit, we save again, so the pickle file contains
        # the equations:
        if not self.temp_equation_file:
            self._checkpoint()

        return self

    def refresh(self, run_directory: PathLike | None = None) -> None:
        """
        Update self.equations_ with any new options passed.

        For example, updating `extra_sympy_mappings`
        will require a `.refresh()` to update the equations.

        Parameters
        ----------
        checkpoint_file : str or Path
            Path to checkpoint hall of fame file to be loaded.
            The default will use the set `equation_file_`.
        """
        if run_directory is not None:
            self.output_directory_ = str(Path(run_directory).parent)
            self.run_id_ = Path(run_directory).name
            self._clear_equation_file_contents()
        check_is_fitted(self, attributes=["run_id_", "output_directory_"])
        self.equations_ = self.get_hof()

    def predict(
        self,
        X,
        index: int | list[int] | None = None,
        *,
        category: ndarray | None = None,
    ) -> ndarray:
        """
        Predict y from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should
        have the same columns as the training data.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape `(n_samples, n_features)`.
        index : int | list[int]
            If you want to compute the output of an expression using a
            particular row of `self.equations_`, you may specify the index here.
            For multiple output equations, you must pass a list of indices
            in the same order.
        category : ndarray | None
            If `expression_spec` is a `ParametricExpressionSpec`, then this
            argument should be a list of integers representing the category
            of each sample in `X`.

        Returns
        -------
        y_predicted : ndarray of shape (n_samples, nout_)
            Values predicted by substituting `X` into the fitted symbolic
            regression model.

        Raises
        ------
        ValueError
            Raises if the `best_equation` cannot be evaluated.
        """
        check_is_fitted(
            self, attributes=["selection_mask_", "feature_names_in_", "nout_"]
        )
        best_equation = self.get_best(index=index)

        # When X is an numpy array or a pandas dataframe with a RangeIndex,
        # the self.feature_names_in_ generated during fit, for the same X,
        # will cause a warning to be thrown during _validate_data.
        # To avoid this, convert X to a dataframe, apply the selection mask,
        # and then set the column/feature_names of X to be equal to those
        # generated during fit.
        if not isinstance(X, pd.DataFrame):
            X = check_array(X)
            X = pd.DataFrame(X)
        if isinstance(X.columns, pd.RangeIndex):
            if self.selection_mask_ is not None:
                # RangeIndex enforces column order allowing columns to
                # be correctly filtered with self.selection_mask_
                X = X[X.columns[self.selection_mask_]]
            X.columns = self.feature_names_in_

        # During fit, we replace spaces in DataFrame column names with
        # underscores. Apply the same normalization here.
        cols_str = X.columns.astype(str)
        if cols_str.str.contains(" ").any():
            X = X.copy()
            X.columns = cols_str.str.replace(" ", "_")
            warnings.warn(
                "Spaces in DataFrame column names are not supported. "
                "Spaces have been replaced with underscores. \n"
                "Please rename the columns to valid names."
            )

        # Without feature information, CallableEquation/lambda_format equations
        # require that the column order of X matches that of the X used during
        # the fitting process. _validate_data removes this feature information
        # when it converts the dataframe to an np array. Thus, to ensure feature
        # order is preserved after conversion, the dataframe columns must be
        # reordered/reindexed to match those of the transformed (denoised and
        # feature selected) X in fit.
        X = X.reindex(columns=self.feature_names_in_)
        X = self._validate_data_X(X)
        if self.expression_spec_.evaluates_in_julia:
            # Julia wants the right dtype
            X = X.astype(self._get_precision_mapped_dtype(X))

        if category is not None:
            offset_for_julia_indexing = 1
            args: tuple = (
                jl_array((category + offset_for_julia_indexing).astype(np.int64)),
            )
        else:
            args = ()

        try:
            if isinstance(best_equation, list):
                assert self.nout_ > 1
                return np.stack(
                    [
                        cast(ndarray, eq["lambda_format"](X, *args))
                        for eq in best_equation
                    ],
                    axis=1,
                )
            else:
                return cast(ndarray, best_equation["lambda_format"](X, *args))
        except Exception as error:
            raise ValueError(
                "Failed to evaluate the expression. "
                "If you are using a custom operator, make sure to define it in `extra_sympy_mappings`, "
                "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                "You can then run `model.refresh()` to re-load the expressions."
            ) from error

    def sympy(self, index: int | list[int] | None = None):
        """
        Return sympy representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : str, list[str] of length nout_
            SymPy representation of the best equation.
        """
        if not self.expression_spec_.supports_sympy:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support sympy export."
            )
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["sympy_format"] for eq in best_equation]
        else:
            return best_equation["sympy_format"]

    def latex(
        self, index: int | list[int] | None = None, precision: int = 3
    ) -> str | list[str]:
        """
        Return latex representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.
        precision : int
            The number of significant figures shown in the LaTeX
            representation.
            Default is `3`.

        Returns
        -------
        best_equation : str or list[str] of length nout_
            LaTeX expression of the best equation.
        """
        if not self.expression_spec_.supports_latex:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support latex export."
            )
        self.refresh()
        sympy_representation = self.sympy(index=index)
        if self.nout_ > 1:
            output = []
            for s in sympy_representation:
                latex = sympy2latex(s, prec=precision)
                output.append(latex)
            return output
        return sympy2latex(sympy_representation, prec=precision)

    def jax(self, index=None):
        """
        Return jax representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a dictionary
        containing {"callable": func, "parameters": params}. To call `func`, pass
        func(X, params). This function is differentiable using `jax.grad`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : dict[str, Any]
            Dictionary of callable jax function in "callable" key,
            and jax array of parameters as "parameters" key.
        """
        if not self.expression_spec_.supports_jax:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support jax export."
            )
        self.set_params(output_jax_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["jax_format"] for eq in best_equation]
        else:
            return best_equation["jax_format"]

    def pytorch(self, index=None):
        """
        Return pytorch representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a PyTorch module
        containing the parameters as trainable attributes. You can use the module like
        any other PyTorch module: `module(X)`, where `X` is a tensor with the same
        column ordering as trained with.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : torch.nn.Module
            PyTorch module representing the expression.
        """
        if not self.expression_spec_.supports_torch:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support torch export."
            )
        self.set_params(output_torch_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            return [eq["torch_format"] for eq in best_equation]
        else:
            return best_equation["torch_format"]

    def get_equation_file(self, i: int | None = None) -> Path:
        if i is not None:
            return (
                Path(self.output_directory_)
                / self.run_id_
                / f"hall_of_fame_output{i}.csv"
            )
        else:
            return Path(self.output_directory_) / self.run_id_ / "hall_of_fame.csv"

    def _read_equation_file(self) -> list[pd.DataFrame]:
        """Read the hall of fame file created by `SymbolicRegression.jl`."""

        try:
            if self.nout_ > 1:
                all_outputs = []
                for i in range(1, self.nout_ + 1):
                    cur_filename = str(self.get_equation_file(i)) + ".bak"
                    if not os.path.exists(cur_filename):
                        cur_filename = str(self.get_equation_file(i))
                    with open(cur_filename, "r", encoding="utf-8") as f:
                        buf = f.read()
                    buf = _preprocess_julia_floats(buf)
                    df = self._postprocess_dataframe(pd.read_csv(StringIO(buf)))
                    all_outputs.append(df)
            else:
                filename = str(self.get_equation_file()) + ".bak"
                if not os.path.exists(filename):
                    filename = str(self.get_equation_file())
                with open(filename, "r", encoding="utf-8") as f:
                    buf = f.read()
                buf = _preprocess_julia_floats(buf)
                all_outputs = [self._postprocess_dataframe(pd.read_csv(StringIO(buf)))]

        except FileNotFoundError:
            raise RuntimeError(
                "Couldn't find equation file! The equation search likely exited "
                "before a single iteration completed."
            )
        return all_outputs

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "Complexity": "complexity",
                "Loss": "loss",
                "Equation": "equation",
            },
        )

        return df

    def get_hof(self, search_output=None) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the equations from a hall of fame file or search output.

        If no arguments entered, the ones used
        previously from a call to PySR will be used.
        """
        check_is_fitted(
            self,
            attributes=[
                "nout_",
                "run_id_",
                "output_directory_",
                "selection_mask_",
                "feature_names_in_",
            ],
        )
        should_read_from_file = (
            not hasattr(self, "equation_file_contents_")
            or self.equation_file_contents_ is None
        )
        if should_read_from_file:
            self.equation_file_contents_ = self._read_equation_file()

        _validate_export_mappings(self.extra_jax_mappings, self.extra_torch_mappings)

        equation_file_contents = cast(List[pd.DataFrame], self.equation_file_contents_)

        ret_outputs: list[pd.DataFrame] = [
            cast(
                pd.DataFrame,
                pd.concat(
                    [
                        output,
                        *(
                            [calculate_scores(output)]
                            if self.loss_scale == "log"
                            else []
                        ),
                        self.expression_spec_.create_exports(
                            self, output, search_output, i if self.nout_ > 1 else None
                        ),
                    ],
                    axis=1,
                ),
            )
            for i, output in enumerate(equation_file_contents)
        ]

        if self.nout_ > 1:
            return ret_outputs
        return ret_outputs[0]

    def latex_table(
        self,
        indices: list[int] | None = None,
        precision: int = 3,
        columns: list[str] = ["equation", "complexity", "loss", "score"],
    ) -> str:
        """Create a LaTeX/booktabs table for all, or some, of the equations.

        Parameters
        ----------
        indices : list[int] | list[list[int]]
            If you wish to select a particular subset of equations from
            `self.equations_`, give the row numbers here. By default,
            all equations will be used. If there are multiple output
            features, then pass a list of lists.
        precision : int
            The number of significant figures shown in the LaTeX
            representations.
            Default is `3`.
        columns : list[str]
            Which columns to include in the table.
            Default is `["equation", "complexity", "loss", "score"]`.

        Returns
        -------
        latex_table_str : str
            A string that will render a table in LaTeX of the equations.
        """
        if not self.expression_spec_.supports_latex:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support latex export."
            )
        self.refresh()

        if isinstance(self.equations_, list):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], list)
                assert len(indices) == self.nout_

            table_string = sympy2multilatextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        elif isinstance(self.equations_, pd.DataFrame):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], int)

            table_string = sympy2latextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        else:
            raise ValueError(
                "Invalid type for equations_ to pass to `latex_table`. "
                "Expected a DataFrame or a list of DataFrames."
            )

        return with_preamble(table_string)


def idx_model_selection(equations: pd.DataFrame, model_selection: str):
    """Select an expression and return its index."""

    # We must default to "accuracy" if no score column is present (like in the case of linear loss_scale)
    model_selection = model_selection if "score" in equations.columns else "accuracy"

    if model_selection == "accuracy":
        chosen_idx = equations["loss"].idxmin()
    elif model_selection == "best":
        threshold = 1.5 * equations["loss"].min()
        filtered_equations = equations.query(f"loss <= {threshold}")
        chosen_idx = filtered_equations["score"].idxmax()
    elif model_selection == "score":
        chosen_idx = equations["score"].idxmax()
    else:
        raise NotImplementedError(
            f"{model_selection} is not a valid model selection strategy."
        )
    return chosen_idx


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scores for each equation based on loss and complexity.

    Score is defined as the negated derivative of the log-loss with respect to complexity.
    A higher score means the equation achieved a much better loss at a slightly higher complexity.
    """
    scores = []
    lastMSE = None
    lastComplexity = 0

    for _, row in df.iterrows():
        curMSE = row["loss"]
        curComplexity = row["complexity"]

        if lastMSE is None:
            cur_score = 0.0
        else:
            if curMSE > 0.0:
                cur_score = -np.log(curMSE / lastMSE) / (curComplexity - lastComplexity)
            else:
                cur_score = np.inf

        scores.append(cur_score)
        lastMSE = curMSE
        lastComplexity = curComplexity

    return pd.DataFrame(
        {
            "score": np.array(scores),
        },
        index=df.index,
    )


def _prepare_guesses_for_julia(guesses, nout) -> VectorValue | None:
    """Convert Python guesses to Julia format.

    Parameters
    ----------
    guesses : list[str] | list[list[str]] | list[dict[str, str]] | list[list[dict[str, str]]] | None
        Initial guesses for equations
    nout : int
        Number of output dimensions

    Returns
    -------
    jl_guesses: VectorValue | None
        Julia-compatible guesses array or None if no guesses provided
    """
    if guesses is None:
        return None

    g = guesses

    if nout == 1:
        if not isinstance(g, list):
            raise ValueError("guesses must be a list for single-output regression")
        elif len(g) == 0:
            g = [[]]
        elif not isinstance(g[0], list):
            g = [g]
        elif len(g) != 1:
            raise ValueError(
                "For single output, provide a list of strings/dicts or "
                "a single-element list of lists"
            )
    else:
        if not (isinstance(g, list) and all(isinstance(x, list) for x in g)):
            raise ValueError(
                "For multi-output (nout > 1) guesses must be a list of lists"
            )
        if len(g) != nout:
            raise ValueError(
                f"Number of guess lists ({len(g)}) must match number of outputs ({nout})"
            )

    julia_guesses = []
    for output_guesses in g:
        julia_output_guesses = []
        for item in output_guesses:
            if isinstance(item, dict):
                # Convert dict to NamedTuple for template expressions
                julia_output_guesses.append(jl_named_tuple(item))
            else:
                # Keep strings as-is
                julia_output_guesses.append(item)
        julia_guesses.append(jl_array(julia_output_guesses))

    return jl_array(julia_guesses)


def _get_batch_size(dataset_size: int, batch_size_param: int | None) -> int:
    """Calculate the actual batch size to use."""
    if batch_size_param is not None:
        return min(dataset_size, batch_size_param)
    elif dataset_size <= 1000:
        return dataset_size
    elif dataset_size < 5000:
        return 128
    elif dataset_size < 50000:
        return 256
    else:
        return 512


def _mutate_parameter(param_name: str, param_value):
    if param_name == "batch_size" and param_value is not None and param_value < 1:
        warnings.warn(
            "Given `batch_size` must be greater than or equal to one. "
            "`batch_size` has been increased to equal one."
        )
        return 1

    if (
        param_name == "progress"
        and param_value == True
        and "buffer" not in sys.stdout.__dir__()
    ):
        warnings.warn(
            "Note: it looks like you are running in Jupyter. "
            "The progress bar will be turned off."
        )
        return False

    return param_value


def _map_parallelism_params(
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None,
    procs: int | None,
    multithreading: bool | None,
) -> tuple[Literal["serial", "multithreading", "multiprocessing"], int | None]:
    """Map old and new parallelism parameters to the new format.

    Parameters
    ----------
    parallelism : str or None
        New parallelism parameter. Can be "serial", "multithreading", or "multiprocessing".
    procs : int or None
        Number of processes parameter.
    multithreading : bool or None
        Old multithreading parameter.

    Returns
    -------
    parallelism : str
        Mapped parallelism mode.
    procs : int or None
        Mapped number of processes.

    Raises
    ------
    ValueError
        If both old and new parameters are specified, or if invalid combinations are given.
    """
    # Check for mixing old and new parameters
    using_new = parallelism is not None
    using_old = multithreading is not None

    if using_new and using_old:
        raise ValueError(
            "Cannot mix old and new parallelism parameters. "
            "Use either `parallelism` and `numprocs`, or `procs` and `multithreading`."
        )
    elif using_old:
        warnings.warn(
            "The `multithreading: bool` parameter has been deprecated in favor "
            "of `parallelism: Literal['multithreading', 'serial', 'multiprocessing']`.\n"
            "Previous usage of `multithreading=True` (default) is now `parallelism='multithreading'`; "
            "`multithreading=False, procs=0` is now `parallelism='serial'`; and "
            "`multithreading=True, procs={int}` is now `parallelism='multiprocessing', procs={int}`."
        )
        if multithreading:
            _parallelism: Literal["multithreading", "multiprocessing", "serial"] = (
                "multithreading"
            )
            _procs = None
        elif procs is not None and procs > 0:
            _parallelism = "multiprocessing"
            _procs = procs
        else:
            _parallelism = "serial"
            _procs = None
    elif using_new:
        _parallelism = cast(
            Literal["serial", "multithreading", "multiprocessing"], parallelism
        )
        _procs = procs
    else:
        _parallelism = "multithreading"
        _procs = None

    if _parallelism not in {"serial", "multithreading", "multiprocessing"}:
        raise ValueError(
            "`parallelism` must be one of 'serial', 'multithreading', or 'multiprocessing'"
        )
    elif _parallelism == "serial" and _procs is not None:
        warnings.warn(
            "`numprocs` is specified but will be ignored since `parallelism='serial'`"
        )
        _procs = None
    elif parallelism == "multithreading" and _procs is not None:
        warnings.warn(
            "`numprocs` is specified but will be ignored since `parallelism='multithreading'`"
        )
        _procs = None
    elif parallelism == "multiprocessing" and _procs is None:
        _procs = cpu_count()

    return _parallelism, _procs
