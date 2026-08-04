"""Various functions to deprecate features."""

import warnings


def install(*args, **kwargs):
    del args, kwargs
    warnings.warn(
        "The `install` function has been removed. "
        "PySR now uses the `juliacall` package to install its dependencies "
        "automatically when the Julia backend is used.",
        FutureWarning,
    )


def init_julia(*args, **kwargs):
    del args, kwargs
    from .julia_import import jl

    warnings.warn(
        "The `init_julia` function has been removed. "
        "Julia is now initialized automatically when the Julia backend is used.",
        FutureWarning,
    )
    return jl


def pysr(X, y, weights=None, **kwargs):  # pragma: no cover
    from .sr import PySRRegressor

    warnings.warn(
        "Calling `pysr` is deprecated. "
        "Please use `model = PySRRegressor(**params); "
        "model.fit(X, y)` going forward.",
        FutureWarning,
    )
    model = PySRRegressor(**kwargs)
    model.fit(X, y, weights=weights)
    return model.equations_


def best(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "`best` has been deprecated. "
        "Please use the `PySRRegressor` interface. "
        "After fitting, you can return `.sympy()` "
        "to get the sympy representation "
        "of the best equation."
    )


def best_row(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "`best_row` has been deprecated. "
        "Please use the `PySRRegressor` interface. "
        "After fitting, you can run `print(model)` to view the best equation, "
        "or "
        "`model.get_best()` to return the best equation's "
        "row in `model.equations_`."
    )


def best_tex(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "`best_tex` has been deprecated. "
        "Please use the `PySRRegressor` interface. "
        "After fitting, you can return `.latex()` to "
        "get the sympy representation "
        "of the best equation."
    )


def best_callable(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "`best_callable` has been deprecated. Please use the `PySRRegressor` "
        "interface. After fitting, you can use "
        "`.predict(X)` to use the best callable."
    )


DEPRECATED_KWARGS = {
    "fractionReplaced": "fraction_replaced",
    "fractionReplacedHof": "fraction_replaced_hof",
    "npop": "population_size",
    "hofMigration": "hof_migration",
    "shouldOptimizeConstants": "should_optimize_constants",
    "weightAddNode": "weight_add_node",
    "weightDeleteNode": "weight_delete_node",
    "weightDoNothing": "weight_do_nothing",
    "weightInsertNode": "weight_insert_node",
    "weightMutateConstant": "weight_mutate_constant",
    "weightMutateOperator": "weight_mutate_operator",
    "weightSwapOperands": "weight_swap_operands",
    "weightRandomize": "weight_randomize",
    "weightSimplify": "weight_simplify",
    "crossoverProbability": "crossover_probability",
    "perturbationFactor": "perturbation_factor",
    "batchSize": "batch_size",
    "warmupMaxsizeBy": "warmup_maxsize_by",
    "useFrequency": "use_frequency",
    "useFrequencyInTournament": "use_frequency_in_tournament",
    "ncyclesperiteration": "ncycles_per_iteration",
    "loss": "elementwise_loss",
    "full_objective": "loss_function",
}
