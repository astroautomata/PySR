from math import comb
from typing import List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from .sr import PySRRegressor
from .utils import ArrayLike, _subscriptify


def _check_assertions(
    X,
    recursive_history_length=None,
    weights=None,
    variable_names=None,
    X_units=None,
):
    if recursive_history_length is not None and recursive_history_length <= 0:
        raise ValueError(
            "The `recursive_history_length` parameter must be greater than 0 (otherwise it's not recursion)."
        )
    if len(X.shape) > 2:
        raise ValueError(
            "Recursive symbolic regression only supports up to 2D data; please flatten your data first"
        )
    if len(X) <= recursive_history_length + 1:
        raise ValueError(
            f"Recursive symbolic regression with a history length of {recursive_history_length} requires at least {recursive_history_length + 2} datapoints."
        )
    if isinstance(weights, np.ndarray) and len(weights) != len(X):
        raise ValueError("The length of `weights` must have shape (n_times,).")
    if isinstance(variable_names, list) and len(variable_names) != X.shape[1]:
        raise ValueError(
            "The length of `variable_names` must be equal to the number of features in `X`."
        )
    if isinstance(X_units, list) and len(X_units) != X.shape[1]:
        raise ValueError(
            "The length of `X_units` must be equal to the number of features in `X`."
        )


class PySRSequenceRegressor(BaseEstimator):
    """
    High performance symbolic regression for recurrent sequences.
    Based off of the `PySRRegressor` class, but with a preprocessing step for recurrence relations.

    Parameters
    ----------
    recursive_history_length : int
        The number of previous time points to use as input features.
        For example, if `recursive_history_length=2`, then the input features
        will be `[2, X[0], X[1]]` and the output will be `X[2]`, where the
        first feature is the output's sequence index when `homogenous=False`.
        This continues as `[n, X[n-2], X[n-1]]` to predict `X[n]`.
        Must be greater than 0.
    homogenous : bool
        Whether the recurrence is independent of the sequence index. If `True`,
        the time feature is omitted. Default is `False`.
    difference_order : int
        Fit a finite difference instead of the sequence value itself. `1` fits
        `X[t] - X[t-1]`; `2` fits `X[t] - 2*X[t-1] + X[t-2]`. Predictions and
        symbolic exports are converted back to sequence values. Default is `0`.
    Other parameters and attributes are inherited from `PySRRegressor`.
    """

    def __init__(
        self,
        *,
        recursive_history_length: int = 0,
        homogenous: bool = False,
        difference_order: int = 0,
        linear_guesses: bool = False,
        **kwargs,
    ):
        super().__init__()
        if difference_order < 0:
            raise ValueError("`difference_order` must be at least 0.")
        if not homogenous and isinstance(kwargs.get("complexity_of_variables"), list):
            kwargs["complexity_of_variables"] = [
                1,
                *kwargs["complexity_of_variables"],
            ]
        self._has_user_guesses = kwargs.get("guesses") is not None
        self._regressor = PySRRegressor(**kwargs)
        self.recursive_history_length = recursive_history_length
        self.homogenous = homogenous
        self.difference_order = difference_order
        self.linear_guesses = linear_guesses

    @staticmethod
    def _as_sequences(X):
        if isinstance(X, np.ndarray) and X.ndim == 3:
            raw_sequences = list(X)
        elif (
            isinstance(X, (list, tuple))
            and X
            and all(isinstance(sequence, np.ndarray) for sequence in X)
        ):
            raw_sequences = list(X)
        else:
            raw_sequences = [X]

        sequences = []
        for sequence in raw_sequences:
            sequence = check_array(sequence, ensure_2d=False)
            if sequence.ndim == 1:
                sequence = sequence.reshape(-1, 1)
            assert sequence.ndim == 2
            sequences.append(sequence)
        return sequences

    @staticmethod
    def _per_sequence(value, count: int, name: str):
        if value is None:
            return [None] * count
        if count == 1:
            if isinstance(value, np.ndarray) and value.ndim > 1 and len(value) == 1:
                return [value[0]]
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and isinstance(value[0], np.ndarray)
            ):
                return list(value)
            return [value]
        if isinstance(value, np.ndarray) and len(value) == count:
            return list(value)
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError(
                f"`{name}` must contain one array for each of the {count} sequences."
            )
        return list(value)

    def _construct_variable_names(
        self, n_features: int, variable_names: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        if not isinstance(variable_names, list):
            if n_features == 1:
                variable_names = ["x"]
                display_variable_names = ["x"]
            else:
                variable_names = [f"x{i}" for i in range(n_features)]
                display_variable_names = [
                    f"x{_subscriptify(i)}" for i in range(n_features)
                ]
        else:
            display_variable_names = variable_names

        # e.g., `x0_tm1`
        variable_names_with_time = [
            f"{var}_tm{j}"
            for j in range(self.recursive_history_length, 0, -1)
            for var in variable_names
        ]
        # e.g., `x₀[t-1]`
        display_variable_names_with_time = [
            f"{var}[t-{j}]"
            for j in range(self.recursive_history_length, 0, -1)
            for var in display_variable_names
        ]

        if self.homogenous:
            return variable_names_with_time, display_variable_names_with_time
        return ["t", *variable_names_with_time], [
            "t",
            *display_variable_names_with_time,
        ]

    def fit(
        self,
        X,
        *,
        weights=None,
        time_values=None,
        variable_names: Optional[List[str]] = None,
        complexity_of_variables: Optional[
            Union[int, float, List[Union[int, float]]]
        ] = None,
        X_units: Optional[ArrayLike[str]] = None,
    ) -> "PySRSequenceRegressor":
        """
        Search for equations to fit the sequence and store them in `self.equations_`.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame | list[ndarray]
            One sequence of shape `(n_times, n_features)` or `(n_times,)`, or
            multiple independent sequences as a list of arrays or a 3D array.
            History windows never cross sequence boundaries.
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `X`.
            Each element is how to weight the mean-square-error loss
            for that particular element of `X`. Alternatively,
            if a custom `loss` was set, it can be used
            in custom ways.
        time_values : ndarray | list[ndarray]
            Physical coordinate for each sequence row when `homogenous=False`.
            Values for future predictions are extrapolated using the final step.
        variable_names : list[str]
            A list of names for the variables, rather than "x0t_1", "x1t_2", etc.
            The sequence index is automatically named "t" when
            `homogenous=False`.
            If `X` is a pandas dataframe, the column name will be used
            instead of `variable_names`. Cannot contain spaces or special
            characters. Avoid variable names which are also
            function names in `sympy`, such as "N".
            The number of variable names must be equal to (n_features,).
        complexity_of_variables : int | float | list[int] | list[float]
            The complexity of each variable in `X`. If a single value is
            passed, it will be used for all variables. If a list is passed,
            its length must be the same as `recurrence_history_length`.
        X_units : list[str]
            A list of units for each variable in `X`. Each unit should be
            a string representing a Julia expression. See DynamicQuantities.jl
            https://symbolicml.org/DynamicQuantities.jl/dev/units/ for more
            information.
            Length should be equal to n_features.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.difference_order > self.recursive_history_length:
            raise ValueError(
                "`difference_order` cannot exceed `recursive_history_length`."
            )
        sequences = self._as_sequences(X)
        feature_count = sequences[0].shape[1]
        if any(sequence.shape[1] != feature_count for sequence in sequences):
            raise ValueError("All sequences must have the same number of features.")
        sequence_weights = self._per_sequence(weights, len(sequences), "weights")
        sequence_times = self._per_sequence(time_values, len(sequences), "time_values")
        if (
            not self.homogenous
            and any(value is None for value in sequence_times)
            and any(value is not None for value in sequence_times)
        ):
            raise ValueError(
                "Provide `time_values` for every sequence or for none of them."
            )
        for sequence, sequence_weight in zip(sequences, sequence_weights):
            _check_assertions(
                sequence,
                self.recursive_history_length,
                sequence_weight,
                variable_names,
                X_units,
            )
        self.variable_names = variable_names  # for latex_table()
        self.n_features = feature_count  # for latex_table()
        self._uses_custom_time = not self.homogenous and time_values is not None

        current_X = np.concatenate(
            [self._difference_targets(sequence) for sequence in sequences], axis=0
        )
        historical_X = np.concatenate(
            [
                self._sliding_window(sequence, values)[:-1]
                for sequence, values in zip(sequences, sequence_times)
            ],
            axis=0,
        )
        y_units = X_units
        if weights is not None:
            weights = np.concatenate(
                [
                    np.asarray(value)[self.recursive_history_length :]
                    for value in sequence_weights
                ],
                axis=0,
            )
        if not self.homogenous and isinstance(complexity_of_variables, list):
            complexity_of_variables = [1, *complexity_of_variables]
        variable_names, display_variable_names = self._construct_variable_names(
            current_X.shape[1], variable_names
        )
        self._feature_variable_names = [
            name.removesuffix("_tm1") for name in variable_names[-feature_count:]
        ]

        self._regressor.fit(
            X=historical_X,
            y=current_X,
            weights=weights,
            variable_names=variable_names,
            display_variable_names=display_variable_names,
            X_units=X_units,
            y_units=y_units,
            complexity_of_variables=complexity_of_variables,
        )
        return self

    def predict(self, X, index=None, num_predictions=1, time_values=None):
        """
        Predict future data from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should
        have the same columns as the training data.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Data of shape `(n_times, n_features)`.
        index : int | list[int]
            If you want to compute the output of an expression using a
            particular row of `self.equations_`, you may specify the index here.
            For multiple output equations, you must pass a list of indices
            in the same order.
        num_predictions : int
            How many predictions to make. If `num_predictions` is less than
            `(n_times - recursive_history_length + 1)`,
            some input data at the end will be ignored.
            Default is `1`.
        time_values : ndarray
            Physical coordinate for each input row when one was supplied to
            `fit`. Future coordinates are extrapolated using the final step.

        Returns
        -------
        x_predicted : ndarray of shape (num_predictions, n_features)
            Values predicted by substituting `X` into the fitted sequence symbolic
            regression model and rolling it out for `num_predictions` steps.

        Raises
        ------
        ValueError
            Raises if the `best_equation` cannot be evaluated.
        """
        X = self._as_sequences(X)
        if len(X) != 1:
            raise ValueError("`predict` accepts one sequence at a time.")
        X = X[0]
        _check_assertions(X, recursive_history_length=self.recursive_history_length)
        if getattr(self, "_uses_custom_time", False) and time_values is None:
            raise ValueError(
                "`time_values` is required because it was supplied to `fit`."
            )
        historical_X = self._sliding_window(X, time_values)
        if num_predictions < 1:
            raise ValueError("num_predictions must be greater than 0.")
        if num_predictions < len(historical_X):
            historical_X = historical_X[:num_predictions]
            transformed = self._regressor.predict(X=historical_X, index=index)
            return self._restore_predictions(transformed, historical_X)
        else:
            extra_predictions = num_predictions - len(historical_X)
            transformed = self._regressor.predict(X=historical_X, index=index)
            pred = self._restore_predictions(transformed, historical_X)
            rolling = [row.copy() for row in X]
            rolling.append(pred[-1].copy())
            time_step = self._time_step(time_values)
            for i in range(extra_predictions):
                history = np.asarray(rolling[-self.recursive_history_length :])
                pred_data = (
                    [history.flatten()]
                    if self.homogenous
                    else [
                        [
                            historical_X[-1, 0] + (i + 1) * time_step,
                            *history.flatten(),
                        ]
                    ]
                )
                transformed = self._regressor.predict(X=pred_data, index=index)
                direct = self._restore_difference(
                    np.asarray(transformed).reshape(-1), history
                ).reshape(1, -1)
                pred = np.concatenate([pred, direct], axis=0)
                rolling.append(direct[0])
            return pred

    def _difference_targets(self, X):
        if self.difference_order == 0:
            return X[self.recursive_history_length :]
        return np.diff(X, n=self.difference_order, axis=0)[
            self.recursive_history_length - self.difference_order :
        ]

    def _restore_difference(self, transformed, history):
        direct = np.asarray(transformed, dtype=float).copy()
        if self.difference_order == 0:
            return direct
        for j, previous in enumerate(history[-self.difference_order :]):
            direct -= (
                (-1) ** (self.difference_order - j)
                * comb(self.difference_order, j)
                * previous
            )
        return direct

    def _restore_predictions(self, transformed, historical_X):
        transformed = np.asarray(transformed)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        offset = 0 if self.homogenous else 1
        feature_count = (
            historical_X.shape[1] - offset
        ) // self.recursive_history_length
        histories = historical_X[:, offset:].reshape(
            -1, self.recursive_history_length, feature_count
        )
        return np.asarray(
            [
                self._restore_difference(value, history)
                for value, history in zip(transformed, histories)
            ]
        )

    @staticmethod
    def _time_step(time_values):
        if time_values is None:
            return 1
        values = np.asarray(time_values, dtype=float)
        return values[-1] - values[-2]

    def _sliding_window(self, X, time_values=None):
        historical_X = np.lib.stride_tricks.sliding_window_view(
            X.flatten(), self.recursive_history_length * np.prod(X.shape[1])
        )[:: X.shape[1]]
        if self.homogenous:
            return historical_X
        if time_values is None:
            target_times = np.arange(
                self.recursive_history_length,
                self.recursive_history_length + len(historical_X),
            )
        else:
            values = np.asarray(time_values, dtype=float)
            if values.ndim != 1 or len(values) not in (len(X), len(X) + 1):
                raise ValueError(
                    "`time_values` must be one-dimensional with one value per "
                    "sequence row, optionally including the next prediction time."
                )
            if not np.isfinite(values).all():
                raise ValueError("`time_values` must contain only finite values.")
            if len(values) == len(X):
                values = np.append(values, values[-1] + self._time_step(values))
            target_times = values[
                self.recursive_history_length : self.recursive_history_length
                + len(historical_X)
            ]
        return np.column_stack((target_times, historical_X))

    @classmethod
    def from_file(
        cls,
        *args,
        recursive_history_length: int,
        homogenous: bool = False,
        difference_order: int = 0,
        linear_guesses: bool = False,
        **kwargs,
    ):
        assert recursive_history_length is not None and recursive_history_length > 0

        model = cls(
            recursive_history_length=recursive_history_length,
            homogenous=homogenous,
            difference_order=difference_order,
            linear_guesses=linear_guesses,
        )
        model._regressor = PySRRegressor.from_file(*args, **kwargs)
        return model

    def __repr__(self):
        return self._regressor.__repr__().replace(
            "PySRRegressor", "PySRSequenceRegressor", 1
        )

    def get_best(self, *args, **kwargs):
        return self._regressor.get_best(*args, **kwargs)

    def refresh(self, *args, **kwargs):
        return self._regressor.refresh(*args, **kwargs)

    def sympy(self, *args, **kwargs):
        expressions = self._regressor.sympy(*args, **kwargs)
        if self.difference_order == 0:
            return expressions
        is_multiple = isinstance(expressions, list)
        expression_list = expressions if is_multiple else [expressions]
        restored = []
        for feature, expression in zip(self._feature_variable_names, expression_list):
            direct = expression
            for j in range(self.difference_order):
                lag = self.difference_order - j
                direct -= (
                    (-1) ** (self.difference_order - j)
                    * comb(self.difference_order, j)
                    * sp.Symbol(f"{feature}_tm{lag}")
                )
            restored.append(sp.expand(direct))
        return restored if is_multiple else restored[0]

    def latex(self, index=None, precision=3):
        if self.difference_order == 0:
            return self._regressor.latex(index=index, precision=precision)
        from .export_latex import sympy2latex

        expressions = self.sympy(index=index)
        if isinstance(expressions, list):
            return [
                sympy2latex(expression, prec=precision) for expression in expressions
            ]
        return sympy2latex(expressions, prec=precision)

    def get_hof(self):
        return self._regressor.get_hof()

    def latex_table(
        self,
        *args,
        **kwargs,
    ):
        """
        Generates LaTeX variable names, then creates a LaTeX table of the best equation(s).
        Refer to `PySRRegressor.latex_table` for information.
        """
        if self.variable_names is not None:
            if len(self.variable_names) == 1:
                variable_names = self.variable_names[0] + "_{tm}"
            else:
                variable_names = [
                    variable_name + "_{tm}" for variable_name in self.variable_names
                ]
        else:
            if self.n_features == 1:
                variable_names = "x_{tm}"
            else:
                variable_names = [f"x_{{{i} tm}}" for i in range(self.n_features)]
        return self._regressor.latex_table(
            *args, **kwargs, output_variable_names=variable_names
        )

    @property
    def equations_(self):
        return self._regressor.equations_
