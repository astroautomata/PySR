from typing import List, Optional, Tuple, Union

import numpy as np
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
        will be `[X[0], X[1]]` and the output will be `X[2]`.
        This continues on for all X: [X[n-1], X[n-2]] to predict X[n].
        Must be greater than 0.
    Other parameters and attributes are inherited from `PySRRegressor`.
    """

    def __init__(
        self,
        *,
        recursive_history_length: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._regressor = PySRRegressor(**kwargs)
        self.recursive_history_length = recursive_history_length

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

        return variable_names_with_time, display_variable_names_with_time

    def fit(
        self,
        X,
        *,
        weights=None,
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
        X : ndarray | pandas.DataFrame
            Sequence of shape (n_times, n_features) or (n_times,)
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `X`.
            Each element is how to weight the mean-square-error loss
            for that particular element of `X`. Alternatively,
            if a custom `loss` was set, it can be used
            in custom ways.
        variable_names : list[str]
            A list of names for the variables, rather than "x0t_1", "x1t_2", etc.
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
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert X.ndim == 2
        _check_assertions(
            X,
            self.recursive_history_length,
            weights,
            variable_names,
            X_units,
        )
        self.variable_names = variable_names  # for latex_table()
        self.n_features = X.shape[1]  # for latex_table()

        current_X = X[self.recursive_history_length :]
        historical_X = self._sliding_window(X)[: -1 : current_X.shape[1], :]
        y_units = X_units
        if isinstance(weights, np.ndarray):
            weights = weights[self.recursive_history_length :]
        variable_names, display_variable_names = self._construct_variable_names(
            current_X.shape[1], variable_names
        )

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

    def predict(self, X, index=None, num_predictions=1):
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
        X = check_array(X, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert X.ndim == 2
        _check_assertions(X, recursive_history_length=self.recursive_history_length)
        historical_X = self._sliding_window(X)[:: X.shape[1], :]
        if num_predictions < 1:
            raise ValueError("num_predictions must be greater than 0.")
        if num_predictions < len(historical_X):
            historical_X = historical_X[:num_predictions]
            return self._regressor.predict(X=historical_X, index=index)
        else:
            extra_predictions = num_predictions - len(historical_X)
            pred = self._regressor.predict(X=historical_X, index=index)
            for _ in range(extra_predictions):
                pred_data = [pred[-self.recursive_history_length :].flatten()]
                pred = np.concatenate(
                    [pred, self._regressor.predict(X=pred_data, index=index)], axis=0
                )
            return pred

    def _sliding_window(self, X):
        return np.lib.stride_tricks.sliding_window_view(
            X.flatten(), self.recursive_history_length * np.prod(X.shape[1])
        )

    @classmethod
    def from_file(
        cls,
        *args,
        recursive_history_length: int,
        **kwargs,
    ):
        assert recursive_history_length is not None and recursive_history_length > 0

        model = cls(recursive_history_length=recursive_history_length)
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
        return self._regressor.sympy(*args, **kwargs)

    def latex(self, *args, **kwargs):
        return self._regressor.latex(*args, **kwargs)

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
