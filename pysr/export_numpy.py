"""Code for exporting discovered expressions to numpy"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sympy import Expr, Symbol, lambdify  # type: ignore


def sympy2numpy(eqn, sympy_symbols, *, selection=None):
    return CallableEquation(eqn, sympy_symbols, selection=selection)


class CallableEquation:
    """Simple wrapper for numpy lambda functions built with sympy"""

    _sympy: Expr
    _sympy_symbols: list[Symbol]
    _selection: NDArray[np.bool_] | None

    def __init__(self, eqn, sympy_symbols, selection=None):
        self._sympy = eqn
        self._sympy_symbols = sympy_symbols
        self._selection = selection

    def __repr__(self):
        return f"PySRFunction(X=>{self._sympy})"

    def __call__(self, X):
        expected_shape = (X.shape[0],)
        if isinstance(X, pd.DataFrame):
            # Lambda function takes as argument:
            return self._lambda(
                **{k: X[k].values for k in map(str, self._sympy_symbols)}
            ) * np.ones(expected_shape)

        if self._selection is not None:
            if X.shape[1] != self._selection.sum():
                warnings.warn(
                    "`X` should be of shape (n_samples, len(self._selection)). "
                    "Automatically filtering `X` to selection. "
                    "Note: Filtered `X` column order may not match column order in fit "
                    "this may lead to incorrect predictions and other errors."
                )
                X = X[:, self._selection]

        return self._lambda(*X.T) * np.ones(expected_shape)

    @property
    def _lambda(self):
        return lambdify(self._sympy_symbols, self._sympy)
