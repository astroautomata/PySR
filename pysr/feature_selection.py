"""Functions for doing feature selection during preprocessing."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .utils import ArrayLike

pysr_logger = logging.getLogger(__name__)

# Rows bootstrapped per tree; ranking features needs far fewer rows than
# large datasets provide.
_MAX_SAMPLES_PER_TREE = 10_000


def run_feature_selection(
    X: ndarray,
    y: ndarray,
    select_k_features: int,
    random_state: np.random.RandomState | None = None,
) -> NDArray[np.bool_]:
    """
    Find most important features.

    Uses a random forest regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    clf = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        max_samples=(_MAX_SAMPLES_PER_TREE if len(X) > _MAX_SAMPLES_PER_TREE else None),
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return cast(NDArray[np.bool_], selector.get_support(indices=False))


# Function has not been removed only due to usage in module tests
def _handle_feature_selection(
    X: ndarray,
    select_k_features: int | None,
    y: ndarray,
    variable_names: ArrayLike[str],
):
    if select_k_features is not None:
        selection = run_feature_selection(X, y, select_k_features)
        selected_indices = np.where(selection)[0]
        pysr_logger.info(
            f"Using features {[variable_names[i] for i in selected_indices]}"
        )
        X = X[:, selection]
    else:
        selection = None

    return X, selection
