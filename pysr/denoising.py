"""Functions for denoising data during preprocessing."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy import ndarray


def denoise(
    X: ndarray,
    y: ndarray,
    Xresampled: ndarray | None = None,
    random_state: np.random.RandomState | None = None,
) -> tuple[ndarray, ndarray]:
    """Denoise the dataset using a Gaussian process."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    gp_kernel = RBF(np.ones(X.shape[1])) + WhiteKernel(1e-1) + ConstantKernel()
    gpr = GaussianProcessRegressor(
        kernel=gp_kernel, n_restarts_optimizer=50, random_state=random_state
    )
    gpr.fit(X, y)

    if Xresampled is not None:
        return Xresampled, cast(ndarray, gpr.predict(Xresampled))

    return X, cast(ndarray, gpr.predict(X))


def multi_denoise(
    X: ndarray,
    y: ndarray,
    Xresampled: ndarray | None = None,
    random_state: np.random.RandomState | None = None,
):
    """Perform `denoise` along each column of `y` independently."""
    y = np.stack(
        [
            denoise(X, y[:, i], Xresampled=Xresampled, random_state=random_state)[1]
            for i in range(y.shape[1])
        ],
        axis=1,
    )

    if Xresampled is not None:
        return Xresampled, y

    return X, y
