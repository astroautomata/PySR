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

    # Standardize the inputs so the initial length scales suit any units of
    # `X`; `normalize_y` below does the same for the targets.
    X_mean = X.mean(axis=0)
    X_scale = X.std(axis=0)
    X_scale[X_scale == 0] = 1.0

    # The `ConstantKernel` factor learns the signal variance. Without it the
    # smooth component has unit variance, so targets far from unit scale are
    # mostly attributed to noise.
    gp_kernel = ConstantKernel() * RBF(np.ones(X.shape[1])) + WhiteKernel(1e-1)
    gpr = GaussianProcessRegressor(
        kernel=gp_kernel,
        normalize_y=True,
        n_restarts_optimizer=50,
        random_state=random_state,
    )
    gpr.fit((X - X_mean) / X_scale, y)

    X_out = X if Xresampled is None else Xresampled
    return X_out, cast(ndarray, gpr.predict((X_out - X_mean) / X_scale))


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
