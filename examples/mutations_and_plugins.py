"""Weight a mutation and add a plugin from PySRRegressor.

Mutations and plugins are ordinary configuration: `mutations` maps a mutation
configuration to a weight, `plugins` lists plugin configurations, and entries of
either override the shipped defaults by type and leave the rest alone.
`BacksolveMutation` ships at weight 0.0, so giving it a weight is what turns it
on for this search for 2.5*cos(3x) + 0.5*x^2 - 1.
"""

import numpy as np

from pysr import AdaptiveMutationWeightsPlugin, BacksolveMutation, PySRRegressor
from pysr.julia_import import jl

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
VARIABLE_NAMES = ["x"]

BASE_KWARGS = dict(
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "exp"],
    niterations=20,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

MODEL_KWARGS = dict(
    **BASE_KWARGS,
    mutations={BacksolveMutation(): 0.1},
    plugins=[AdaptiveMutationWeightsPlugin()],
)


def check(model):
    """The noiseless target reproduced to within 1e-5 everywhere."""
    return bool(np.max(np.abs(model.predict(X) - y)) < 1e-5)


def _backsolve_weight(model):
    for mutation, weight in model.julia_options_.mutations:
        if "BacksolveMutation" in jl.string(mutation):
            return float(weight)
    raise AssertionError("BacksolveMutation missing from the resolved mutation table")


def _run(kwargs, label):
    model = PySRRegressor(**kwargs, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(f"=== {label}")
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    print(f"max error: {np.max(np.abs(model.predict(X) - y)):.3g}")
    print(f"backsolve weight: {_backsolve_weight(model)}")
    print(f"plugins: {jl.string(model.julia_options_.plugins)}\n")
    return model


def main():
    _run(BASE_KWARGS, "default mutations and plugins")
    model = _run(MODEL_KWARGS, "backsolve weighted, adaptive weights on")
    print(f"exact recovery: {check(model)}")


if __name__ == "__main__":
    main()
