"""Recover y = 2.5382*cos(x3) + x0^2 - 0.5 from 20,000 rows.

PySR defaults to `batching="auto"`, which evolves populations against a mini-batch
while still ranking the hall of fame on the full dataset. With `batch_size=None` the
batch is chosen from the row count: the full dataset at N <= 1000, 128 below 5000,
256 below 50,000, and 512 above. This problem has 20,000 rows, so evolution sees 256
rows per comparison and the hall of fame is still scored on all 20,000.

Setting `batching=False` runs every population comparison against all 20,000 rows.
That is the same search on the same data, and it is far slower.
"""

import numpy as np

from pysr import PySRRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (20000, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

VARIABLE_NAMES = ["x0", "x1", "x2", "x3", "x4"]

MODEL_KWARGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    batching="auto",
    niterations=40,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


def check(model):
    """The target is recovered when a front member reaches numerically zero loss."""
    return bool(model.equations_["loss"].min() < 1e-10)


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    print(f"\nbest loss: {model.equations_['loss'].min():.3e}")
    print(f"recovered: {check(model)}")


if __name__ == "__main__":
    main()
