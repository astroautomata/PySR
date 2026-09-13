"""Rediscover Conway's Game of Life (B3/S23) from its transition table.

The search sees every combination of a cell state and a neighbour count, and nothing
else. Its vocabulary is and / or / not plus an integer equality test; the neighbour
thresholds are integer constants it has to find.
"""

import numpy as np

from pysr import PySRRegressor, TypeSpec

TABLE = [(a, n, int(n == 3 or (a == 1 and n == 2))) for a in (0, 1) for n in range(9)]
REPLICAS = 8
X = np.array([[a, n] for a, n, _ in TABLE] * REPLICAS, dtype=object)
y = np.array([t for _, _, t in TABLE] * REPLICAS, dtype=object)
VARIABLE_NAMES = ["alive", "n"]

SPEC = TypeSpec(
    "Cell",
    fields={"v": "Int"},
    sample="rng -> Cell(rand(rng, 0:8))",
    mutate="(rng, value, temperature) -> Cell(mod(value.v + rand(rng, (-1, 1)), 9))",
    string="value -> string(value.v)",
)

MODEL_KWARGS = dict(
    type_spec=SPEC,
    operators={
        1: ["not(a::Cell) = Cell(a.v == 0 ? 1 : 0)"],
        2: [
            "and(a::Cell, b::Cell) = Cell((a.v != 0 && b.v != 0) ? 1 : 0)",
            "or(a::Cell, b::Cell) = Cell((a.v != 0 || b.v != 0) ? 1 : 0)",
            "eq(a::Cell, b::Cell) = Cell(a.v == b.v ? 1 : 0)",
        ],
    },
    elementwise_loss="cell_loss(prediction::Cell, target::Cell)::Float64 = prediction.v == target.v ? 0.0 : 1.0",
    niterations=160,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


def check(model):
    """Every row of the transition table reproduced exactly."""
    prediction = np.array([int(v) for v in model.predict(X)])
    return bool((prediction == np.array([int(v) for v in y])).all())


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    prediction = np.array([int(v) for v in model.predict(X)])
    target = np.array([int(v) for v in y])
    print(f"\nexact rows: {int((prediction == target).sum())}/{len(target)}")


if __name__ == "__main__":
    main()
