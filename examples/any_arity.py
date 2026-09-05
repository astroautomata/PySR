"""Rediscover a clipped envelope written with a four-argument minimum.

Operators are keyed by arity, so an operator of any arity can be supplied: here a
three-argument conditional and a four-argument minimum sit side by side in the
same set. The target `min4(x + 2, 2 - x, sin(3x) + 1.2, 0.8)` folds a rising
ramp, a falling ramp, a shifted sinusoid and a flat cap together in one call,
which binary `min` would need three nested calls to write.

Recovery is judged on 4001 points the search never saw, not on the 400 it fitted.
"""

import numpy as np
import sympy

from pysr import PySRRegressor

DOMAIN = (-4.0, 4.0)


def envelope(x):
    return np.minimum.reduce(
        [x + 2.0, 2.0 - x, np.sin(3.0 * x) + 1.2, np.full_like(x, 0.8)]
    )


_x = np.sort(np.random.default_rng(0).uniform(*DOMAIN, 400))
X = _x.reshape(-1, 1).astype(np.float32)
y = envelope(_x).astype(np.float32)
VARIABLE_NAMES = ["x"]

_unseen = np.linspace(*DOMAIN, 4001)
X_UNSEEN = _unseen.reshape(-1, 1).astype(np.float32)
Y_UNSEEN = envelope(_unseen)

MODEL_KWARGS = dict(
    operators={
        1: ["sin", "cos"],
        2: ["+", "-", "*"],
        3: ["ifelse(t, a, b) = t > 0 ? a : b"],
        4: ["min4(a, b, c, d) = min(a, b, c, d)"],
    },
    # A custom operator is rejected at fit time unless it has a SymPy image.
    extra_sympy_mappings={
        "ifelse": lambda t, a, b: sympy.Piecewise((a, t > 0), (b, True)),
        "min4": lambda a, b, c, d: sympy.Min(a, b, c, d),
    },
    niterations=500,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

TOLERANCE = 1e-4


def unseen_errors(model):
    """Largest error of each front member on the points the search never saw."""
    return {
        i: float(
            np.abs(
                np.asarray(model.predict(X_UNSEEN, index=i), np.float64) - Y_UNSEEN
            ).max()
        )
        for i in model.equations_.index
    }


def check(model):
    """A front member reproduces the law off the fitted samples, not just on them."""
    return bool(any(e < TOLERANCE for e in unseen_errors(model).values()))


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)

    front = model.equations_[["complexity", "loss", "equation"]]
    errors = unseen_errors(model)
    print(
        front.assign(unseen_error=[errors[i] for i in front.index]).to_string(
            index=False
        )
    )

    uses = front["equation"].str.contains
    print(
        f"\nfront members using the arity-3 ifelse: {int(uses('ifelse').sum())}, using the arity-4 min4: {int(uses('min4').sum())}"
    )

    print("true law: min4(x + 2, 2 - x, sin(3x) + 1.2, 0.8)")
    recovered = [i for i, e in errors.items() if e < TOLERANCE]
    if recovered:
        best = min(recovered, key=lambda i: front.complexity[i])
        print(
            f"recovered at complexity {int(front.complexity[best])}, max error {errors[best]:.3g} over 4001 unseen points:"
        )
        print(f"  {front.equation[best]}")
    else:
        print("no front member agrees with the law off the fitted samples")


if __name__ == "__main__":
    main()
