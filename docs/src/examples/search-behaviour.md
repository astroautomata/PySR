# Search behaviour

## Automatic batching on a large dataset

`batching` reduces the rows used to score population members during evolution. Hall-of-fame comparisons continue to use every row. With `batching=False`, every evolutionary comparison uses all rows. When `batch_size` is `None`, automatic batching can use every row on small datasets. PySR chooses all rows through 1000, 128 below 5000, 256 below 50,000, and 512 from 50,000 onward. At 20,000 rows, evolution therefore sees 256 rows per comparison, while hall-of-fame scoring sees all 20,000. `batching="auto"` is the default; this example names it to make the choice explicit.

The target depends on two of five input columns. The remaining three inputs act as distractors.

<details>
<summary>Data generation code</summary>

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (20000, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

</details>

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    batching="auto",
    niterations=40,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    random_state=0,
)
model.fit(X, y, variable_names=["x0", "x1", "x2", "x3", "x4"])
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```

One recovered expression is:

```
(cos(x3) * 2.5382) + ((x0 * x0) + -0.5)
```

The fixed random state and deterministic serial settings make the example reproducible. For parallel search, omit `deterministic=True` and `parallelism="serial"`. Run the complete example with `examples/automatic_batching.py`.

## Operators of any arity

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/S5.mp4"></video>

The generic `operators` argument defines an expression grammar by arity. Each integer key gives the number of arguments accepted by the operators listed under it, so one search can mix unary and binary operations with ternary or wider functions. `operators` cannot be combined with `binary_operators` or `unary_operators`.

The target is a four-way minimum,

$$
y = \min(x + 2,\; 2 - x,\; \sin(3x) + 1.2,\; 0.8).
$$

A binary `min` would write this reduction as three nested calls. The custom `min4` writes the same mathematical operation in one call, giving the search a shorter structural route to the target.

<details>
<summary>Data generation code</summary>

```python
import numpy as np

DOMAIN = (-4.0, 4.0)


def envelope(x):
    return np.minimum.reduce(
        [x + 2.0, 2.0 - x, np.sin(3.0 * x) + 1.2, np.full_like(x, 0.8)]
    )


_x = np.sort(np.random.default_rng(0).uniform(*DOMAIN, 400))
X = _x.reshape(-1, 1).astype(np.float32)
y = envelope(_x).astype(np.float32)

_unseen = np.linspace(*DOMAIN, 4001)
X_UNSEEN = _unseen.reshape(-1, 1).astype(np.float32)
Y_UNSEEN = envelope(_unseen)
```

</details>

Custom operator definitions are Julia source strings under their arity keys. A SymPy-capable model also needs a matching entry in `extra_sympy_mappings`; PySR rejects a named custom function during setup when its symbolic image is missing. The mappings here provide the `Piecewise` representation for `ifelse` and the `Min` representation for `min4`.

```python
import sympy

from pysr import PySRRegressor

model = PySRRegressor(
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
    random_state=0,
)
model.fit(X, y, variable_names=["x"])
print(model.equations_)
```

One recovered expression is:

```
min4(0.8, 2.0 - x, 2.0 + x, 1.1999999 + sin(x * 3.0))
```

The complete example, `examples/any_arity.py`, checks predictions on an unseen grid with `model.predict(X_UNSEEN, index=i)` for each candidate equation.

## Mutations and plugins

PySR accepts mutation and plugin configuration as `PySRRegressor` arguments. `mutations` maps mutation entries to weights, and `plugins` lists plugin entries. Entries merge with shipped defaults by concrete type: a matching type replaces the default entry, while a new type is appended alongside it. This lets one behavior change without reproducing the complete default set.

The following configuration activates Backsolve and names the adaptive mutation-weight plugin.

```python
from pysr import (
    AdaptiveMutationWeightsPlugin,
    BacksolveMutation,
    PySRRegressor,
)

model = PySRRegressor(
    mutations={BacksolveMutation(): 0.1},
    plugins=[AdaptiveMutationWeightsPlugin()],
)
```

`BacksolveMutation` is present at default weight 0.0, so it is available without being sampled. Assigning weight 0.1 makes it eligible for this search. Backsolve inverts the operators above a selected subtree and fits a replacement from expressions already in the population, helping an additive target assemble from separately discovered pieces.

`AdaptiveMutationWeightsPlugin` is part of the default plugin set. It updates mutation weights from observed improvement rates and resets its runtime state for each `fit`. Naming it in `plugins` replaces the existing entry of that type, which is also where `smoothing`, `floor`, and `reward` can be changed. The resolved list keeps simulated annealing and adaptive parsimony alongside adaptive mutation weights. The two settings work together: the mutation weight makes Backsolve eligible, and the adaptive plugin can change how much of the mutation budget each kind receives.

The runnable configuration uses the small target

$$ y = 2.5\cos(3x) + 0.5x^2 - 1. $$

```python
import numpy as np

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
```

```python
from pysr import AdaptiveMutationWeightsPlugin, BacksolveMutation, PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "exp"],
    niterations=20,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    mutations={BacksolveMutation(): 0.1},
    plugins=[AdaptiveMutationWeightsPlugin()],
    random_state=0,
)
model.fit(X, y, variable_names=["x"])
print(model.equations_)
```

After fitting, inspect `model.julia_options_.mutations` to see the resolved entries and weights. Run the example with `examples/mutations_and_plugins.py`.

## Adaptive mutation weights

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D9.mp4"></video>

`AdaptiveMutationWeightsPlugin` adjusts mutation weights from their observed success rates. It is enabled by default, and its learned state resets on each `fit`. Pass an instance through `plugins` to change its settings.

- `smoothing` controls the exponential moving average of success rates.
- `floor` limits the success-rate ratio used to update a multiplier to $[\text{floor}, 1/\text{floor}]$.
- `reward` selects whether improvement is measured by `"cost"` or `"loss"`.

This example fits a cosine and a quadratic term:

<details>
<summary>Data generation code</summary>

```python
import numpy as np

rng = np.random.default_rng(20260817)
X = rng.uniform(-3.0, 3.0, size=(200, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

</details>

```python
from pysr import AdaptiveMutationWeightsPlugin, PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    maxsize=14,
    niterations=300,
    populations=1,
    plugins=[AdaptiveMutationWeightsPlugin(smoothing=0.02, floor=0.05, reward="cost")],
    deterministic=True,
    parallelism="serial",
    random_state=0,
    verbosity=0,
)
model.fit(X, y)
print(model.equations_)
```

For an optional diagnostic that records mutation counts and learned multipliers, see `examples/adaptive_mutation_weights.py`.

## The backsolve mutation

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D10.mp4"></video>

Ordinary mutations change an expression and test the result. Backsolve chooses a non-root subtree and inverts the operators above it to determine the values that subtree would need to produce. It then fits a sparse replacement from expressions already in the population. Greedy forward selection over a basis library finds the replacement within the remaining complexity budget. The replaced subtree can remain in the library, and rows with invalid inverse values are masked. Both `+` and `*` are required for the weighted sum. An event returns a child only when its cost strictly improves on the parent. Each event requires an inversion and a linear solve, so enabling Backsolve can substantially increase search time.

Backsolve applies to ordinary binary scalar expression trees with a real target vector. It rejects unsupported wrappers, shared nodes, root or single-node choices, missing targets, and missing weighted-sum operators. The mutation is experimental and starts at default weight 0.0.

The target combines an oscillation with a quadratic:

$$ y = 2.5\cos(3x) + 0.5x^2 - 1. $$

```python
import numpy as np

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
```

The example omits division and a power operator, so the quadratic must appear as `x * x`. `precision=64` supports the linear solve, and `maxsize=30` leaves room for the complete form.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "exp"],
    weight_backsolve=1.0,
    maxsize=30,
    precision=64,
    niterations=10,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    random_state=0,
)
model.fit(X, y, variable_names=["x"])
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```

One recovered expression is:

```
((cos((x + x) + x) * 2.5) + ((x * x) * 0.5)) + -1.0
```

Check the selected equation against the training target:

```python
import numpy as np

prediction = np.asarray(model.predict(X), dtype=float)
print(f"max abs error: {np.max(np.abs(prediction - y)):.3g}")
```

Use `examples/backsolve_mutation.py` for the complete example.
