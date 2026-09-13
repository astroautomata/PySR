# Objectives and losses

## Custom objectives

`loss_function` gives the objective the candidate's expression tree. Use it when the score needs to manipulate the tree or read dataset-level information. Use `loss_function_expression` when the score needs the complete expression object. Both modes accept Julia source with one of the signatures below.

```julia
objective(tree_or_expression, dataset, options)
objective(tree_or_expression, dataset, options, idx=nothing)
```

With three arguments, the objective sees the dataset given to the current evaluation. Automatic batching makes that dataset the active batch. A method with `idx=nothing` can receive the full dataset and the selected row indices, allowing the objective to slice the rows itself.

The following objective reproduces ordinary unweighted mean squared error, while retaining access to the optional batch indices:

```python
from pysr import PySRRegressor

objective = """
function mse_objective(tree, dataset::Dataset{T,L}, options, idx=nothing)::L where {T,L}
    X = idx === nothing ? dataset.X : dataset.X[:, idx]
    y = idx === nothing ? dataset.y : dataset.y[idx]
    prediction, complete = eval_tree_array(tree, X, options)
    complete || return L(Inf)
    return sum(abs2, prediction .- y) / length(y)
end
"""

model = PySRRegressor(
    loss_function=objective,
    binary_operators=["+", "-", "*", "/"],
)
```

`eval_tree_array` returns a completion flag with the predictions. Check that flag before using the array and assign an infinite, or suitably large, loss to an incomplete evaluation. Return the dataset's loss type `L`; it must remain real even when the data type `T` is complex.

PySR imports `SymbolicRegression` into Julia before evaluating these source strings, making `Dataset` and `eval_tree_array` available without another import.

A full objective can also attach a different meaning to the stored tree. Here the two children of a binary root supply the numerator and denominator, forming the rational function $P(X)/Q(X)$. The fully qualified `DE.get_child` calls below access the backend's tree interface.

```python
from pysr import PySRRegressor

objective = """
function rational_objective(tree, dataset::Dataset{T,L}, options, idx=nothing)::L where {T,L}
    tree.degree == 2 || return L(Inf)
    X = idx === nothing ? dataset.X : dataset.X[:, idx]
    y = idx === nothing ? dataset.y : dataset.y[idx]

    numerator = SymbolicRegression.InterfaceDynamicExpressionsModule.DE.get_child(tree, 1)
    denominator = SymbolicRegression.InterfaceDynamicExpressionsModule.DE.get_child(tree, 2)
    p, p_complete = eval_tree_array(numerator, X, options)
    q, q_complete = eval_tree_array(denominator, X, options)
    p_complete && q_complete || return L(Inf)

    prediction = p ./ q
    return sum(abs2, prediction .- y) / length(y)
end
"""

model = PySRRegressor(
    loss_function=objective,
    binary_operators=["+", "-"],
)
```

This objective ignores the root operator and interprets only its two children. The equation table displays the stored tree. `predict` and symbolic export likewise retain the raw tree and cannot automatically reconstruct the quotient. Keep the objective source and apply the child transformation yourself whenever you evaluate the selected equation.

## Writing the objective in Python

The objective itself can live in Python while Julia supplies a thin shim. The search releases Python's global interpreter lock (GIL) while it runs, so the shim reacquires the lock, calls the Python function, and converts its scalar result to the dataset's loss type. The Python function receives the tree, dataset, and options as Julia objects, and it can call back into Julia, including `eval_tree_array`:

```python
import numpy as np

from pysr import PySRRegressor, jl


def python_objective(tree, dataset, options):
    prediction, completed = jl.SymbolicRegression.eval_tree_array(
        tree, dataset.X, options
    )
    if not completed:
        return float("inf")

    prediction = np.asarray(prediction)
    target = np.asarray(dataset.y)
    return float(np.mean((prediction - target) ** 2))


jl.python_objective = python_objective

jl.seval("""
using PythonCall

function python_objective_shim(
    tree, dataset::Dataset{T,L}, options
)::L where {T,L}
    PythonCall.GIL.@lock begin
        return pyconvert(L, python_objective(tree, dataset, options))
    end
end
""")

model = PySRRegressor(
    loss_function="python_objective_shim",
    precision=64,
)
```

`precision=64` makes the data and loss use `Float64`, matching the Python float returned by this objective.

Use this bridge when the objective needs Python-specific logic. Every evaluation crosses the Julia/Python boundary, and concurrent callbacks serialize on the GIL. Keep throughput-sensitive objectives in Julia.

## Swinging up a cart-pole with a rollout objective

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D11.mp4"></video>

A supervised example gives a target for each row. A control policy has no label of that kind for its force: a force is good only if the subsequent trajectory is good. `loss_function_expression` provides the candidate expression to a rollout objective that can evaluate it throughout the closed loop. This example runs 500 steps from each of 16 training starts and returns the negative mean reward per step. Most of the starts leave the pole hanging down, so the expression must swing it up, catch it, and balance it with one closed-form policy.

### The plant and what the policy sees

The implemented plant uses a $1.0\,\mathrm{kg}$ cart and a $0.1\,\mathrm{kg}$ pole with half-length $0.5\,\mathrm{m}$, gravity $9.8$, and semi-implicit Euler integration with $dt=0.02\,\mathrm{s}$. Its 500-step horizon is therefore $10\,\mathrm{s}$. The policy output is clipped to `[-1, 1]` and multiplied by $10\,\mathrm{N}$. The five inputs are scaled cart position, scaled cart speed, $\sin\theta$, $\cos\theta$, and scaled pole rate. The sine/cosine pair avoids the wrap discontinuity at $\pm\pi$, while the cosine input `c` supplies a direct uprightness signal.

<details>
<summary>Constants and the plant step</summary>

```python
import numpy as np

M_C, M_P, LENGTH, GRAVITY = 1.0, 0.1, 0.5, 9.8
FORCE_CAP, DT = 10.0, 0.02
RAIL, V_SCALE, OMEGA_SCALE = 2.4, 4.0, 8.0
HORIZON = 500  # 10 s of control at dt = 0.02


def step(state, force):
    x, v, theta, omega = state.T
    total = M_C + M_P
    sin, cos = np.sin(theta), np.cos(theta)
    q = (force + M_P * LENGTH * omega**2 * sin) / total
    theta_dd = (GRAVITY * sin - cos * q) / (LENGTH * (4 / 3 - M_P * cos**2 / total))
    x_dd = q - M_P * LENGTH * theta_dd * cos / total
    v, omega = v + DT * x_dd, omega + DT * theta_dd
    return np.stack([x + DT * v, v, theta + DT * omega, omega], axis=-1)
```

</details>

Use the initial-state arrays and `observe` helper in `examples/cartpole_objective.py` to construct `X`. Its columns contain the five policy observations, and `y` is a zero placeholder that the objective ignores. The 16 training starts cover hanging, horizontal, and near-upright states; the 64 held-out starts use wider ranges of position, speed, angle, and pole rate.

### The reward

At each step, the objective accumulates the reward

$$ r = 2\cos\theta - 0.05\,x^2 - 0.01\left(\frac{F}{10}\right)^2 - 0.005\left(\frac{F - F_{\text{prev}}}{10}\right)^2 $$

The upright term contributes up to 2 per step. The remaining terms penalize, in order, cart displacement, normalized actuator force, and normalized force changes. Here $F$ is the clipped force and $F_{\mathrm{prev}}$ is the preceding force. The objective returns the negative mean of $r$ over 500 steps and 16 starts, so an upright stationary policy has loss near $-2$.

The complete objective source is defined in `examples/cartpole_objective.py` as the Python string `CARTPOLE_REWARD`. The block below shows its Julia function body and abbreviates the state unpacking, observation-buffer setup, integration step, and divergence check supplied by that script. Treat it as a behavior excerpt, not a standalone definition.

```julia
function cartpole_reward(ex, dataset::Dataset{T,L}, options)::L where {T,L}
    n = size(dataset.X, 2)
    # ... unpack x, v, theta, omega from the observation columns ...
    for _ in 1:500
        # ... refill obs from the current state ...
        raw, ok = eval_tree_array(ex, obs, options)
        (ok && length(raw) == n && all(isfinite, raw)) || return L(Inf)

        force = 10.0 .* clamp.(raw, -1.0, 1.0)
        earned += sum(
            2.0 .* cos_t .- 0.05 .* x .^ 2 .-
            0.01 .* (force ./ 10.0) .^ 2 .-
            0.005 .* ((force .- previous) ./ 10.0) .^ 2
        )
        previous = force
        # ... one semi-implicit Euler step, then the divergence check ...
    end
    return L(-earned / (500 * n))
end
```

Invalid evaluations and divergent trajectories receive a penalty.

### The search

```python
import sympy

from pysr import PySRRegressor

model = PySRRegressor(
    operators={
        1: ["square", "abs", "tanh"],
        2: ["+", "-", "*", "/", "max", "min"],
        3: ["ifelse(t, a, b) = t > 0 ? a : b"],
    },
    extra_sympy_mappings={
        "ifelse": lambda t, a, b: sympy.Piecewise((a, t > 0), (b, True))
    },
    loss_function_expression=CARTPOLE_REWARD,
    loss_scale="linear",  # rewards make the loss negative, which log scaling forbids
    maxsize=28,
    maxdepth=10,
    parsimony=0.001,
    niterations=120,
    populations=32,
    population_size=64,
    parallelism="multithreading",
    random_state=0,
)
model.fit(X, y, variable_names=["x_n", "v_n", "s", "c", "omega_n"])
```

`loss_scale="linear"` is required here because logarithmic scaling accepts nonnegative losses, including zero, while this reward can make the loss negative. The operator set supplies arithmetic, `max`, and `min`; the user-defined ternary `ifelse` can switch between swing-up and balancing behavior. `extra_sympy_mappings` translates that custom operator to a SymPy `Piecewise` expression for export. The search uses multithreading, so repeated fits can differ even with the same nominal seed.

### Evaluating a policy

One controller returned by the search is:

```
(tanh(((0.46341985 * v_n) + (omega_n + (s + omega_n))) / 0.00043365502) - v_n) / 0.97654253
```

A low training loss alone does not establish successful swing-up from new initial states. In `examples/cartpole_objective.py`, `mean_rewards(model, HELD_OUT_STARTS, index=...)` evaluates a selected equation on the wider held-out set, and `check(model)` requires positive mean reward on every start. Use these rollout checks when choosing a policy.

Allow roughly two hours for the search configuration above. The complete runnable example is `examples/cartpole_objective.py`.

## Inventing a pseudorandom generator with no target

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D7.mp4"></video>

Ordinary regression supplies `y` and compares every candidate output with a target. This example instead searches for a 32-bit state update $x \mapsto f(x)$ whose iterates have specified properties. Its score comes from how the map behaves, so the objective defines what counts as a good generator without consulting a target array.

The data rows provide 32 random nonzero seeds. `y` is a zero column retained for the estimator interface; the objective never reads it.

```python
import numpy as np

from pysr import PySRRegressor, TypeSpec

WIDTH = 32
SEEDS = [int(v) for v in np.random.default_rng(0).integers(1, 1 << WIDTH, size=32)]
X = np.empty((len(SEEDS), 1), dtype=object)
for i, seed in enumerate(SEEDS):
    X[i, 0] = seed
y = np.zeros(len(SEEDS), dtype=object)
```

Each state is an unsigned 32-bit word, wrapped in a Julia type. The `shl` and `shr` operators below pass their `UInt32` shift count directly; this example does not mask counts into `0:31`, so arbitrary sampled counts follow Julia's shift semantics and bits shifted out of the 32-bit word are discarded. The operations do not rotate the state. The `sample` hook favors values from 0 through 31 because those are common shift distances, while still sampling arbitrary `UInt32` values. Half of mutations flip a single bit, allowing a local change to a tap constant; the other half adds either 1 or `typemax(UInt32)`. The `string` hook displays values below 32 in decimal and larger words as eight hexadecimal digits.

```python
from pysr import TypeSpec

SPEC = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
    loss_type="Float64",
)
```

The operators form a restricted bitwise instruction set: complement, XOR, AND, OR, left shift, and right shift. The search excludes arithmetic and floating-point operations.

### What the objective scores

`examples/prng_period.py` defines the objective as a Julia source string named `PRNG_LOSS`. It includes the period-certificate implementation used below.

For every candidate, the objective walks 256 steps from each seed and adds six terms that penalize departures from their desired properties:

- `period` is $1 - \log_2(\mathrm{proven})/32$, with `proven` equal to the certified orbit length when certification succeeds.
- `balance` is the mean squared bias of each bit column away from one half.
- `autocorr` is the mean squared normalized self-correlation of each bit column at lags 1, 2, 3, and 5.
- `diffusion` penalizes an average of fewer than four changed output bits when each input bit is flipped on a fixed probe set.
- `edge` penalizes output bits that respond to fewer than two distinct input-bit flips across that probe set.
- `shear` measures correlations between bit `b` and shifted bit `b + d` at one- and two-step lags, for nonzero shifts `d` from -3 to 3. It keeps a plain shift from receiving a good score.

### Certifying a period of four billion without walking it

The requested nonzero orbit contains $2^{32}-1 = 4{,}294{,}967{,}295$ states, so a direct walk is impractical. The certificate evaluates the candidate at three groups of inputs: zero, all 32 basis states, and 32 fixed probes. It requires zero to map to zero and the image of each probe to agree with the matrix assembled from the basis images. These finite checks only screen for GF(2) linearity on the supplied evidence; they do not prove global linearity for an arbitrary expression.

A successful certificate gives the candidate's period provided the expression is globally GF(2)-linear. If certification fails, the objective uses the orbit length observed during the finite walk. The period-checking implementation is in `examples/prng_period.py`.

### Settings

The settings below pass the source-defined `PRNG_LOSS` objective to PySR:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=SPEC,
    operators={
        1: ["bnot(a::Word) = Word(~a.bits)"],
        2: [
            "bxor(a::Word, b::Word) = Word(xor(a.bits, b.bits))",
            "band(a::Word, b::Word) = Word(a.bits & b.bits)",
            "bor(a::Word, b::Word) = Word(a.bits | b.bits)",
            "shl(a::Word, b::Word) = Word(a.bits << b.bits)",
            "shr(a::Word, b::Word) = Word(a.bits >> b.bits)",
        ],
    },
    loss_function=PRNG_LOSS,
    batching=False,
    maxsize=20,
    populations=8,
    population_size=30,
    ncycles_per_iteration=30,
    niterations=800,
    deterministic=True,
    parallelism="serial",
    random_state=0,
    verbosity=0,
)
model.fit(X, y, variable_names=["x"])
```

Keep `batching=False` because the objective walks every seed; a minibatch would score a different problem. Each candidate requires a 256-step walk over all 32 seeds. Allow about an hour for this search.

### Results

The script's `check(model)` looks for a front member that passes a separate finite linearity screen and whose inferred GF(2) matrix has order $2^{32}-1$. As with the objective's certificate, this establishes the candidate's period only if the expression is globally GF(2)-linear.

One recovered expression is:

```
bxor(shl(x, 1), bxor(shr(bxor(shr(x, 1), x), 1), band(x, 0xffffffe3)))
```

Fixed shifts, XOR, and AND with a constant mask make this expression globally linear by construction. For such a map, matrix order $2^{32}-1$ implies that every nonzero word lies on one orbit of that length. The remaining score terms assess the generator's finite-sample balance, correlations, and diffusion.

The complete runnable example is `examples/prng_period.py`.
