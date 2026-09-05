# Objectives and losses

## Preamble

```python
import numpy as np

from pysr import *
```

## Custom objectives

Use `loss_function` when scoring needs the expression tree, symbolic
manipulation, or auxiliary data from the dataset. Use
`loss_function_expression` when it needs the complete expression object rather
than its underlying tree. Both accept Julia source defining one of these
signatures:

```julia
objective(tree_or_expression, dataset, options)
objective(tree_or_expression, dataset, options, idx=nothing)
```

With automatic batching, a three-argument objective receives a dataset already
restricted to the active batch. A four-argument objective receives the full
dataset and the selected row indices. This can be useful when needing to do
custom batching operations.

The following batching-aware mean squared error is equivalent to the default
objective for an unweighted scalar dataset:

```python
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

Always check the completion flag returned by `eval_tree_array`; an incomplete
evaluation must receive an infinite or suitably large loss. Return the
dataset's loss type `L`, which needs to be real even when the data type `T` is
complex.

A full objective can also reinterpret the tree. This example treats the two
children of every binary root as a rational function $P(X)/Q(X)$:

```python
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

The root operator is deliberately ignored in this objective. The equation
table therefore prints the stored tree rather than the interpreted rational
function, and automatic prediction or symbolic export cannot reproduce that
reinterpretation. Keep the objective source and apply the same transformation
when evaluating the selected equation.

## Writing the objective in Python

This pattern requires the GIL-releasing search added in PySR versions newer
than 2.1.0 (unreleased at the time of writing). On older versions, the search
thread holds Python's global interpreter lock (GIL) for the whole fit, and a
`loss_function` that calls back into Python will crash or deadlock.

The other examples on this page keep the whole objective in Julia. You can
instead write the objective in pure Python, and keep only a thin shim on the
Julia side. The shim locks the GIL, calls the Python function, and converts
the result to the loss type. The Python function receives the tree, dataset,
and options as Julia objects, and may itself call back into Julia, including
`eval_tree_array`:

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

Three details matter here:

- `PythonCall.GIL.@lock` is required. The search releases the GIL, so a shim
  that calls Python without re-locking it crashes the process.
- `precision=64` stores the data as `Float64`, so the Python float converts
  exactly to the loss type `L`.
- Always check the `completed` flag from `eval_tree_array`, and return an
  infinite loss when it is false.

This is an escape hatch rather than a fast path. Every objective evaluation
crosses the language boundary, and concurrent evaluations serialise on the
GIL. As a scale reference: a one-iteration search on a 12-point dataset called
the Python objective 970 times with `parallelism="serial"` and 929 times with
`parallelism="multithreading"` (one run each, Apple M1 Pro). A realistic
search multiplies that by the iteration count. Objectives that need to run
fast belong in Julia.

## Swinging up a cart-pole with a rollout objective

Every other example on this page has a target column: rows go in, predictions come
out, and the loss compares them one row at a time. A control policy has no such
column. Nobody can write down the correct force for a given cart-pole state,
because what makes a force correct is what happens over the next ten seconds. What
we can score is the behaviour of the whole expression in closed loop, and that is
what `loss_function_expression` is for: it hands your Julia function the candidate
expression itself, to call as often as you like before returning a number. Here it
drives a cart-pole plant with the candidate for 500 steps from each of 16 starting
states and returns minus the mean per-step reward. Most starts hang the pole
straight down, so a policy has to pump energy in, catch the pole at the top and
hold it there, all from one closed-form expression.

### The plant and what the policy sees

The plant is the textbook cart-pole: a 1.0 kg cart, a 0.1 kg pole of half-length
0.5 m, gravity 9.8, integrated semi-implicitly at `dt = 0.02` s, so the 500-step
horizon is 10 s of control. The expression returns a force in units of the 10 N
actuator cap: its output is clamped to `[-1, 1]` and multiplied by 10. The policy
sees five numbers rather than the four physical coordinates: scaled cart offset and
speed, the pole angle as a unit vector `(s, c)`, and scaled pole rate. Splitting the
angle into sine and cosine removes the wrap at $\pm\pi$ and hands the search `c` as
a measure of uprightness.

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

The 16 training starts are two exactly hanging states at cart positions 0.0 and
0.25, six more scattered around hanging, two around each horizontal pole angle, and
four near upright. The 64 held-out starts use the same recipe with every range
widened: cart offset and speed up to 0.55 instead of 0.35, angle spread 0.70
instead of 0.35, pole rate up to 1.0 instead of 0.5. `X` holds the five
observations of the training starts; `y` is zeros the objective never reads.

### The reward

Each step pays

$$ r = 2\cos\theta - 0.05\,x^2 - 0.01\left(\frac{F}{10}\right)^2 - 0.005\left(\frac{F - F_{\text{prev}}}{10}\right)^2 $$

so the pole earns up to 2 per step for standing up and the other three terms charge
for drifting along the rail, for motor effort, and for jerk. The loss is minus the
mean of $r$ over all 500 steps and all 16 starts, so a policy that holds the pole
upright and motionless scores about $-2$, and a constant one about $+1.046$.

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

Two escape hatches matter. An expression that fails to evaluate, or returns the
wrong count or anything non-finite, scores `Inf`; a rollout that leaves the
numerical box, 20 m off the rail or either rate past 100, scores `1e12`.

### The search

```python
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

`loss_scale="linear"` is the one setting you cannot omit: PySR's default complexity
scaling is logarithmic in the loss, which a reward-shaped objective breaks the
moment the loss goes negative. The operators are plain arithmetic plus `max`, `min`
and a ternary `ifelse`, enough to express a switch between a swing-up law and a
balancing law; `ifelse` is user-defined, so `extra_sympy_mappings` supplies the
SymPy equivalent that lets the champion be exported. The search runs multithreaded
and is not reproducible seed for seed.

### What the runs found

Five seeds of the script as written land champions between loss -0.956 and -1.640
at complexity 16 to 23, each taking 5831.6 to 7244.8 seconds, so budget about two
hours per run:

| seed | wall (s) | complexity | loss | all 64 held-out positive |
|---:|---:|---:|---:|:--|
| 0 | 6866.26 | 20 | -1.0682108 | yes |
| 1 | 6580.95 | 16 | -0.95603234 | yes |
| 2 | 5831.59 | 16 | -1.5824542 | yes |
| 3 | 7244.76 | 22 | -1.5051432 | no |
| 4 | 6289.25 | 23 | -1.6398351 | no |

Three of the five champions earn positive mean reward on all 64 held-out starts, so
that claim holds on three runs in five and no single run can be relied on for it.

Training loss does not order the held-out outcome, and it inverts it here. The two
seeds with the best training losses, 4 at -1.6398351 and 3 at -1.5051432, are the
two that fail; seed 1, the worst of the five at -0.95603234, holds all 64. Sixteen
starts is a small training set for a policy, and a law can exploit them while
staying fragile at the wider angles and rates of the held-out set. To make the
held-out property reliable, widen or enlarge the training starts rather than
lengthen the search. Seed 2 gets both: complexity 16, the second-best training
loss, and all 64 held-out starts positive.

```
(tanh(((0.46341985 * v_n) + (omega_n + (s + omega_n))) / 0.00043365502) - v_n) / 0.97654253
```

Dividing by 0.00043365502 before the `tanh` makes it a smooth sign function, so the
law is close to bang-bang: drive the actuator to one rail or the other by the sign
of $0.463\,v_n + 2\,\omega_n + s$, damp with `v_n`, scale by 1.024.

Five seeds at `niterations=50` and nothing else changed finish in 1779.5 to 3379.4
seconds, land champions from -0.96431893 to -1.437396 at complexity 16 to 25, and
hold all 64 held-out starts on two of five:

| seed | wall (s) | complexity | loss | all 64 held-out positive |
|---:|---:|---:|---:|:--|
| 0 | 3379.40 | 22 | -0.96431893 | no |
| 1 | 2454.43 | 17 | -1.12998 | yes |
| 2 | 1849.11 | 17 | -1.386335 | no |
| 3 | 1779.51 | 16 | -1.2796379 | no |
| 4 | 2365.11 | 25 | -1.437396 | yes |

The Pareto fronts agree closely on the way up. Every seed at either budget starts
from a constant at loss about 1.046, the do-nothing policy that lets the pole hang,
then passes `omega_n - x_n` at complexity 3 and `tanh(omega_n) - x_n` at complexity
4 with identical losses of 0.648032 and 0.281043, and first goes negative between
complexity 4 and 7. The film clip shows this front filling in, rolling its rungs
out from one shared hanging start.

The full runnable script is `examples/cartpole_objective.py`.

## Inventing a pseudorandom generator with no target

Every other example here fits a target: you have `y`, and the loss measures how close a
candidate gets to it. This search has nothing to fit. We want a 32-bit state update
`x -> f(x)` that behaves like a pseudorandom generator, and no array of correct answers
exists, because what is being asked for is a property of the map rather than a value at
each row. The definition of a good expression lives entirely in the loss function.

The inputs are 32 random nonzero seed states, one per row, and `y` is a column of zeros
that the objective never reads:

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

Values are unsigned 32-bit words wrapped in a Julia type, so shifts wrap and truncate the
way a real generator does. The `sample` hook is biased toward small values, since shift
distances are the common case for a constant here, and the `mutate` hook flips single bits
half the time so evolution can adjust one bit of a tap constant without discarding it. The
`string` hook prints small words as decimals and everything else as eight hex digits:

```python
SPEC = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
    loss_type="Float64",
)
```

The operator set is the instruction set of a shift-register generator: complement, xor,
and, or, and the two shifts. No arithmetic and no floating point.

### What the objective scores

The loss iterates each candidate for 256 steps from every seed and scores the generator
that walk describes. It is a sum of six terms, each of which is 0 exactly when its
requirement is met:

- `period`, `1 - log2(proven) / 32`, where `proven` is the certified orbit length.
- `balance`, the mean squared bias of each of the 32 bit columns away from half ones.
- `autocorr`, the mean squared correlation of each bit column with itself at lags 1, 2, 3
  and 5.
- `diffusion`, a penalty when flipping one input bit moves fewer than four output bits.
- `edge`, a penalty when some output bit depends on fewer than two input bits.
- `shear`, the correlation of bit `b` at one step with bit `b + d` a step or two later, for
  shifts `d` from -3 to 3, which is what stops a plain shift from scoring well.

### Certifying a period of four billion without walking it

The interesting term is the period, because the orbit we are asking for is 4294967295
states long and nothing may iterate that far. The certificate is linear algebra over
GF(2). The objective evaluates the candidate on 0, on the 32 basis states and on 32 fixed
probe states. If the output is zero at zero and the probe images agree with the matrix
read off the basis images, the map is GF(2)-linear on that evidence, and that matrix is
its columns. It then builds the minimal polynomial of the first seed's Krylov sequence
and tests it for irreducibility; if it is irreducible, the orbit length is exactly the
multiplicative order of `x` modulo that polynomial, found by dividing the prime factors
out of `2^deg - 1`. A candidate whose period cannot be certified is credited only with
the orbit it was seen to walk in 256 steps, so a map is paid for what it can prove.

<details><summary>The certificate entry point</summary>

```julia
function _prng_certified(tree, options, seed::UInt32)
    m = 33 + length(PRNG_PROBES)
    grid = Matrix{Word}(undef, 1, m)
    grid[1, 1] = Word(UInt32(0))
    for b in 0:31
        grid[1, 2 + b] = Word(UInt32(1) << b)
    end
    for (i, p) in enumerate(PRNG_PROBES)
        grid[1, 33 + i] = Word(p)
    end
    out, ok = eval_tree_array(tree, grid, options)
    ok || return UInt64(0)
    out[1].bits == 0 || return UInt64(0)

    cols = [out[2 + b].bits for b in 0:31]
    for (i, p) in enumerate(PRNG_PROBES)
        _prng_matvec(cols, p) == out[33 + i].bits || return UInt64(0)
    end

    f = _prng_minpoly(cols, seed)
    f == 0 && return UInt64(0)
    deg = _prng_deg(f)
    _prng_irreducible(f, deg) || return UInt64(0)
    return _prng_order(f, deg)
end
```

</details>

### Settings

```python
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

`batching=False` matters: the objective walks every seed and indexes the sample axis, so a
minibatch would score a different problem. The layout is eight small populations evolved
deeply, because each candidate costs a 256-step walk of all 32 seeds, which makes
generations worth more than members. `deterministic=True`, `parallelism="serial"` and a
fixed `random_state` make a run reproducible.

### Results

Success is a property of the front, and it is re-derived in Python by a different route
than the loss uses: `check` reads each candidate's GF(2) matrix out of `predict`, confirms
the map is linear on 256 random states, and computes the order of that matrix by repeated
squaring. Order 4294967295 forces the period, since that order divides a power of two
times a product of `2^d - 1` over the degrees of the minimal polynomial's irreducible
factors, and 65537 divides it only when 32 divides `d`.

On PySR 2.1.0, all 5 of 5 seeds put a certified full-period generator on the front, at
complexities 15, 17, 17, 19 and 19, with losses between 0.000326609973347 and
0.0315960661365118. Once the period is certified its term is essentially zero, so the
residual loss is the statistical terms: how balanced, uncorrelated and diffusive the
generator is beyond having the right orbit length. The film clip quotes fifteen nodes,
which is the smallest of the five certified winners; the others land two or four nodes
larger on the same kind of expression. Different seeds settle at different sizes, and a
spread of 15 to 19 nodes at comparable loss is the ordinary shape of that.

The smallest one, from the seed that reached complexity 15 at loss 0.0060682148590457, is

```
bxor(shl(x, 1), bxor(shr(bxor(shr(x, 1), x), 1), band(x, 0xffffffe3)))
```

which is a linear-feedback shift register in the form an evolutionary search finds: a left
shift for the state advance, a right-shifted xor of the state feeding taps back in, and a
mask selecting which low bits are fed.

This is the slowest example in the set, at roughly 54 minutes per seed, measured at
3214.45 to 3248.18 seconds. The cost sits in the objective rather than the certificate:
every candidate evaluation is a 256-step walk of 32 seeds plus 32 single-bit flip probes,
while the algebraic period test is a few dozen 32-bit operations.

The full runnable script is `examples/prng_period.py`.
