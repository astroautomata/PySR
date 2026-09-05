# Search behaviour

## Automatic batching on a large dataset

Most of the work in a symbolic regression run is evaluating candidate expressions
against data, so the cost of a search grows with the number of rows. Beyond a few
thousand rows, though, the ranking between two population members is usually settled
well before every row has been read. PySR's `batching` option exploits that. The
docstring states the contract:

> Whether to compare population members on small batches during evolution. Still uses
> full dataset for comparing against hall of fame. `"auto"` lets SymbolicRegression.jl
> choose based on the dataset.

The two halves matter separately. Evolution, the inner loop where members of a
population are compared against each other, sees a mini-batch. The hall of fame, which
is what you actually read off at the end, is still scored on every row. A batch only
decides which candidates survive to be considered, and the accuracy you are shown is
measured against the whole dataset.

The batch size follows the row count when `batch_size` is left at `None`: the full
dataset at 1000 rows or fewer, 128 below 5000, 256 below 50,000, and 512 at 50,000 and
above. The problem here has 20,000 rows, so evolution compares members on 256 rows at a
time while the hall of fame is still scored on all 20,000. `batching="auto"` is the
default in PySR 2.1.0, and it is written out explicitly below only because it is the
subject of this example.

The target is a two-term function of five inputs, three of which are irrelevant:

<details>
<summary>Data generation code</summary>

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (20000, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

</details>

Nothing about the search is tuned for the dataset size. The operator set is the usual
four arithmetic operations plus `cos` and `exp`, and `maxsize`, `populations`,
`population_size`, and `ncycles_per_iteration` are all left at their defaults:

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

`deterministic=True` with `parallelism="serial"` and a fixed `random_state` makes the
run reproducible; the other examples on this page use the same three settings for the
same reason. Drop the first two to use all your cores.

Across five seeds, all five recover the target exactly, taking between 54 and 90 seconds
each. The simplest exact member of the front is complexity 10 on every seed, at a loss of
9.4e-14, and up to the argument order of the commutative operators it is the same
expression each time:

```
(cos(x3) * 2.5382) + ((x0 * x0) + -0.5)
```

The constant 2.5382 is recovered to the digits printed, and the three irrelevant inputs
never appear. Every seed's front shows the answer assembling in two steps: at complexity 8
the best expression has the shape right and the amplitude close, with no offset, sitting at
a loss of 0.2489 on all five seeds. Seed 0 reaches `(cos(x3) * 2.4986) + (x0 * x0)` there.
Adding the constant term is what takes it to numerically zero loss.

To measure what batching itself is worth, the same 20,000-row workload can be run under
PySR 2.1.0 three times over with only the `batching` argument changed. On one node at
four threads, on an AMD EPYC 7742, with five reps interleaved one condition at a time so
that machine drift cannot favour any of them:

| `batching` | median | min | max |
| --- | --- | --- | --- |
| `False` | 103.05 s | 72.82 s | 124.41 s |
| `"auto"` | 4.12 s | 3.67 s | 10.94 s |
| `True` | 4.27 s | 3.20 s | 4.54 s |

That is a 25.0x speedup from the default over the same search with batching disabled,
and `"auto"` tracks `True` closely, which is what it should do at 20,000 rows. The
speedup does not come out of search quality at this size: the median best loss was
3.3e-02 with `"auto"` against 3.9e-02 with batching off.

These timings use a shorter iteration budget than the example above, since they exist to
compare three conditions rather than to run a search to completion. The example itself
takes longer because it runs 40 iterations serially on one thread.

The full runnable script is `examples/automatic_batching.py`.
## Operators of any arity

PySR v2 keys operators by arity, so an operator set is a dictionary from number of
arguments to the operators taking that many arguments. Unary and binary are no
longer the only options: a three-argument conditional and a four-argument minimum
can sit side by side in the same search.

The target here is a clipped envelope, a rising ramp, a falling ramp, a shifted
sinusoid and a flat cap folded together by a single minimum:

$$
y = \min(x + 2,\; 2 - x,\; \sin(3x) + 1.2,\; 0.8).
$$

Written with binary `min` this needs three nested calls. Written with a
four-argument `min4` it is one call, which is why the wider operator wins the
front.

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

The 400 fitted samples are noiseless, and recovery is judged on the 4001-point
grid the search never saw, so an expression that merely interpolates the sample
points does not count as a hit.

Custom operators of any arity are defined as Julia source strings under their
arity key. Each one also needs a SymPy image before PySR will accept it: a custom
operator without an entry in `extra_sympy_mappings` is rejected at fit time,
because PySR cannot convert the discovered expression back into a symbolic form.

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

`deterministic=True` with `parallelism="serial"` and a fixed `random_state` makes
the run reproducible, at the cost of using one core. Drop those three arguments
for a faster, non-reproducible search.

At the shipped setting of 500 iterations, 4 of 5 seeds recover the law exactly on
the unseen grid, taking 244 to 362 s per seed and landing at complexity 14 to 16.
A successful front member reads

```
min4(0.8, 2.0 - x, 2.0 + x, 1.1999999 + sin(x * 3.0))
```

Reliability tracks the iteration budget, but only up to a point:

|`niterations`|seeds exact|wall per seed|recovered complexity|
|---|---|---|---|
|50|1 of 5|38 s|16|
|100|2 of 5|94 s|14, 16|
|200|3 of 5|153 s|14, 14, 16|
|500|4 of 5|292 s|14, 14, 16, 16|
|2000|4 of 5|1180 s|14, 14, 14, 14|

It saturates at 4 of 5 for a specific reason. One seed returns a bit-identical
expression and loss at 500 and at 2000 iterations, so the extra 1500 iterations
move nothing at all. Its front is dominated by the three-argument `ifelse` where
the successful seeds use `min4`, so it is sitting in an attractor that a longer
run cannot leave. Widening the search is what escapes it: `populations=40` with
`population_size=100` recovers that seed. The example ships at 500 iterations and
does not buy it back.

The full runnable script is `examples/any_arity.py`.

## Mutations and plugins

The mutation table and the plugin list are ordinary `PySRRegressor` arguments in
PySR 2. `mutations` maps a mutation configuration object to a weight, and
`plugins` takes a list of plugin configuration objects. Both arguments merge with
the shipped defaults by type: an entry replaces the default of the same type and
leaves every other default alone, so you never restate the whole table to change
one thing.

This is the configuration in full:

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

`BacksolveMutation` ships in the default table at weight 0.0, meaning it is
available but never sampled. Giving it a weight is what turns it on. Backsolve
inverts the operators sitting above a chosen subtree and fits the missing piece
from material already present in the population, so a search can complete an
additive target one term at a time instead of stumbling onto every term at once.

`AdaptiveMutationWeightsPlugin` rescales the mutation weights during the search
from the observed success rate of each mutation, keeping a multiplier per
mutation that is updated as a moving average and reinitialized on every call to
`fit`. On PySR 2.1.0 the resolved plugin set already carries an entry of this
type at its default parameters, which you can see by printing
`model.julia_options_.plugins` from a model configured with neither argument:
it reports `SimulatedAnnealingPlugin(3.17)`,
`AdaptiveParsimonyPlugin(true, true)` and
`AdaptiveMutationWeightsPlugin(0.02, 0.05, :cost, 0.5)`. Naming the plugin here
replaces that entry with an identical one, and it is where you would change
`smoothing`, `floor` or `reward`. Pairing it with a newly weighted mutation is
the point of the pair, since the adaptive weights decide how much of the search
budget that mutation receives.

The runnable version of the same configuration fits a small target so the effect
is visible:

$$ y = 2.5\cos(3x) + 0.5x^2 - 1. $$

```python
import numpy as np

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
```

```python
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

Across seeds 0 through 4, this configuration recovered the target to within
1e-5 everywhere on 3 of 5 seeds, taking 31 to 61 seconds per seed, with final
complexities of 20 to 27. The cleanest exact form was
`(cos(x + (x + x)) * 2.5) + ((((x + (x * 0.5)) - x) * x) + -1.0)` at complexity
20, which is the target with the multiplications written out as repeated
addition.

The same script with both arguments removed recovered the target on 1 of 5
seeds, in 8 to 26 seconds. Since the plugin entry resolves to the same
parameters the defaults already carry, the difference here comes from the
mutation weight: it buys a higher chance of landing the target under the same
twenty iterations, at three to four times the wall time. Neither setting reaches
the target on every seed.

`model.julia_options_.mutations` reports the merged table, and printing it after
a fit is how you check that your entry replaced a default rather than sitting
beside it. In the configured run above it lists `BacksolveMutation` at 0.1; in
the default run it lists the same mutation at 0.0.

The full runnable script is `examples/mutations_and_plugins.py`.

## Adaptive mutation weights

Every mutation kind in PySR has a base weight that decides how often the search
draws it. `AdaptiveMutationWeightsPlugin` makes those weights respond to the run:
per mutation kind it counts attempts and strictly improving children, and scales
that kind's base weight by a learned multiplier. Kinds that keep paying off get
sampled more often, and kinds that keep failing get sampled less. The plugin is
part of PySR's default plugin set, so an ordinary `fit` already uses it. This
example shows how to configure it, and how to read the learned weights back out.

The target is a simple one, chosen so the run finishes quickly and the interesting
part is the configuration rather than the discovery:

$$y = 2.5382\cos(x_3) + x_0^2 - 0.5.$$

<details>
<summary>Data generation code</summary>

```python
import numpy as np

rng = np.random.default_rng(20260817)
X = rng.uniform(-3.0, 3.0, size=(200, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

</details>

The plugin has four settings. `smoothing` is the EMA factor for the multiplier
update, so small values mean the multipliers move slowly and reflect a long
history. `floor` clamps the sampled kind's target ratio to `[floor, 1/floor]`
before that update, which stops a single lucky or unlucky streak from pinning a
weight at zero or blowing it up. `reward` selects the objective used to decide
whether a child improved, either `"cost"` (the default, loss plus the parsimony
term) or `"loss"`. `adaptation_strength` sets how much of the learned multiplier
is actually applied, in log space: zero reproduces the shipped weights, and one
applies the multipliers with no regularization. The defaults are `smoothing=0.02`,
`floor=0.05`, `reward="cost"`, and `adaptation_strength=0.5`. The first three are
fields of the Python `AdaptiveMutationWeightsPlugin` dataclass;
`adaptation_strength` is a keyword of the Julia plugin only, so changing it means
constructing that plugin from Julia, as the tracer below already does.

Statistics are kept independently per population, so the counters you read after
a fit are an average over populations unless you run only one. This example uses
`populations=1` so that the printed numbers describe a single trajectory rather
than a mean over several. `maxsize=14` leaves a little room above the complexity
of the target, which is 10, without letting the front drift into large bloated
expressions.

Configuring the plugin means replacing the default set rather than adding to it.
Plugins merge by type: an entry in `plugins` overrides a default plugin of the
same type, while a plugin of a *new* type is appended and leaves the shipped
adaptive plugin in place, so the search would then adapt twice. The tracer below
is a new type, so it goes in `default_plugins` alongside explicit copies of the
other two shipped defaults:

```python
from pysr import PySRRegressor
from pysr.plugins import AdaptiveParsimonyPlugin, SimulatedAnnealingPlugin

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    maxsize=14,
    niterations=300,
    populations=1,
    plugins=[],
    default_plugins=[
        SimulatedAnnealingPlugin(alpha=3.17),
        AdaptiveParsimonyPlugin(),
        TracedAdaptiveMutationWeights(),
    ],
    deterministic=True,
    parallelism="serial",
    random_state=0,
    verbosity=0,
)
model.fit(X, y)
```

`deterministic=True` with `parallelism="serial"` and a fixed `random_state` is
what makes the counters reproducible from run to run. The `alpha=3.17` passed to
the annealing plugin is PySR's own default, restated here only because listing
`default_plugins` explicitly means restating everything in it.

The tracer is what lets us observe the weights. SymbolicRegression.jl exposes no
public accessor for the plugin's learned state, so `TracedAdaptiveMutationWeights`
holds a real `AdaptiveMutationWeightsPlugin` and forwards `init_plugin_state`,
`on_mutation_end!` and `condition_mutation_weights!` to it unchanged. All of the
adaptation arithmetic during the run is the library's own; the wrapper only counts
draws per mutation kind and snapshots the plugin's `multipliers` vector as it
moves. None of this is required to use the feature. Drop the wrapper and pass
`AdaptiveMutationWeightsPlugin()` (or nothing at all, since it is a default) and
the search adapts exactly the same way, silently.

<details>
<summary>Tracer plugin code</summary>

```python
from dataclasses import dataclass, field

from pysr.julia_import import AnyValue, jl
from pysr.plugins import AbstractPlugin, AdaptiveMutationWeightsPlugin

_TRACER_JL = """
if !isdefined(Main, :DocsTracedAdaptive)

const DocsAMW = SymbolicRegression.AdaptiveMutationWeightsModule

struct DocsTracedAdaptive <: SymbolicRegression.AbstractPlugin
    inner::DocsAMW.AdaptiveMutationWeightsPlugin
    stride::Int
end

DocsTracedAdaptive(; stride::Int=1000, kws...) =
    DocsTracedAdaptive(DocsAMW.AdaptiveMutationWeightsPlugin(; kws...), stride)

mutable struct DocsTracedState
    inner::DocsAMW.AdaptiveMutationWeightsState
    draws::Vector{Float64}
    seen::Int
    snapshots::Vector{Tuple{Int,Vector{Float64}}}
    registered::Bool
end

const DOCS_TRACED_STATES = DocsTracedState[]

function SymbolicRegression.init_plugin_state(p::DocsTracedAdaptive, options, dataset)
    inner = SymbolicRegression.init_plugin_state(p.inner, options, dataset)
    n = length(options.mutations)
    return DocsTracedState(inner, zeros(n), 0, Tuple{Int,Vector{Float64}}[], false)
end

function SymbolicRegression.on_mutation_end!(
    s::DocsTracedState,
    p::DocsTracedAdaptive,
    mutation::SymbolicRegression.AbstractMutation,
    event::SymbolicRegression.MutationEvent,
    dataset,
    options::SymbolicRegression.AbstractOptions,
)
    if !s.registered
        push!(DOCS_TRACED_STATES, s)
        s.registered = true
    end
    s.draws[event.mutation_idx] += 1.0
    s.seen += 1
    SymbolicRegression.on_mutation_end!(s.inner, p.inner, mutation, event, dataset, options)
    if s.seen % p.stride == 0
        push!(s.snapshots, (s.seen, copy(s.inner.multipliers)))
    end
    return nothing
end

function SymbolicRegression.condition_mutation_weights!(
    weights::AbstractVector,
    s::DocsTracedState,
    p::DocsTracedAdaptive,
    member,
    options::SymbolicRegression.AbstractOptions,
    curmaxsize,
    nfeatures,
)
    return SymbolicRegression.condition_mutation_weights!(
        weights, s.inner, p.inner, member, options, curmaxsize, nfeatures
    )
end

end
"""


@dataclass(frozen=True)
class TracedAdaptiveMutationWeights(AbstractPlugin):
    """`AdaptiveMutationWeightsPlugin`, wrapped so its learned state can be read."""

    stride: int = 1000
    inner: AdaptiveMutationWeightsPlugin = field(
        default_factory=AdaptiveMutationWeightsPlugin
    )

    def julia_plugin(self) -> AnyValue:
        jl.seval(_TRACER_JL)
        # PySR calls this once per `fit`, while building that fit's options.
        jl.seval("empty!(DOCS_TRACED_STATES)")
        return jl.DocsTracedAdaptive(
            stride=self.stride,
            smoothing=self.inner.smoothing,
            floor=self.inner.floor,
            reward=jl.Symbol(self.inner.reward),
        )
```

</details>

After the fit, the state lives in `DOCS_TRACED_STATES`, one entry per population.
The counters worth reading are `inner.attempts` and `inner.successes` (raw per-kind
tallies), `inner.multipliers` (the learned factor, normalized to unit mean over the
active kinds), and `draws` from the wrapper. The effective weight of a kind is its
base weight times `multiplier ** adaptation_strength`, so comparing the shipped
weight share against the effective share shows where the budget moved. Note that
the drawn share will not match either exactly, because the engine also zeroes
mutation kinds that are illegal for a particular member.

<details>
<summary>Reading the counters out</summary>

```python
jl.DOCS_TRACED_OPTIONS = model.julia_options_
names = [str(v) for v in jl.seval(
    "[string(typeof(first(p))) for p in DOCS_TRACED_OPTIONS.mutations]"
)]
base = [float(v) for v in jl.seval(
    "[Float64(last(p)) for p in DOCS_TRACED_OPTIONS.mutations]"
)]
attempts = [float(v) for v in jl.seval("DOCS_TRACED_STATES[1].inner.attempts")]
successes = [float(v) for v in jl.seval("DOCS_TRACED_STATES[1].inner.successes")]
multipliers = [float(v) for v in jl.seval("DOCS_TRACED_STATES[1].inner.multipliers")]
draws = [float(v) for v in jl.seval("DOCS_TRACED_STATES[1].draws")]
strength = float(
    jl.seval("DocsAMW.AdaptiveMutationWeightsPlugin().adaptation_strength")
)
```

</details>

The run driven by `main()`, at `random_state=0`, draws 182,666 mutations and ends
with these counters. `improves` is the Laplace-smoothed rate the plugin scores
with, `shipped` and `learned` are the kind's share of the total weight before and
after the learned multipliers are applied:

| mutation | draws | improves | multiplier | shipped | learned |
| --- | --- | --- | --- | --- | --- |
| `ConstantMutation` | 243 | 0.1184 | 2.216 | 0.406% | 0.702% |
| `InsertNodeMutation` | 317 | 0.1129 | 1.815 | 0.132% | 0.206% |
| `AddNodeMutation` | 74,244 | 0.1053 | 1.729 | 29.017% | 44.266% |
| `RandomizeMutation` | 13 | 0.0667 | 1.340 | 0.006% | 0.008% |
| `SimplifyMutation` | 49 | - | 1.000 | 0.025% | 0.028% |
| `DoNothingMutation` | 7,078 | - | 1.000 | 3.207% | 3.721% |
| `DeleteNodeMutation` | 19,345 | 0.0439 | 0.713 | 10.220% | 10.016% |
| `FeatureMutation` | 2,109 | 0.0384 | 0.611 | 1.175% | 1.065% |
| `OperatorMutation` | 5,175 | 0.0259 | 0.427 | 3.442% | 2.609% |
| `RotateTreeMutation` | 72,114 | 0.0240 | 0.395 | 50.045% | 36.480% |
| `SwapOperandsMutation` | 1,979 | 0.0066 | 0.111 | 2.326% | 0.899% |

`SimplifyMutation` and `DoNothingMutation` sit at a multiplier of exactly 1.000
because the plugin excludes them from adaptation, which is why their rate column
is empty. The rest are ordered by what the run learned about them. Adding a node
improved a member about four times as often as rotating the tree did, and the
budget follows: growth takes the share that tree rotation loses, while operand
swapping, the least productive kind here, falls to under two fifths of its
shipped share.

Across five seeds the target came out exactly in all five, at complexity 10 and a
loss of 8.2e-14 against the example's threshold of 1e-12. Seed 0 recovers
`((cos(x3) * 2.5382) + (x0 * x0)) - 0.50000006`, and the other four the same
expression written as `+ -0.50000006`; the last digits are where constant
optimization in the default float32 precision settled rather than on exactly
`-0.5`. The first `fit` in a process takes about 28 seconds and every later one
about 2.3 seconds, so nearly all of that first number is Julia compilation rather
than search.

How much of a difference does the adaptation make to how often constants get
changed? Setting the Julia plugin's `adaptation_strength` to zero leaves everything
else identical, including the counters the plugin keeps, and only stops the learned
multipliers from reaching the sampling weights. Running both conditions on the same
five seeds, `ConstantMutation` accounts for 1,104 of 912,590 mutations with
adaptation at its default and 659 of 912,057 with it switched off, so constants get
changed 1.67 times as often, pooled over the five seeds. Per seed the factor ranges
from 1.20 to 2.49, on between 79 and 244 constant mutations, so a single seed says
little on its own. The final learned multiplier on `ConstantMutation` ranges from
1.46 to 2.65 across the five, and the within-run weight-share ratio, the shipped
column against the learned column above, from 1.36 to 1.78. Adaptation also
recovered the target on all five seeds where switching it off recovered on three,
which five seeds are too few to make more of than a hint.

The full runnable script is `examples/adaptive_mutation_weights.py`.

## The backsolve mutation

Most mutations edit an expression and hope the edit helps. The backsolve
mutation works in the other direction. It picks a subtree, inverts the operators
sitting above that subtree to work out what value would have to appear in the
hole for the expression to match the data, and then fits a replacement for the
hole as a sparse weighted sum of subexpressions already present in the
population. The question changes from "how do I improve this whole expression"
to "what belongs in this one slot", and that second question is a linear problem
once the parts are in hand.

This mutation is off by default: `weight_backsolve` starts at 0.0. Below we turn
it on for a target that the ordinary mutations struggle with.

Our target mixes an oscillation with a quadratic:

$$ y = 2.5\cos(3x) + 0.5x^2 - 1. $$

```python
import numpy as np

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
```

The operator set deliberately omits division and any square, so the quadratic
has to be assembled as `x * x` while the amplitude and frequency of the cosine
are found as constants. `precision=64` keeps the linear solve behind the
mutation well conditioned, and `maxsize=30` leaves room for the full three-term
form. Like the other examples here, this one pins `deterministic=True`,
`parallelism="serial"`, and `random_state` so the run reproduces exactly:

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

The interesting way to run this is as an on/off comparison: the same data, the
same operators, the same ten iterations, with only `weight_backsolve` changing.
Both conditions were run on seeds 0 through 4, and "recovered" below means the
maximum absolute error over all 200 rows falls under 1e-10.

| `weight_backsolve` | Recovered | Best front loss | Median | Wall |
| --- | --- | --- | --- | --- |
| 1.0 | 5 of 5 | 1.0e-33 to 1.2e-29 | 1.5e-31 | 116 to 167 s |
| 0.0 (default) | 0 of 5 | 6.6e-20 to 8.7e-03 | 4.1e-04 | 5.0 to 5.3 s |

With the mutation enabled every seed lands on the target in closed form. The
cleanest recovery is seed 4, whose simplest exact front member is complexity 16
and reads

```
((cos((x + x) + x) * 2.5) + ((x * x) * 0.5)) + -1.0
```

Seeds 0 and 3 give the same complexity-16 shape with the constants perturbed in
the last few digits. Without the mutation the search usually stops well short,
at a loss around 4e-04 and a maximum absolute error of a few percent, on
expressions of comparable complexity that nest cosines instead of separating the
two terms. One seed in five is an exception worth naming: seed 4 with the
mutation off reached a loss of 6.6e-20 and a maximum absolute error of 4.6e-10,
so it came within a factor of five of passing the same check. The ordinary
mutations can find this target, they just rarely do inside ten iterations.

The wall times are not a case of one run finishing early. Both conditions use
the whole ten-iteration budget; the enabled runs are roughly twenty times slower
per iteration because each backsolve event pays for an inversion and a linear
solve.

You can confirm the recovery directly rather than reading the loss column:

```python
prediction = np.asarray(model.predict(X), dtype=float)
print(f"max abs error: {np.max(np.abs(prediction - y)):.3g}")
```

To watch individual events rather than the outcome, `backsolve_events` in the
script applies one backsolve mutation to every member of the finished search's
populations and returns the ones the mutation's own acceptance gate let through,
each with the parent's loss before and the child's loss after. Across the five
seeds it swept 837 members per seed and accepted between 295 and 414 events. The
median accepted parent carried a loss between 4.2 and 6.2, and the median child
came back between 2.0e-21 and 1.2e-16: by the end of the run the population
holds the parts that compose the answer exactly, so the fit that fills the hole
is exact too.

Earlier in the search the drops are large but finite. Stopping the same search
after two iterations, where parent losses are still order one, the accepted
events fall by a median factor of about 0.04, and the ones whose parent loss sits
between 2 and 4 land at a median child loss of 0.12. Seed 0 contains an event
that takes a loss of 2.80 down to 0.0269, which is the pair shown in the release
video, so that figure is representative of a mid-search backsolve rather than a
lucky outlier.

Backsolve pays off when the target is a sum or composition of pieces that the
search can plausibly discover separately, since that is exactly the case where
the missing slot is a linear combination of parts the population already holds.
Each event costs a linear solve, so it is weighted alongside the other mutations
rather than applied for free. Raise `weight_backsolve` when your target looks
additive and the search is plateauing; leave it at 0.0 when it is not.

The full runnable script is `examples/backsolve_mutation.py`.
