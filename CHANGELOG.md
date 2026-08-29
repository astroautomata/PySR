# Changelog

## [2.1.0](https://github.com/astroautomata/PySR/compare/v2.0.0...v2.1.0) (2026-08-29)


### Features

* stop autoloading the juliacall IPython extension by default ([#1325](https://github.com/astroautomata/PySR/issues/1325)) ([b3f91a1](https://github.com/astroautomata/PySR/commit/b3f91a17cb75da8b2a5a520bf157f7491dd74e4c))
* stop searches gracefully on interrupt instead of killing the kernel ([#1310](https://github.com/astroautomata/PySR/issues/1310)) ([5bd53f6](https://github.com/astroautomata/PySR/commit/5bd53f662ee1d3516f084d556f7871e2299411a0))


### Documentation

* add two research showcase papers ([#1323](https://github.com/astroautomata/PySR/issues/1323)) ([7727b01](https://github.com/astroautomata/PySR/commit/7727b0179a97114de317f136c15b1f83bb7ceff0))
* update SKILL.md to v2 ([#1321](https://github.com/astroautomata/PySR/issues/1321)) ([d64ef62](https://github.com/astroautomata/PySR/commit/d64ef62b48926b1871dd0930cc5ee21d6618814c))

## [2.0.0](https://github.com/astroautomata/PySR/compare/v1.5.9...v2.0.0) (2026-08-25)

PySR 2.0.0 brings a two-year transformation of the library into the Python API, moving from a fixed scalar-tree search interface to a modular PyTorch-like framework for symbolic learning while keeping the familiar v1 estimator workflow. Operators can take any number of arguments, `TypeSpec` supports user-defined value types, and mutations, crossovers, and the search loop are configurable objects. Guesses mix expressions into populations throughout a run, which helps connect PySR to agentic coding loops. Automatic batching and a reusable backend evaluation buffer make large searches faster with less configuration.

---

### Highlights

#### Operators with any number of arguments

`operators` takes an arity-keyed dict, so ternary operators like `clamp`, `fma`, and `muladd`, along with `max`/`min` over three or more arguments, are searchable ([#999](https://github.com/astroautomata/PySR/pull/999)). Before v2 you had to fake `clamp(x0*x1, -1, 1)` as a tall nest of binary operators, which the search rarely found and never found cheaply. `binary_operators` and `unary_operators` still work, and they are mutually exclusive with `operators`.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    operators={1: ["sin"], 2: ["+", "*", "-"], 3: ["clamp", "fma"]},
    niterations=40,
)
model.fit(X, y)
```

<details>
<summary>How arity flows through the tree type, dimensional analysis, and SymPy export</summary>

The node type became `Node{T,D}` in DynamicExpressions ([#127](https://github.com/SymbolicML/DynamicExpressions.jl/pull/127)), where `D` is the maximum arity, and SymbolicRegression.jl generalized mutation, crossover, constraint checking, and dimensional analysis over it ([#471](https://github.com/astroautomata/SymbolicRegression.jl/pull/471), [#472](https://github.com/astroautomata/SymbolicRegression.jl/pull/472), [#464](https://github.com/astroautomata/SymbolicRegression.jl/pull/464)).

`constraints` entries must now match operator arity exactly: a tuple of length N for an N-argument operator, else `ValueError: Operator '<op>' has arity N but constraint tuple has length M`. Unary operators still default to `-1`, and arity 2 and above default to `tuple([-1] * arity)`.

SymPy export keeps up ([#999](https://github.com/astroautomata/PySR/pull/999)): `Max(*args)` and `Min(*args)` replace the old two-argument `Piecewise` form, and `fma`, `muladd`, and `clamp` gained mappings. That is what makes these operators usable outside Julia.

</details>

#### Seed the search with `guesses`

Give PySR any guess for the final expressions, and it mixes those guesses into the populations throughout the search ([#999](https://github.com/astroautomata/PySR/pull/999); backend [#469](https://github.com/astroautomata/SymbolicRegression.jl/pull/469), [#500](https://github.com/astroautomata/SymbolicRegression.jl/pull/500)). When `should_optimize_constants=True`, constants in a guess are optimized before the candidate enters the population, so the structure can be useful even when its initial constants are inaccurate. `fraction_replaced_guesses` controls the fraction of each population drawn from guesses at the end of every cycle. Guesses also support custom value types ([#1316](https://github.com/astroautomata/PySR/pull/1316)).

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["sin"],
    guesses=["sin(x0 * 2.1 - 0.5)", "x0 * 3.0 + x2"],
    fraction_replaced_guesses=0.01,
)
model.fit(X, y)
```

<details>
<summary>Accepted guess shapes</summary>

`guesses` is a `PySRRegressor` constructor parameter, not a `fit` keyword.

Accepted shapes:

- `list[str]` for single-output regression.
- `list[list[str]]` for multi-output, one inner list per output. A plain list with `nout > 1` raises `ValueError: For multi-output (nout > 1) guesses must be a list of lists`.
- `list[dict[str, str]]` for `TemplateExpressionSpec`, keyed by sub-expression name, using `#1`, `#2` as placeholders for the sub-expression arguments.

Preparation happens in `_prepare_guesses_for_julia` (`pysr/sr.py:3413`).

</details>

#### `TypeSpec`: symbolic regression over your own value type

Declare a Julia struct and PySR will search over it ([#1280](https://github.com/astroautomata/PySR/pull/1280)). 2D vectors, strings, tensors, variable-length constant containers: anything you can write as a struct with a sampler. PySR compiles the declaration into an isolated fingerprinted Julia module and wires up evaluation, mutation, constant optimization, printing, and serialization. The hall of fame then prints equations whose leaves are not numbers.

```python
from pysr import PySRRegressor, TypeSpec

type_spec = TypeSpec(
    name="Vec2",
    fields={"data": "Vector{Float64}"},
    sample="rng -> Vec2(randn(rng, 2))",
    scalar_constants="value -> value.data",
    with_scalar_constants="(value, constants) -> Vec2(constants)",
)
model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: ["rotate90(x::Vec2) = Vec2([-x.data[2], x.data[1]])"],
        2: ["add_vectors(x::Vec2, y::Vec2) = Vec2(x.data + y.data)"],
    },
    elementwise_loss="vector_loss(x::Vec2, y::Vec2)::Float64 = sum(abs2, x.data - y.data)",
)
model.fit(X, y)
```

<details>
<summary>Validation, module compilation, and what each hook buys you</summary>

`TypeSpec.__post_init__` validates eagerly: field names must be identifiers, `fields` must be non-empty, `scalar_constants` and `with_scalar_constants` must be supplied together, and a spec without them requires an explicit `mutate` so the search has some way to move.

`scalar_constants` and `with_scalar_constants` are what turn on continuous constant optimization: the first flattens a value to a `Vector{Float64}`, the second rebuilds a value from optimized numbers. Supply `mutate` to define discrete moves instead, which is how string and container types work.

Each spec compiles to one deterministic fingerprinted module per process. Identical source reuses the existing module; conflicting source for the same name warns on replacement, and checkpoint restore refuses a conflicting definition rather than silently binding to the wrong struct. Expression specs opt in through a new `supports_type_spec` property, `True` for `ExpressionSpec` and `TemplateExpressionSpec`.

Under `parallelism="multiprocessing"`, a preamble that needs an external Julia package also needs `worker_imports=[...]`.

For boxed element types, DynamicExpressions added fused-kernel indexing ([#198](https://github.com/SymbolicML/DynamicExpressions.jl/pull/198)): 27.9% faster on `String` keep-left and 15.5% faster on custom-struct addition in the backend's own benchmarks.

</details>

#### Mutations are objects you can configure

The thirteen built-in mutations became classes, passed as a weighted mapping; two of them, `ConstantMutation` and `BacksolveMutation`, carry their own hyperparameters ([#1282](https://github.com/astroautomata/PySR/pull/1282); backend [#610](https://github.com/astroautomata/SymbolicRegression.jl/pull/610), [#645](https://github.com/astroautomata/SymbolicRegression.jl/pull/645), [#663](https://github.com/astroautomata/SymbolicRegression.jl/pull/663)). `mutations=` overrides or extends the defaults by type. `default_mutations=` replaces the whole set.

```python
from pysr import PySRRegressor, ConstantMutation, BacksolveMutation

model = PySRRegressor(
    mutations={
        ConstantMutation(perturbation_factor=0.1, probability_negate=0.02): 0.05,
        BacksolveMutation(
            max_library_size=1000,
            max_terms=12,
            min_improvement=1e-4,
            node_attempts=16,
        ): 0.1,
    },
)
model.fit(X, y)
```

<details>
<summary>The full class list, how weights resolve, and writing your own in Julia</summary>

Exported from `pysr.mutations`: `AbstractMutation`, `ConstantMutation(perturbation_factor=0.086, probability_negate=0.01)`, `OperatorMutation`, `FeatureMutation`, `SwapOperandsMutation`, `AddNodeMutation`, `InsertNodeMutation`, `DeleteNodeMutation`, `RotateTreeMutation`, `BacksolveMutation(max_library_size=500, max_terms=8, min_improvement=1e-3, node_attempts=8)`, `SimplifyMutation`, `RandomizeMutation`, `OptimizeMutation`, `DoNothingMutation`.

The scalar `weight_*` parameters still work and now default to `None`, with fallbacks equal to the v1 numbers. Reading `PySRRegressor().weight_add_node` gives `None` rather than `2.47`. `FeatureMutation` is new as a first-class move ([#475](https://github.com/astroautomata/SymbolicRegression.jl/pull/475), with `weight_mutate_feature` falling back to 0.1): rewiring a leaf to a different input column used to happen only as an accident of delete-then-add, and it helps most on wide-feature problems.

On the Julia side, a mutation is a type plus a `mutate!` method, so you can define your own and pass it in the same mapping without touching the search loop.

</details>

#### A composable search loop through plugins

PySR 2.0 introduces plugins as the extension interface for the search loop ([#1282](https://github.com/astroautomata/PySR/pull/1282); backend [#645](https://github.com/astroautomata/SymbolicRegression.jl/pull/645), [#663](https://github.com/astroautomata/SymbolicRegression.jl/pull/663)). Simulated annealing, adaptive parsimony, adaptive mutation weights, and mutation bursts use explicit hooks for lifecycle events, selection biasing, mutation conditioning, and population seeding. This separates optional search behavior from the optimized core and lets several plugins compose in one run.

```python
from pysr import PySRRegressor, AdaptiveMutationWeightsPlugin, MutationBurstPlugin

model = PySRRegressor(
    plugins=[
        AdaptiveMutationWeightsPlugin(smoothing=0.05, reward="loss"),
        MutationBurstPlugin(retry_attempts=8),
    ],
)
model.fit(X, y)
```

<details>
<summary>The plugin interface, defaults, and composition rules</summary>

Exported from `pysr.plugins`: `AbstractPlugin`, `SimulatedAnnealingPlugin(alpha=0.1)`, `AdaptiveParsimonyPlugin(tournament=True, mutation_acceptance=True)`, `AdaptiveMutationWeightsPlugin(smoothing=0.02, floor=0.05, reward="cost"|"loss")`, and `MutationBurstPlugin(retry_attempts=4, compound_probability=0.25, compound_max_steps=2)`.

`plugins=` extends and overrides `default_plugins=` by plugin type. `default_plugins=[]` runs the core search without the default plugin set. A custom plugin implements `AbstractPlugin` and the hooks it needs; unrelated hooks retain their default behavior.

`AdaptiveMutationWeightsPlugin` is enabled by default ([#678](https://github.com/astroautomata/SymbolicRegression.jl/pull/678)). Mutation probabilities move toward operations that improve cost. In the backend's ASV runs, multithreaded runtime was 12.9 s without adaptation and 13.1 s with it, while the aggregate score improved by 0.0143.

`MutationBurstPlugin` retries rejected mutations and can chain further mutations after an acceptance. `MutationBurstPlugin(retry_attempts=1, compound_max_steps=1)` restores the previous single-attempt behavior without consuming an extra RNG draw.

The annealing schedule was ported to a plugin bit for bit. Restoring the original quotient after a `LinRange` rewrite preserved the previous hall-of-fame hashes ([#652](https://github.com/astroautomata/SymbolicRegression.jl/pull/652)).

</details>

#### `BacksolveMutation`: analytic inversion plus sparse regression

`BacksolveMutation` inverts the evaluation path to work out what a subtree *should* have returned, then fits a replacement by greedy forward selection over a library of the population's best subtrees, constrained by the remaining complexity budget ([#573](https://github.com/astroautomata/SymbolicRegression.jl/pull/573), thanks @ayagh19; exposed by [#1282](https://github.com/astroautomata/PySR/pull/1282)). It targets exactly the case random perturbation is bad at: a correct outer form with a wrong inner argument, like `sin(3.7*x0 + 0.1)` when the true phase is 2.4.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "*", "-"],
    unary_operators=["sin"],
    weight_backsolve=0.5,   # effective default 0.0, experimental: knobs may move
)
model.fit(X, y)
```

<details>
<summary>How the inversion and the sparse fit work</summary>

For a target subtree, backsolve walks up the tree, inverting each operator on the path to the root to get the residual target the subtree needs to produce. It then builds a library of candidate basis functions from the best subtrees currently in the population and uses greedy forward selection to fit a sparse combination within the remaining complexity budget.

Tuning lives on the mutation object rather than in global weights: `BacksolveMutation(max_library_size=500, max_terms=8, min_improvement=1e-3, node_attempts=8)`. The `weight_backsolve` keyword sets its weight in the default mapping.

</details>

#### Automatic batching

`batching="auto"` and `batch_size=None` are the new defaults ([#1045](https://github.com/astroautomata/PySR/pull/1045), [#1285](https://github.com/astroautomata/PySR/pull/1285); backend [#676](https://github.com/astroautomata/SymbolicRegression.jl/pull/676)). Above 1000 rows PySR minibatches, choosing 128, 256, or 512 from the dataset size. Large fits get much faster with no configuration at all. Minibatches guide the inner evolution; hall-of-fame candidates are reevaluated on the full dataset before they are returned. Pass `batching=False` for the v1 search contract.

```python
from pysr import PySRRegressor

# 200k rows: v2 minibatches on its own
model = PySRRegressor(niterations=40)
model.fit(X, y)

# full-data evolution, as in v1
model = PySRRegressor(niterations=40, batching=False, batch_size=50)
model.fit(X, y)
```

Measured locally: the identical default `PySRRegressor` script on a 20000x5 dataset took a median 63.9 s on 1.5.9 and 12.7 s on PR #1280 head `e6064687456c8d7a1c8746f47f003b83f5d39bfb` (stale package metadata `2.0.0a2`; M1 Pro, 4 threads, 3 repeats, warmed). This is the batching default acting, and v1 prints its own warning recommending batching at this size. It is a defaults win, separate from the engine work below.

<details>
<summary>The exact batching thresholds</summary>

With `batching="auto"` and `batch_size=None`: full data for N <= 1000, batch size 128 for N < 5000, 256 for N < 50000, and 512 above that. Setting `batch_size` explicitly overrides the choice; setting `batching=True` or `batching=False` overrides the automatic decision.

</details>

#### A reusable evaluation arena in the backend

Evaluation buffers are now allocated once in a contiguous arena and reused across mutation, crossover, loss evaluation, constant optimization, and template inner calls ([#654](https://github.com/astroautomata/SymbolicRegression.jl/pull/654), [#668](https://github.com/astroautomata/SymbolicRegression.jl/pull/668); DynamicExpressions [#180](https://github.com/SymbolicML/DynamicExpressions.jl/pull/180), [#186](https://github.com/SymbolicML/DynamicExpressions.jl/pull/186); pulled into PySR by [#1282](https://github.com/astroautomata/PySR/pull/1282)). On SymbolicRegression.jl's own 8-thread benchmark this took a full search from 9.541 s to 5.880 s median with allocations dropping from 59.10 GB to 10.71 GB, and the hall of fame came out byte-identical. There is no API change; you get it by upgrading.

```python
from pysr import PySRRegressor

# nothing to configure
model = PySRRegressor(niterations=40)
model.fit(X, y)
```

Measured locally: the identical 2000x5 default fit script, warmed, 5 alternating repeats at 8 threads on an M1 Pro, ran a median 6.36 s on 1.5.9 and 5.26 s on PR #1280 head `e6064687456c8d7a1c8746f47f003b83f5d39bfb` (stale package metadata `2.0.0a2`). With batching pinned equal at 20000 rows, 1.5.9 was the faster of the two on this machine, so the arena figures above remain the backend's own result. Full numbers and method caveats are in the release measurement log.

<details>
<summary>Where the allocations went, and the Julia-level API change behind it</summary>

The 38.4% runtime and 81.9% allocation figures are SymbolicRegression.jl's own 8-thread arena benchmark, not a PySR measurement; treat them as the backend's numbers.

`EvalOptions` was renamed `EvalContext` and arena lifetimes became caller-owned ([#187](https://github.com/SymbolicML/DynamicExpressions.jl/pull/187), [#192](https://github.com/SymbolicML/DynamicExpressions.jl/pull/192), [#186](https://github.com/SymbolicML/DynamicExpressions.jl/pull/186); [#668](https://github.com/astroautomata/SymbolicRegression.jl/pull/668), [#670](https://github.com/astroautomata/SymbolicRegression.jl/pull/670)). The deprecated `EvalOptions` binding is kept. Unknown evaluation keywords now error instead of being ignored. None of this is visible from Python.

</details>

#### Faster startup

Backend precompilation was narrowed to the workload PySR's `precision=32` default actually uses, the Float32 single-output search ([#642](https://github.com/astroautomata/SymbolicRegression.jl/pull/642)), and PySR sets `precompile_float64: false` in its Julia preferences ([#1279](https://github.com/astroautomata/PySR/pull/1279)). On a clean depot, precompilation drops from about 43 s to about 17 s and the cache from 59 to 39 MiB. This is precompile time, not steady-state fit time; the backend reports no repeatable change in default first-fit time.

```python
# on a fresh Julia depot
import pysr
from pysr import PySRRegressor

PySRRegressor(niterations=1).fit(X, y)
```

<details>
<summary>Why the Float32 path</summary>

PySR defaults to `precision=32`, so precompiling the Float64 path paid for a workload most users never hit. The preference is set with the backend pin in `pysr/juliapkg.json`.

</details>

---

### Notable changes

- `autodiff_backend` accepts `"Zygote"`, `"Mooncake"`, and `"Enzyme"`, loading the Julia package on demand; Enzyme is no longer experimental ([#999](https://github.com/astroautomata/PySR/pull/999); [#468](https://github.com/astroautomata/SymbolicRegression.jl/pull/468), [#632](https://github.com/astroautomata/SymbolicRegression.jl/pull/632), [#537](https://github.com/astroautomata/SymbolicRegression.jl/pull/537), [#566](https://github.com/astroautomata/SymbolicRegression.jl/pull/566)).
- DynamicDiff v0.3 adds symbolic differentiation support for expressions containing n-arity operator nodes ([DynamicDiff #4](https://github.com/MilesCranmer/DynamicDiff.jl/pull/4)).
- All `weight_*` parameters default to `None`, with fallbacks equal to the v1 numbers, so behavior is preserved but `PySRRegressor().weight_add_node` reads `None` instead of `2.47` ([#1282](https://github.com/astroautomata/PySR/pull/1282)).
- Adaptive mutation weights are on by default: probabilities move during the run based on observed cost improvement, at parity overhead in the backend's ASV runs ([#678](https://github.com/astroautomata/SymbolicRegression.jl/pull/678), [#1282](https://github.com/astroautomata/PySR/pull/1282)).
- BREAKING: `cluster_manager="slurm"` now loads SlurmClusterManager.jl and expects an existing allocation. Request resources with `sbatch` or `salloc`, run Python once inside the allocation, and set `procs` to the allocation's task count; PySR no longer allocates for you. Other managers keep ClusterManagers ([#794](https://github.com/astroautomata/PySR/pull/794)).
- BREAKING: `requires-python >= 3.9`, `juliacall>=0.9.28,<0.9.36`, `pandas<4`, and the SymbolicRegression.jl `~2.0.0` backend requirement affect environment resolution ([#1052](https://github.com/astroautomata/PySR/pull/1052), [#1035](https://github.com/astroautomata/PySR/pull/1035), [#1129](https://github.com/astroautomata/PySR/pull/1129), [#1047](https://github.com/astroautomata/PySR/pull/1047), [#1312](https://github.com/astroautomata/PySR/pull/1312)).
- SymPy export gained `Max(*args)` and `Min(*args)` in place of two-argument `Piecewise`, plus mappings for `fma`, `muladd`, and `clamp`. Re-exported v1 models print differently ([#999](https://github.com/astroautomata/PySR/pull/999)).
- `FeatureMutation` and `weight_mutate_feature` (default 0.1) make rewiring a leaf to a different input column its own weighted move, which also removes a generate-and-reject loop in templates ([#475](https://github.com/astroautomata/SymbolicRegression.jl/pull/475), [#999](https://github.com/astroautomata/PySR/pull/999), [#1282](https://github.com/astroautomata/PySR/pull/1282)).
- `MutationBurstPlugin` retries a rejected mutation (4 attempts) and chains further mutations after acceptance (p = 0.25), flagged extra experimental; `retry_attempts=1, compound_max_steps=1` reproduces the old inner loop without consuming an extra RNG draw ([#645](https://github.com/astroautomata/SymbolicRegression.jl/pull/645), [#1282](https://github.com/astroautomata/PySR/pull/1282)).
- Custom JAX operator mappings survive checkpoint round trips ([#1199](https://github.com/astroautomata/PySR/pull/1199)).
- Documentation now includes the complete v1-to-v2 migration guide, a PDE discovery example, and updated agent skill guidance ([#1302](https://github.com/astroautomata/PySR/pull/1302), [#1311](https://github.com/astroautomata/PySR/pull/1311)).
- `worker_imports` and `worker_timeout` make external-package operators, objectives, and TypeSpec preambles work under multiprocessing, and restart a worker that stops responding ([#999](https://github.com/astroautomata/PySR/pull/999); [#488](https://github.com/astroautomata/SymbolicRegression.jl/pull/488)).

  <details>
  <summary>A multiprocessing run with an external Julia package</summary>

  ```python
  from pysr import PySRRegressor

  model = PySRRegressor(
      parallelism="multiprocessing",
      procs=8,
      worker_imports=["SpecialFunctions"],
      worker_timeout=120.0,
      unary_operators=["myerf(x) = SpecialFunctions.erf(x)"],
  )
  model.fit(X, y)
  ```

  Workers see only the listed modules; arbitrary bindings from `Main` remain unavailable.

  </details>

- With `warm_start=True`, a fit that raises now restores the previous model state before re-raising, so a bad operator or a Ctrl-C no longer destroys the equations you already had ([#1280](https://github.com/astroautomata/PySR/pull/1280), `_rollback_failed_warm_start`).
- User Julia code survives pickling and multiprocessing: operators, objectives, complexity mappings, early-stop conditions, templates, and worker definitions are replayed where v1 could fail or silently lose them ([#1280](https://github.com/astroautomata/PySR/pull/1280)).
- Mis-shaped Julia loss functions are caught before the search starts, with a message naming the wrong signature instead of a `MethodError` five minutes in ([#1138](https://github.com/astroautomata/PySR/pull/1138), [#1184](https://github.com/astroautomata/PySR/pull/1184)).
- `torch_format` modules register constants as buffers, so an exported equation follows `.to(device)`, `.cuda()`, and `.half()` and appears in `state_dict()` ([#1058](https://github.com/astroautomata/PySR/pull/1058)).
- Backend correctness fixes: constant optimization escapes zero-valued constants, multiprocessing teardown stops hanging, an `X`/`y` row mismatch errors early with a `DimensionMismatch`, expression-level losses skip simplification, discrete custom-value mutation works, and hall-of-fame CSV quoting is fixed ([#637](https://github.com/astroautomata/SymbolicRegression.jl/pull/637), [#641](https://github.com/astroautomata/SymbolicRegression.jl/pull/641), [#660](https://github.com/astroautomata/SymbolicRegression.jl/pull/660), [#674](https://github.com/astroautomata/SymbolicRegression.jl/pull/674), [#687](https://github.com/astroautomata/SymbolicRegression.jl/pull/687), [#698](https://github.com/astroautomata/SymbolicRegression.jl/pull/698) via [#1282](https://github.com/astroautomata/PySR/pull/1282)).
- `tempdir=` is respected for temporary equation files, which unblocks read-only `/tmp` and quota-constrained HPC setups ([#1207](https://github.com/astroautomata/PySR/pull/1207)).
- The canonical repo is now [github.com/astroautomata/PySR](https://github.com/astroautomata/PySR) (old links redirect) and the docs at [ai.damtp.cam.ac.uk/pysr](https://ai.damtp.cam.ac.uk/pysr) moved from MkDocs to VitePress, so bookmarked deep links and local docs commands change. The PyPI name `pysr` and `import pysr` are unchanged ([#1272](https://github.com/astroautomata/PySR/pull/1272), [#1056](https://github.com/astroautomata/PySR/pull/1056); docs [#483](https://github.com/astroautomata/SymbolicRegression.jl/pull/483), [#491](https://github.com/astroautomata/SymbolicRegression.jl/pull/491)).
- The repo ships an agent skill at `skills/pysr/SKILL.md`, so coding agents read current v2 API guidance instead of guessing at v0.x-era arguments ([#1264](https://github.com/astroautomata/PySR/pull/1264)).
- Julia only: `machine`/`fit!`/`predict`/`report` work without MLJ or MLJBase loaded, via a new `SymbolicRegressionTablesExt` ([#680](https://github.com/astroautomata/SymbolicRegression.jl/pull/680)).
- Julia only: custom crossovers via `AbstractCrossover`, mirroring the mutation API with `SubtreeCrossover`, a `crossovers=` mapping, and an `attempt` counter so an expensive crossover can cheap out on retries. No PySR keyword ([#664](https://github.com/astroautomata/SymbolicRegression.jl/pull/664), [#666](https://github.com/astroautomata/SymbolicRegression.jl/pull/666)).
- BREAKING, Julia only: `use_recorder`/`recorder_file` are replaced by `use_tracing`/`tracing_file` writing versioned JSONL. Memory scales with in-flight records rather than the whole search history, and disabled tracing is verified zero-allocation ([#651](https://github.com/astroautomata/SymbolicRegression.jl/pull/651)).

---

### Migration guide

Ordered by the sequence you will actually hit them.

#### Hard failures

##### 1. Python 3.8 install fails

`requires-python` moved from `>=3.8` to `>=3.9`. `juliacall` is constrained to `>=0.9.28,<0.9.36`, so an environment holding another juliacall consumer may fail to resolve ([#1052](https://github.com/astroautomata/PySR/pull/1052), [#1047](https://github.com/astroautomata/PySR/pull/1047), [#1312](https://github.com/astroautomata/PySR/pull/1312)).

##### 2. No v1 checkpoint loads (`ValueError`)

```python
# v1-written model.pkl
import pickle
model = pickle.load(open("model.pkl", "rb"))
# v2: ValueError: Unsupported PySR checkpoint schema: expected 3, found None.
```

Refitting is the only path. `warm_start` from a v1 checkpoint is impossible. Pickles from the 2.0 prerelease builds (schema 2) also fail at schema 3 ([#1282](https://github.com/astroautomata/PySR/pull/1282), gate at `pysr/sr.py:1623-1631`).

##### 3. `ParametricExpressionSpec` is gone (`ImportError`)

No shim ([#1277](https://github.com/astroautomata/PySR/pull/1277); backend [#656](https://github.com/astroautomata/SymbolicRegression.jl/pull/656)).

```python
# v1
from pysr import PySRRegressor, ParametricExpressionSpec
model = PySRRegressor(expression_spec=ParametricExpressionSpec(max_parameters=2))
model.fit(X, y, category=category)
model.predict(X, category=category)

# v2
from pysr import PySRRegressor, TemplateExpressionSpec
spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p": n_categories},
    combine="p[category] * f(x1, x2)",
)
model = PySRRegressor(expression_spec=spec)
model.fit(X, y)      # category is now a column of X, 1-indexed for Julia
model.predict(X)
```

##### 4. `category=` is gone from `fit` and `predict` (`TypeError`)

Same fix as above. The Julia `NamedTuple{(:class,)}` extra-data path was removed with it ([#1277](https://github.com/astroautomata/PySR/pull/1277)).

##### 5. Legacy positional `TemplateExpressionSpec` removed

Dangerous because a positional call now silently changes meaning: `combine` is the first dataclass field, where `function_symbols` used to be ([#1280](https://github.com/astroautomata/PySR/pull/1280)).

```python
# v1 legacy form (accepted in 1.5.9)
spec = TemplateExpressionSpec(["f", "g"], "sin(f(x1, x2)) + g(x3)^2", {"f": 2, "g": 1})

# v2
spec = TemplateExpressionSpec(
    combine="sin(f(x1, x2)) + g(x3)^2",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
```

`julia_expression_options()` and the `num_features` keyword are gone. A custom `AbstractExpressionSpec` subclass must implement `supports_type_spec`, `_julia_expression_spec_source(*, prototype)`, and `_julia_expression_spec_function_selector()` to support `type_spec`.

##### 6. `operators` and `binary_operators`/`unary_operators` are mutually exclusive (`ValueError`)

`_validate_and_modify_params` rejects the combination: "Cannot use `operators` with `binary_operators` or `unary_operators`."

```python
# v1
model = PySRRegressor(binary_operators=["+", "*"], unary_operators=["sin"])

# v2, either style, never both
model = PySRRegressor(operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp", "fma"]})
```

##### 7. `constraints` tuples must match operator arity exactly (`ValueError`)

`Operator '<op>' has arity N but constraint tuple has length M`. Unary operators still default to `-1`; arity 2 and above default to `tuple([-1] * arity)`.

```python
# v1
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["sin"],
    constraints={"sin": 9, "*": (-1, 9)},
)

# v2, one entry per argument
model = PySRRegressor(
    operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp"]},
    constraints={"sin": 9, "*": (-1, 9), "clamp": (-1, 5, 5)},
)
```

##### 8. HPC: `cluster_manager="slurm"` no longer allocates

Request resources with `sbatch` or `salloc`, run Python once inside the allocation, and set `procs` to the allocation's task count ([#794](https://github.com/astroautomata/PySR/pull/794)). Other managers (`pbs`, `lsf`, `sge`, `qrsh`, `scyld`, `htc`) are unchanged.

For example, a minimal batch script for 16 workers is:

```bash
#!/bin/bash
#SBATCH --ntasks=16

python script.py
```

Submit it with `sbatch pysr_job.sh`. Do not wrap the Python command in `srun`, which would start one Python driver per task.

```python
# inside script.py
model = PySRRegressor(parallelism="multiprocessing", cluster_manager="slurm", procs=16)
```

##### 9. Julia side only

`use_recorder`/`recorder_file` become `use_tracing`/`tracing_file`; `Node{T}` becomes `Node{T,D}`; `EvalOptions` becomes `EvalContext`; `ParametricExpression` and `ParametricNode` are removed from SymbolicRegression.jl; SymbolicUtils is pinned to v4, so old SciML stacks cannot co-install; `EquationSearch`, `score_func`, and the old `calculate_pareto_frontier` signatures are deprecated.

#### Silent behavior changes to re-tune or pin

| parameter | v1 | v2 | note |
|---|---|---|---|
| `annealing` | `False` | `True` | matches SymbolicRegression.jl; changes every accept/reject decision |
| `crossover_probability` | `0.0259` | `0.2` | about 8x more recombination |
| `batching` | `False` | `"auto"` | on above 1000 rows; hall-of-fame candidates are reevaluated on the full dataset before return |
| `batch_size` | `50` | `None` | full data for N <= 1000, 128 for N < 5000, 256 for N < 50000, 512 above |
| all `weight_*` | floats | `None` | fallbacks match the v1 numbers, and adaptive weights now move them during the run |
| adaptive mutation weights | off | on | `AdaptiveMutationWeightsPlugin` is in the default set |
| SymPy export of `max`/`min` | `Piecewise` | `Max`/`Min` | re-exported v1 models print differently |
| torch export constants | Python floats | registered buffers | now in `state_dict()` and follow `.to(device)` |

New `weight_*` entries: `weight_mutate_feature` (`None`, falling back to 0.1) and `weight_backsolve` (`None`, falling back to 0.0, so off).

The `crossover_probability` move to 0.2 came out of a 560-search factorial ablation (+2.24% aggregate held-out Pareto NMSE) and a 420-search sweep in which 0.20 was the only setting that helped ([#643](https://github.com/astroautomata/SymbolicRegression.jl/pull/643)). `annealing=True` matches the backend default ([#1283](https://github.com/astroautomata/PySR/pull/1283); [#652](https://github.com/astroautomata/SymbolicRegression.jl/pull/652)).

#### v1-like configuration

For a search close to the v1 defaults, set the changed values explicitly and omit the adaptive-mutation plugin:

```python
from pysr import AdaptiveParsimonyPlugin, PySRRegressor

model = PySRRegressor(
    batching=False,
    batch_size=50,
    annealing=False,
    crossover_probability=0.0259,
    default_plugins=[AdaptiveParsimonyPlugin()],
    weight_add_node=2.47,
    weight_insert_node=0.0112,
    weight_delete_node=0.87,
    weight_do_nothing=0.273,
    weight_mutate_constant=0.0346,
    weight_mutate_operator=0.293,
    weight_mutate_feature=0.0,
    weight_swap_operands=0.198,
    weight_rotate_tree=4.26,
    weight_randomize=0.000502,
    weight_simplify=0.00209,
    weight_optimize=0.0,
    weight_backsolve=0.0,
)
```

This preserves the v1 search configuration, but it does not make a run bit-for-bit identical. Backend implementation changes still alter random-number consumption and search trajectories.

---

### Other changes

Docs:

- `mutations` and `plugins` are placed in `pysr/param_groupings.yml`, so the generated options docs group them sensibly.

Bug fixes:

- `exports["jax_format"]` and `exports["torch_format"]` are `pd.Series` on the equations index rather than raw lists (`pysr/export.py`).
- Fitting with DataFrame column names containing spaces no longer breaks `predict` ([#1136](https://github.com/astroautomata/PySR/pull/1136)).
- `TemplateExpressionSpec.num_features` keys are converted to Julia symbols, fixing a silently ignored per-sub-expression feature limit on the legacy path that [#1280](https://github.com/astroautomata/PySR/pull/1280) then removed ([#1209](https://github.com/astroautomata/PySR/pull/1209)).
- `Complex{T}` derivatives work in DynamicDiff's ForwardDiff fallback and throw `DomainError` when Cauchy-Riemann disagrees, rather than returning a wrong gradient ([DynamicDiff #10](https://github.com/MilesCranmer/DynamicDiff.jl/pull/10)).

Misc:

- `pysr test` gained `autodiff` and `slurm` groups, so you can verify an Enzyme or Mooncake install, or a Slurm setup, locally ([#1111](https://github.com/astroautomata/PySR/pull/1111)).
- BREAKING: custom `AbstractExpressionSpec` subclasses must implement `supports_type_spec`, `_julia_expression_spec_source(*, prototype)`, and `_julia_expression_spec_function_selector()` to support `type_spec`; `julia_expression_options()` is gone ([#1280](https://github.com/astroautomata/PySR/pull/1280)).
- DynamicExpressions gained fused-kernel indexing for boxed element types: 27.9% faster on `String` keep-left, 15.5% on custom-struct addition, and about 1% for plain `Float64`, which matters to `TypeSpec` users ([#198](https://github.com/SymbolicML/DynamicExpressions.jl/pull/198)).
- SymbolicUtils is pinned to v4 and the SymbolicRegression.jl deprecated-API surface (`EquationSearch`, `score_func`, old `calculate_pareto_frontier` signatures) is formally deprecated in `src/deprecates.jl`.

---

### Backend versions

PySR `2.0.0` (tag `v2.0.0`) requires:

- [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl) `~2.0.0`, with `preferences: {"precompile_float64": false}` in `pysr/juliapkg.json`. [Release notes](https://github.com/astroautomata/SymbolicRegression.jl/releases/tag/v2.0.0).
- [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl) `~2.10` (up from `~1.10.1`). [Release notes](https://github.com/SymbolicML/DynamicExpressions.jl/releases).
- [DynamicDiff.jl](https://github.com/MilesCranmer/DynamicDiff.jl) `0.3` (up from `0.2`). [Release notes](https://github.com/MilesCranmer/DynamicDiff.jl/releases).
- SymbolicUtils.jl `4`.

Docs: [ai.damtp.cam.ac.uk/pysr](https://ai.damtp.cam.ac.uk/pysr). Repo: [github.com/astroautomata/PySR](https://github.com/astroautomata/PySR).


## [2.0.0-beta.4](https://github.com/astroautomata/PySR/compare/v2.0.0-beta.3...v2.0.0-beta.4) (2026-08-24)


### Features

* support guesses with custom types ([#1316](https://github.com/astroautomata/PySR/issues/1316)) ([6115158](https://github.com/astroautomata/PySR/commit/61151582bb21a58eefbccca9c979ce1fbe85f3fc))

## [2.0.0-beta.3](https://github.com/astroautomata/PySR/compare/v2.0.0-beta.2...v2.0.0-beta.3) (2026-08-22)


### Bug Fixes

* preserve custom JAX mappings in checkpoints ([#1199](https://github.com/astroautomata/PySR/issues/1199)) ([0d78783](https://github.com/astroautomata/PySR/commit/0d78783b8aca3c3e42e8001397ee3a5a819dc6dd))


### Dependencies

* raise jax, ipython, ipykernel, and pytest-cov ceilings ([#1307](https://github.com/astroautomata/PySR/issues/1307)) ([3f37488](https://github.com/astroautomata/PySR/commit/3f37488da78d735df9bee37bf3a23bf01f98ee1c))
* raise juliacall ceiling to 0.9.36 ([#1312](https://github.com/astroautomata/PySR/issues/1312)) ([494798e](https://github.com/astroautomata/PySR/commit/494798e76511aed9db1f57078c6ec8366c6e06fc))


### Documentation

* add PDE discovery example and skill guidance ([#1311](https://github.com/astroautomata/PySR/issues/1311)) ([459a720](https://github.com/astroautomata/PySR/commit/459a720bd72cb0bd02e96ba7b877460d4a2739c2))
* add PySR v1 to v2 migration guide ([#1302](https://github.com/astroautomata/PySR/issues/1302)) ([b0bb321](https://github.com/astroautomata/PySR/commit/b0bb321451fc8a489c711a5a7c37bff03fbb50c9))
* replace python feature card ([#1303](https://github.com/astroautomata/PySR/issues/1303)) ([b99a314](https://github.com/astroautomata/PySR/commit/b99a314da625d3bce5a9dc25e7618a8681a8ffc4))

## [2.0.0-beta.2](https://github.com/astroautomata/PySR/compare/v2.0.0-beta.1...v2.0.0-beta.2) (2026-08-18)


### ⚠ BREAKING CHANGES

* the deprecated positional and `function_symbols=...` forms of `TemplateExpressionSpec` are removed; pass explicit `combine=`, `expressions=`, and `variable_names=` keywords.
* checkpoints written before this change (schema 2, from v2.0.0-beta.1 and earlier betas) fail to load with an explicit schema error rather than restoring incomplete state.

### Features

* accept sample_weight as an alias for weights in fit ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))
* require keyword arguments for TemplateExpressionSpec ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))
* support custom value types via TypeSpec ([#1280](https://github.com/astroautomata/PySR/issues/1280)) ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))


### Bug Fixes

* avoid duplicate PyPI attestations ([#1292](https://github.com/astroautomata/PySR/issues/1292)) ([71242e8](https://github.com/astroautomata/PySR/commit/71242e8420e90db7be4f8ff064ab3dc9df2c52d2))
* bump the checkpoint schema to version 3 ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))
* rebuild Julia-backed equation columns after unpickling ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))


### Documentation

* rewrite the examples for TypeSpec and template expressions ([e2a159b](https://github.com/astroautomata/PySR/commit/e2a159b615b8258c192608cccde90a19fcc31563))

## [2.0.0-beta.1](https://github.com/astroautomata/PySR/compare/v2.0.0a2...v2.0.0-beta.1) (2026-08-11)


### ⚠ BREAKING CHANGES

* enable annealing by default ([#1283](https://github.com/astroautomata/PySR/issues/1283))
* remove ParametricExpressionSpec ([#1277](https://github.com/astroautomata/PySR/issues/1277))
* switch to SlurmClusterManager.jl for slurm allocations ([#794](https://github.com/astroautomata/PySR/issues/794))

### Features

* expose new plugin interface + upgrade to v2.0.0-beta.3 ([#1282](https://github.com/astroautomata/PySR/issues/1282)) ([e20c880](https://github.com/astroautomata/PySR/commit/e20c88015cc4294ea41d52cc2e9345bed7f1ebac))
* remove ParametricExpressionSpec ([#1277](https://github.com/astroautomata/PySR/issues/1277)) ([d5f0bb0](https://github.com/astroautomata/PySR/commit/d5f0bb0b4b1e5d13463ba988c3ff15fba00bfe13))
* set precompile_float64=false preference for SymbolicRegression ([#1279](https://github.com/astroautomata/PySR/issues/1279)) ([b89f920](https://github.com/astroautomata/PySR/commit/b89f9209d8ead59974bcff8f0f295b71c4a8fb7c))
* switch to SlurmClusterManager.jl for slurm allocations ([#794](https://github.com/astroautomata/PySR/issues/794)) ([49f44a4](https://github.com/astroautomata/PySR/commit/49f44a420c3c08c4406c8c9685ba9d34d7773b23))


### Bug Fixes

* convert num_features dict keys to Julia Symbols ([#1209](https://github.com/astroautomata/PySR/issues/1209)) ([8aa59b8](https://github.com/astroautomata/PySR/commit/8aa59b82bfbe29daba59e38c2c063d8184c9dd0b)), closes [#811](https://github.com/astroautomata/PySR/issues/811)
* enable annealing by default ([#1283](https://github.com/astroautomata/PySR/issues/1283)) ([f4dc86b](https://github.com/astroautomata/PySR/commit/f4dc86b21df97724f0e5efc8f9bc4ce34b8814d4))
* respect tempdir for temporary equation files ([#1207](https://github.com/astroautomata/PySR/issues/1207)) ([beaa405](https://github.com/astroautomata/PySR/commit/beaa4053a1352789176b1b3bae356007dcbebabd))


### Documentation

* add agent skill for using PySR effectively ([#1264](https://github.com/astroautomata/PySR/issues/1264)) ([fdedcc8](https://github.com/astroautomata/PySR/commit/fdedcc892db4ae2fed289601718074ec45a596d0))
* add angular coefficients paper ([da6d2d2](https://github.com/astroautomata/PySR/commit/da6d2d27f782156e5378ecd0255474cad5dc684d))
* add biomass pyrolysis paper ([ccb473e](https://github.com/astroautomata/PySR/commit/ccb473ec715f353e32729f6921352a572f6ca71f))
* add dark energy symbolic regression paper ([da9ad80](https://github.com/astroautomata/PySR/commit/da9ad80bc0187ca65f12b331d919c322264ea874))
* add human mobility models paper ([f47c4d2](https://github.com/astroautomata/PySR/commit/f47c4d27496659ea97cc72ab9cfe138964c3ea53))
* add microbial growth models paper ([f7c72fb](https://github.com/astroautomata/PySR/commit/f7c72fbc3ba13ed6e7e5f69f946c3c57d0a2d755))
* add paper showcase entries ([9283914](https://github.com/astroautomata/PySR/commit/9283914523b52e5e992f4238d7ca6692de7825d1))
* add s-stars chaos paper ([cfc4907](https://github.com/astroautomata/PySR/commit/cfc490762cac17ea248cfb83555c261779fa61e9))
* add skin friction estimation paper ([8ec43f0](https://github.com/astroautomata/PySR/commit/8ec43f082e2b449d7dece4cb5ba2372915b81a7c))
* add yawed wind turbines paper ([e1dc986](https://github.com/astroautomata/PySR/commit/e1dc986ef096b3c0c14e7f7f4429aee6b730f429))
* update contributors list ([#1286](https://github.com/astroautomata/PySR/issues/1286)) ([10b3637](https://github.com/astroautomata/PySR/commit/10b36376b2866e09e8382669df20e4a7a1539ec5))
* use Float32 literals in custom loss example ([#1276](https://github.com/astroautomata/PySR/issues/1276)) ([2bd7db2](https://github.com/astroautomata/PySR/commit/2bd7db238a70773c9b4882259b940f3dec8c8591))

## [2.0.0a2](https://github.com/MilesCranmer/PySR/compare/v2.0.0a1...v2.0.0a2) (2026-05-15)

This is an alpha release of v2.0.0. It includes backend, packaging, export, and documentation updates since `v2.0.0a1`.

### Backend and Packaging

* update backend to SymbolicRegression.jl v2.0.0-alpha.9 ([#1132](https://github.com/MilesCranmer/PySR/pull/1132))
* allow pandas `<4.0.0` ([#1129](https://github.com/MilesCranmer/PySR/pull/1129))
* repair PySR v2 release automation without relying on release-please to publish PEP 440 alpha versions ([#1162](https://github.com/MilesCranmer/PySR/pull/1162), [#1169](https://github.com/MilesCranmer/PySR/pull/1169), [#1185](https://github.com/MilesCranmer/PySR/pull/1185))

### Features

* add Slurm tests using docker compose ([0364d43](https://github.com/MilesCranmer/PySR/commit/0364d43c01ba058784e9e8eaab357f927895ce1e))
* raise friendly error when loss functions have bad signatures ([#1138](https://github.com/MilesCranmer/PySR/pull/1138))

### Bug Fixes

* correct type in elementwise loss validation ([#1184](https://github.com/MilesCranmer/PySR/pull/1184))
* normalize DataFrame column spaces in `predict` ([#1136](https://github.com/MilesCranmer/PySR/pull/1136))
* fix torch export with constant arguments ([bb721b5](https://github.com/MilesCranmer/PySR/commit/bb721b5687248165f4cdbb08807498185947ae4c), [e280034](https://github.com/MilesCranmer/PySR/commit/e280034231acb03b146e6a0333e02d3fc38acebc))
* fix typing issue in export format ([2b173ff](https://github.com/MilesCranmer/PySR/commit/2b173ffa1fbe8ff6a07ab223f1132343a2b4c324))

### Documentation

* overhaul documentation with VitePress and add language/version picker
* add example papers for active matter, Lyman-alpha forest analysis, batteries, implied volatility, and math discovery

## [2.0.0a1] (2025-10-08)

This is an _alpha_ release of v2.0.0. There will still be changes before the release of v2.0.0, likely including new hyperparameter defaults.

### What's Changed

#### Major changes

#### Multiple features (update backend to 2.0) ([#999](https://github.com/MilesCranmer/PySR/pull/999))

This PR updates the backend to SymbolicRegression.jl 2.0.0-alpha.8 and exposes several major new features:

- **N-ary operators**: Support for operators with arbitrary arity (not just unary/binary)
  - Added 3-arity operators: `fma` (fused multiply-add), `clamp`, etc.
  - This can be used via a new `operators` parameter dictionary: `operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp"]}`
- **Equation guesses**: Pass initial equation guesses to guide the search using the `guesses` parameter to `fit`
  - For example: `guesses=["sin(x0 * 2.1 - 0.5)", "x0 * 3.0 + x2"]` provides two guesses
  - Control injection rate with `fraction_replaced_guesses`
- **Advanced autodiff backends**: Experimental support for Mooncake.jl and Enzyme.jl
  - Enzyme.jl support via `autodiff_backend="Enzyme"` (fragile/experimental)
  - Mooncake.jl (experimental - currently disabled pending upstream fix)
- **Feature node mutation**: New mutation operator that directly modifies which features are used
  - Control mutation weight with `weight_mutate_feature`
- **Worker management**:
  - `worker_imports`: specify Julia packages to import on workers
  - `worker_timeout`: control timeout for worker processes

#### **Automatic batching for big data** ([#1045](https://github.com/MilesCranmer/PySR/pull/1045))

#### Other changes

* docs: add vector expression example by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/1041
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/MilesCranmer/PySR/pull/1008
* chore(deps): bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1009
* ci: update Dockerfile image by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/1022
* test: update docker versions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/1023
* chore(deps): bump actions/setup-python from 5 to 6 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1027
* deps: update min python to 3.9 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/1052
* deps: bump juliacall requirement by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1035
* chore(deps): bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1049
* chore(deps): bump github/codeql-action from 3 to 4 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1050
* chore(deps): bump actions/setup-python from 5 to 6 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1051
* chore(deps): update beartype requirement from <0.22,>=0.19 to >=0.19,<0.23 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/1047
* chore: update pyjuliacall requirement in environment.yml by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/1054

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.9...v2.0.0a1

## [1.5.9] (2025-07-15)

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/MilesCranmer/PySR/pull/853
* Fix type error in feature selection code by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/952
* chore(deps): update juliacall requirement from <0.9.26,>=0.9.24 to >=0.9.24,<0.9.27 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/980


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.8...v1.5.9

## [1.5.8] (2025-05-20)

### What's Changed
* fix: compat with python 3.8 by removing beartype by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/935
* ci: update workflows to test 3.13 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/929
* style: fix newline in warning by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/931
* ci: switch to codecov by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/932
* deps: fix local conda env versions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/933


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.7...v1.5.8

## [1.5.7] (2025-05-19)

### What's Changed
* Enable negative losses by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/928
* Recommend TemplateExpressionSpec over ParametricExpressionSpec @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/920
* Fix multi-output template expressions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/921
* build: switch to hatchling by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/888
* chore(deps): bump juliacall from 0.9.24 to 0.9.25 by @dependabot in https://github.com/MilesCranmer/PySR/pull/925
* fix: turn off double warning for ParametricExpressionSpec by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/930


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.6...v1.5.7

## [1.5.6] (2025-05-04)

### What's Changed
* Added paper contribution and image by @manuel-morales-a in https://github.com/MilesCranmer/PySR/pull/824
* fix: pickling of inv by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/910
* Automated update to backend: v1.10.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/890

### New Contributors
* @manuel-morales-a made their first contribution in https://github.com/MilesCranmer/PySR/pull/824

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.5...v1.5.6

## [1.5.5] (2025-04-02)

### What's Changed
* fix: typing extensions dependency by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/885


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.4...v1.5.5

## [1.5.4] (2025-04-01)

### What's Changed
* Compat with older Python by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/884


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.3...v1.5.4

## [1.5.3] (2025-03-28)

### What's Changed
* fix: change sympy mappings ordering by @romanovzky in https://github.com/MilesCranmer/PySR/pull/868

### New Contributors
* @romanovzky made their first contribution in https://github.com/MilesCranmer/PySR/pull/868

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.2...v1.5.3

## [1.5.2] (2025-03-05)

### What's Changed
* fix: mapping of cbrt by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/858


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.1...v1.5.2

## [1.5.1] (2025-03-01)

### What's Changed
* fix: comparison operator parsing by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/845


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.0...v1.5.1

## [1.5.0] (2025-02-25)

### Backend Changes

#### Major Changes

* Change behavior of batching to resample only every iteration; not every eval in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/421
  * This result in a speed improvement for code with `batching=true`
  * It should also result in improved search results with batching, because comparison within a single population is more stable during evolution. In other words, there is no _lucky batch_ phenomenon.
  * This also refactors the batching interface to be cleaner. There is a `SubDataset <: Dataset` rather than passing around an array `idx` explicitly.
  * Note that other than the slight behaviour change, this is otherwise backwards compatible - the old way to write custom loss functions that take `idx` will still be handled.

#### Other changes

* feat: better error for mismatched eltypes by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/414
* CompatHelper: bump compat for Optim to 1, (keep existing compat) by @github-actions in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/403
* feat: explicitly monitor errors in workers by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/417
* feat: allow recording crossovers by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/415
* add script for converting record to graphml by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/416
* ci: redistribute part 1 of test suite by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/424
* refactor: rename to `.cost` by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/423
* fix: batched dataset for optimisation by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/426
* refactor: task local storage instead of thread local by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/427

### Frontend Changes

* Update backend to v1.8.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/833
* test: update deprecated sklearn test syntax by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/834
* chore(deps): bump juliacall from 0.9.23 to 0.9.24 by @dependabot in https://github.com/MilesCranmer/PySR/pull/815
* use standard library logging by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/835
* Remove warning about many features, as not really relevant anymore by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/837
* chore(deps): update beartype requirement from <0.20,>=0.19 to >=0.19,<0.21 by @dependabot in https://github.com/MilesCranmer/PySR/pull/838
* chore(deps): update jax[cpu] requirement from <0.5,>=0.4 to >=0.4,<0.6 by @dependabot in https://github.com/MilesCranmer/PySR/pull/810


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.4.0...v1.5.0

## [1.4.0] (2025-02-13)

### What's Changed

[#823](https://github.com/MilesCranmer/PySR/pull/823) adds support for _parameters in template expressions_, allowing you to learn expressions under a template, that have custom coefficients which can be optimized.

Along with this, the `TemplateExpressionSpec` API has changed. (The old API will continue to function, but will not have parametric expressions available).

```python
spec = TemplateExpressionSpec(
    "fx = f(x); p[1] + p[2] * fx + p[3] * fx^2",
    expressions=["f"],
    variable_names=["x"],
    parameters={"p": 3},
)
```

This would learn three parameters, for the expression $y = p_1 + p_2 f(x) + p_3 f(x)^2.$

You can have multiple parameter vectors, and these parameter vectors can also be indexed by categorical features. For example:

```python
### Learn different parameters for each class:
spec = TemplateExpressionSpec(
    "p1[category] * f(x1, x2) + p2[1] * g(x1^2)",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": 3, "p2": 1},
)
```

This will learn an equation of the form:
$$y = \alpha_c\,f(x_1,x_2) + \beta g(x_1 ^2)$$
where $c$ is the category, $\alpha_c$ is a learned parameter specific to each category, and $\beta$ is a normal scalar category. Note that **unlike ParametricExpressionSpec**, this feature of TemplateExpressionSpec would have you pass the `category` variable _in_ `X` rather than as a category keyword (floating point versions of the categories). This difference means that in a TemplateExpressionSpec, you can actually have _multiple_ categories!

* Added support for expression-level loss functions via `loss_function_expression`, which allows you to specify custom loss functions that operate on the full expression object rather than just its evaluated output. This is particularly useful when working with template expressions.

* Note that the old template expression syntax using function-style definitions is deprecated. Use the new, cleaner syntax instead:

```python
### # Old:
### spec = TemplateExpressionSpec(
###     function_symbols=["f", "g"],
###     combine="((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)"
### )

### New:
spec = TemplateExpressionSpec(
    "sin(f(x1, x2)) + g(x3)"
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
```


**Full Changelog:** [v1.3.1...v1.4.0](https://github.com/MilesCranmer/PySR/compare/v1.3.1...v1.4.0)

## [1.3.1] (2024-12-27)

### What's Changed
* Automated update to backend: v1.5.1 by @github-actions in https://github.com/MilesCranmer/PySR/pull/790


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.3.0...v1.3.1

## [1.3.0] (2024-12-15)

### What's Changed

- Expanded support for differential operators via backend 1.5.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/782

e.g., say we wish to integrate $\frac{1}{x^2 \sqrt{x^2 - 1}}$ for $x > 1$:

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

x = np.random.uniform(1, 10, (1000,))  # Integrand sampling points
y = 1 / (x**2 * np.sqrt(x**2 - 1))     # Evaluation of the integrand

expression_spec = TemplateExpressionSpec(
    ["f"], "((; f), (x,)) -> D(f, 1)(x)"
)

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    expression_spec=expression_spec,
    maxsize=20,
)
model.fit(x[:, np.newaxis], y)
```

which should correctly find $\frac{\sqrt{x^2 - 1}}{x}$.


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.2.0...v1.3.0

## [1.2.0] (2024-12-14)

### What's Changed
* Compatibility with new scikit-learn API and test suite by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/776
* Add differential operators and input stream specification by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/780
  * (Note: the differential operators aren't yet in a stable state, and are not yet documented. However, they do work!)
  * This PR also adds various GC allocation improvements in the backend.

**Frontend Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.1.0...v1.2.0

**Backend Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v1.2.0...v1.4.0

## [1.1.0] (2024-12-09)

### What's Changed
* Automated update to backend: v1.2.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/770


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.2...v1.1.0

## [1.0.2] (2024-12-07)

### What's Changed
* logger fixes: close streams and persist during warm start by @BrotherHa in https://github.com/MilesCranmer/PySR/pull/763
* Let sympy use log2(x) instead of log(x)/log(2) by @nerai in https://github.com/MilesCranmer/PySR/pull/712

### New Contributors
* @BrotherHa made their first contribution in https://github.com/MilesCranmer/PySR/pull/763
* @nerai made their first contribution in https://github.com/MilesCranmer/PySR/pull/712

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.1...v1.0.2

## [1.0.1] (2024-12-06)

### What's Changed
* Automated update to backend: v1.1.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/762
* Fall back to `eager` registry when needed by @DilumAluthge in https://github.com/MilesCranmer/PySR/pull/765

### New Contributors
* @DilumAluthge made their first contribution in https://github.com/MilesCranmer/PySR/pull/765

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.0...v1.0.1

## [1.0.0] (2024-12-01)

### PySR v1.0.0 Release Notes

PySR 1.0.0 introduces new features for imposing specific functional forms and finding parametric expressions. It also includes TensorBoard support, along with significant updates to the core algorithm, including some important bug fixes. The default hyperparameters have also been updated based on extensive tuning, with a maxsize of 30 rather than 20.

### Major New Features

#### Expression Specifications

PySR 1.0.0 introduces new ways to specify the structure of equations through "Expression Specifications", that expose the new backend feature of `AbstractExpression`:

#### Template Expressions
`TemplateExpressionSpec` allows you to define a specific structure for your equations. For example:

```python
expression_spec = TemplateExpressionSpec(["f", "g"], "((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)")
```

#### Parametric Expressions
`ParametricExpressionSpec` enables fitting expressions that can adapt to different categories of data with per-category parameters:

```python
expression_spec = ParametricExpressionSpec(max_parameters=2)
model = PySRRegressor(
    expression_spec=expression_spec
    binary_operators=["+", "*", "-", "/"],
)
model.fit(X, y, category=category)  # Pass category labels
```

#### Improved Logging with TensorBoard

The new `TensorBoardLoggerSpec` enables logging of the search process, as well as hyperparameter recording, which exposes the `AbstractSRLogger` feature of the backend:

```python
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)
model = PySRRegressor(logger_spec=logger_spec)
```

Features logged include:

- Loss curves over time at each complexity level
- Population statistics
- Pareto "volume" logging (measures performance over all complexities with a single scalar)
- The min loss over time

### Algorithm Improvements

#### Updated Default Parameters

The default hyperparameters have been significantly revised based on testing:

- Increased default `maxsize` from 20 to 30, as I noticed that many people use the defaults, and this maxsize would allow for more accurate expressions.
- New mutation operator weights optimized for better performance, along the new mutation "rotate tree."
- Improved search parameters tuned using Pareto front volume calculations.
- Default `niterations` increased from 40 to 100, also to support better accuracy (at the expense of slightly longer default search times).

#### Core Changes

- New output organization: Results are now stored in `outputs/<run_id>/` rather than in the directory of execution.
- Improved performance with better parallelism handling
- Support for Python 3.10+
- Updated Julia backend to version 1.10+
- Fix for aliasing issues in crossover operations

### Breaking Changes

- Minimum Python version is now 3.10, and minimum Julia version is 1.10
- Output file structure has changed to use directories
- Parameter name updates:
  - `equation_file` → `output_directory` + `run_id`
  - Added clearer naming for parallelism options, such as `parallelism="serial"` rather than the old `multithreading=False, procs=0` which was unclear

### Documentation

The documentation has a new home at https://ai.damtp.cam.ac.uk/pysr/

## [0.19.4] (2024-08-23)

### What's Changed
* Create `load_all_packages` to install Julia extensions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/688
* Apptainer definition file for PySR by @wkharold in https://github.com/MilesCranmer/PySR/pull/687
* JuliaCall 0.9.23 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/703
    * build(deps): bump juliacall from 0.9.21 to 0.9.22 by @dependabot in https://github.com/MilesCranmer/PySR/pull/695

### New Contributors
* @wkharold made their first contribution in https://github.com/MilesCranmer/PySR/pull/687

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.19.3...v0.19.4

## [0.19.3] (2024-07-29)

### What's Changed
* build(deps): bump juliacall from 0.9.20 to 0.9.21 by @dependabot in https://github.com/MilesCranmer/PySR/pull/678


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.19.2...v0.19.3

## [0.19.2] (2024-07-15)

### What's Changed
* Avoid automatic upgrade to Julia 1.11 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/671


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.19.1...v0.19.2

## [0.19.1] (2024-07-15)

### What's Changed
* Bump docker/setup-qemu-action from 2 to 3 by @dependabot in https://github.com/MilesCranmer/PySR/pull/506
* fix: `from pysr import *` by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/670


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.19.0...v0.19.1

## [0.19.0] (2024-06-22)

### What's Changed
* BREAKING: Disable automatic sympy simplification by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/658
* Build: update numpy version by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/650
* Build: bump docker/build-push-action from 5 to 6 by @dependabot in https://github.com/MilesCranmer/PySR/pull/652


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.5...v0.19.0

## [0.18.5] (2024-06-16)

### What's Changed

#### New features

* Per-variable custom complexities by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/649

    ```python
    model.fit(X, y, complexity_of_variables=[1, 3])
    # run a search with feature 1 having complexity 1 and feature 2 with complexity 3
    ```

* Automatically suggest similar parameters by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/620

#### Other

* Bump julia-actions/cache from 1 to 2 by @dependabot in https://github.com/MilesCranmer/PySR/pull/621
* Update pysr_demo.ipynb by @VishalJ99 in https://github.com/MilesCranmer/PySR/pull/624
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/612
* Bump docker/login-action from 2 to 3 by @dependabot in https://github.com/MilesCranmer/PySR/pull/509
* More extensive typing stubs and associated refactoring by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/609

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.4...v0.18.5

### Backend changes

#### New features

- Allow per-variable complexity (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/324) (@MilesCranmer)

#### Other

- ci: split up test suite into multiple runners (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/311) (@MilesCranmer)
- chore(deps): bump julia-actions/cache from 1 to 2 (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/315) (https://github.com/dependabot[bot])
- CompatHelper: bump compat for DynamicQuantities to 0.14, (keep existing compat) (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/317) (@github-actions[bot])
- Use DispatchDoctor.jl to wrap entire package with `@stable` (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/321) (@MilesCranmer)
- CompatHelper: bump compat for MLJModelInterface to 1, (keep existing compat) (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/322) (@github-actions[bot])
- Mark more functions as stable (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/323) (@MilesCranmer)
- Refactor tests to use TestItems.jl (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/325) (@MilesCranmer)

**Full Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.24.4...v0.24.5

### New Contributors
* @VishalJ99 made their first contribution in https://github.com/MilesCranmer/PySR/pull/624

## [0.18.4] (2024-05-04)

### Frontend changes
* Add dimensionless constants mode; update Python version constraints; upgrade juliacall to 0.9.20 (https://github.com/MilesCranmer/PySR/pull/608) (@MilesCranmer)
* Fix sign typo in example docs (https://github.com/MilesCranmer/PySR/pull/611) (@hvaara)


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.3...v0.18.4

### Backend changes

- Up to 40% speedup for default settings via more parallelism inside workers (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/304) (@MilesCranmer)
- feat: use `?` for wildcard units instead of `⋅` (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/307) (@MilesCranmer)
- refactor: fix some more type instabilities (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/308) (@MilesCranmer)
- refactor: remove unused Tricks dependency (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/309) (@MilesCranmer)
- Add option to force dimensionless constants (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/310) (@MilesCranmer)

**Full Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.24.2...v0.24.4

### New Contributors
* @hvaara made their first contribution in https://github.com/MilesCranmer/PySR/pull/611

## [0.18.3] (2024-04-26)

### Frontend changes

* Automated update to backend: v0.24.3 by @github-actions in https://github.com/MilesCranmer/PySR/pull/605

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.2...v0.18.3

### Backend changes

**Full Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.24.1...v0.24.2

## [0.18.2] (2024-04-15)

### Frontend changes

* Add missing `greater` operator in sympy mapping by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/590
* Bump julia-actions/setup-julia from 1 to 2 by @dependabot in https://github.com/MilesCranmer/PySR/pull/591
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/537
* Automated update to backend: v0.24.2 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/598

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.1...v0.18.2

### Backend changes

**Merged pull requests:**
- Bump julia-actions/setup-julia from 1 to 2 (MilesCranmer/SymbolicRegression.jl#300) (@dependabot[bot])
- [pre-commit.ci] pre-commit autoupdate (MilesCranmer/SymbolicRegression.jl#301) (@pre-commit-ci[bot])
- A small update on examples.md for 1-based indexing (MilesCranmer/SymbolicRegression.jl#302) (@liuyxpp)
- Fixes for Julia 1.11 (MilesCranmer/SymbolicRegression.jl#303) (@MilesCranmer)

**Closed issues:**
- API Overhaul (MilesCranmer/SymbolicRegression.jl#187)
- [Feature]: Training on high dimensions X  (MilesCranmer/SymbolicRegression.jl#299)

**Full Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.24.1...v0.24.2

## [0.18.1] (2024-03-26)

### What's Changed
* Revert GitHub-based registry for backend by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/587


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.18.0...v0.18.1

## [0.18.0] (2024-03-24)

### Frontend changes
* fix TypeError when a variable name matches a builtin python function by @tomjelen in https://github.com/MilesCranmer/PySR/pull/558
* Update to backend: v0.24.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/564
* Fix extensions not being added to package env by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/579
* Bump backend version and switch to GitHub-based registry by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/580

### Backend changes

_Filtered to only include relevant ones for Python frontend. Also note that not all backend features, like graph-based expressions/program synthesis, are supported yet, so I don't mention those changes yet._

- (BREAKING) The `swap_operands` mutation contributed by @foxtran now has a default weight of 0.1 rather than 0.0.
- (BREAKING) The Dataset struct has had many of its field declared immutable, as a safety precaution.
    - If you had relied on the mutability of the struct to set parameters after initializing it, or had changed any properties of the dataset within a loss function (which actually would break assumptions outside the loss function anyways), you will need to modify your code. Note you can always copy fields of the dataset to variables and then modify those variables
- LoopVectorization.jl has been moved to a package extension. PySR will install automatically at first use of `turbo=True` rather than by default, which means faster install time and startup time.
    - Note that LoopVectorization will no longer result in improved performance in Julia 1.11 and thus `turbo=True` will have no effect on that version (due to internal changes in Julia), which is why I have instead done the following:
- Bumper.jl support added. Passing `bumper=true` to `PySRRegressor()` will result in faster performance.
    - Uses bump allocation (see rust package [bumpalo](https://docs.rs/bumpalo/latest/bumpalo) for a good explanation) in the expression evaluation which can get speeds equivalent to LoopVectorization and sometimes even better due to better management of allocations rather than relying on garbage collection. Seems like a pretty good alternative, and doesn't rely on manipulating Julia internals for performance (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/287)
- Various fixes to distributed compute; confirmed Slurm support again!
    - Maybe from https://github.com/MilesCranmer/SymbolicRegression.jl/pull/297 - ensures ClusterManagers.jl is loaded on workers
- Now prefer to use new keyword-based constructors for nodes:

    ```julia
    Node{T}(feature=...)        # leaf referencing a particular feature column
    Node{T}(val=...)            # constant value leaf
    Node{T}(op=1, l=x1)         # operator unary node, using the 1st unary operator
    Node{T}(op=1, l=x1, r=1.5)  # binary unary node, using the 1st binary operator
    ```
    rather than the previous constructors Node(op, l, r) and Node(T; val=...) (though those will still work; just with a depwarn). If you did any construction of nodes manually, note the new syntax. (Old syntax will still work though)
- Formatting overhaul of backend (https://github.com/MilesCranmer/SymbolicRegression.jl/pull/278)
- Upgraded Optim to 1.9
- Upgraded DynamicQuantities to 0.13
- Upgraded DynamicExpressions to 0.16
- The main search loop in the backend has been greatly refactored for readability and improved type inference. It now looks like this (down from a monolithic ~1000 line function)
    ```julia
    function _equation_search(
        datasets::Vector{D}, ropt::RuntimeOptions, options::Options, saved_state
    ) where {D<:Dataset}
        _validate_options(datasets, ropt, options)
        state = _create_workers(datasets, ropt, options)
        _initialize_search!(state, datasets, ropt, options, saved_state)
        _warmup_search!(state, datasets, ropt, options)
        _main_search_loop!(state, datasets, ropt, options)
        _tear_down!(state, ropt, options)
        return _format_output(state, ropt)
    end
    ```


**Backend changes**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.23.1...v0.24.1

### New Contributors
* @tomjelen made their first contribution in https://github.com/MilesCranmer/PySR/pull/558

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.17.4...v0.18.0

## [0.17.4] (2024-03-21)

Small patch to Julia version to avoid buggy libgomp in 1.10.1 and 1.10.2.

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.17.3...v0.17.4

## [0.17.3] (2024-03-20)

### What's Changed
* Bump juliacall from 0.9.15 to 0.9.19 by @dependabot in https://github.com/MilesCranmer/PySR/pull/569
  * Upstreamed patching of `seval` to support multiple expressions
* remove repeated operator by @RaulPL in https://github.com/MilesCranmer/PySR/pull/573

### New Contributors
* @RaulPL made their first contribution in https://github.com/MilesCranmer/PySR/pull/573

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.17.2...v0.17.3

## [0.17.2] (2024-03-12)

### What's Changed
* All cell state in bio image paper by @chris-soelistyo in https://github.com/MilesCranmer/PySR/pull/560
* Refactor update_backend.yml workflow by @sefffal in https://github.com/MilesCranmer/PySR/pull/562
* Limit to Julia 1.6.7-1.10.0 and 1.10.3+ by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/565

### New Contributors
* @chris-soelistyo made their first contribution in https://github.com/MilesCranmer/PySR/pull/560
* @sefffal made their first contribution in https://github.com/MilesCranmer/PySR/pull/562

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.17.1...v0.17.2

## [0.17.1] (2024-02-13)

### What's Changed
* Fix y_units bug by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/545


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.17.0...v0.17.1

## [0.17.0] (2024-02-12)

### What's Changed
* Bump docker/build-push-action from 3 to 5 by @dependabot in https://github.com/MilesCranmer/PySR/pull/510
* Bump actions/cache from 3 to 4 by @dependabot in https://github.com/MilesCranmer/PySR/pull/526
* Update colab notebook to use juliaup by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/531
* Bump peter-evans/create-pull-request from 5 to 6 by @dependabot in https://github.com/MilesCranmer/PySR/pull/539
* (BREAKING) Rewrite Julia interface with PyJulia -> JuliaCall; other changes by @MilesCranmer @cjdoris @mkitti in https://github.com/MilesCranmer/PySR/pull/535

#### Detailed changes from #535

- (BREAKING) Changed PyJulia with JuliaCall
  - Need to change `eval` -> `seval`
  - Manually converting to `Vector` when calling SymbolicRegression.jl functions (otherwise would get passed as `PyList{Any}`; see https://github.com/JuliaPy/PythonCall.jl/issues/441)
  - Wrapped `equation_search` code with `jl.PythonCall.GC.disable()` to avoid multithreading-related segfaults (https://github.com/JuliaPy/PythonCall.jl/issues/298)
  - Manually convert `np.str_` to `str` before passing to `variable_names`, otherwise it becomes a `PyArray` and not a `String` (might be worth adding a workaround, it seems like PyJulia does this automatically)
- (BREAKING) Julia is now installed automatically when you import `pysr` (via JuliaCall)
- (BREAKING) The user no longer needs to run `python -m pysr install`. The install process is done by JuliaCall at import time.
  - Removed code related to `pysr.install()` and `python -m pysr install` because JuliaCall now handles this.
  - `python -m pysr install` will not give a warning and do nothing.
- (BREAKING) Remove the feynman problems dataset. Didn't seem good to have a dataset within a library itself.
- (BREAKING) Deprecated `julia_project` argument (ignored; no effect). The user now needs to set this up by customizing `juliapkg.json`. See updated documentation for instructions.
- (BREAKING) Switch from `python -m pysr.test [test]` to `python -m pysr test [test]`.
- Switches to `pyproject.toml` for building rather than `setup.py`. However, `setup.py install` should still work.
- Dependencies are now managed by pyjuliapkg rather than the custom code we made. Simplifies things a lot!
- Rather than storing the raw julia variables in `PySRRegressor`, I am now storing a serialized version of them. This means you can now pickle the search state and warm-start the search from a file, in another Python process!
  - Not breaking! Because `self.raw_julia_state_` will deserialize it automatically for you
- SymbolicRegression is now available to import from PySR:

```python
from pysr import SymbolicRegression as SR
x1 = SR.Node(feature=1)  # Create expressions manually
```

- SymbolicRegression options are accessible in `<model>.julia_options_` (generated from a serialized format for pickle safety) so that the user can call a variety of functions in `SymbolicRegression.jl` directly.
- Deprecated various kwargs to match SymbolicRegression.jl (old names will still work, so this is not breaking):
  - `ncyclesperiteration => ncycles_per_iteration`
  - `loss => elementwise_loss`
  - `full_objective => loss_function`
- Fixes Jupyter printing by automatically loading the `juliacall.ipython` extension at import time
- Adds Zygote.jl to environment by default
- Does unittesting on an example Jupyter notebook


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.9...v0.17.0

## [0.16.9] (2024-01-05)

### What's Changed
* Swap operands mutation by @foxtran in https://github.com/MilesCranmer/PySR/pull/512

### New Contributors
* @foxtran made their first contribution in https://github.com/MilesCranmer/PySR/pull/512

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.8...v0.16.9

## [0.16.8] (2023-12-31)

### What's Changed
* Install `typing_extensions` for compatibility with Python 3.7 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/497
* Create dependabot.yml by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/500
* Fix docker CI nightly by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/499
* Enforce upper bound compats by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/498


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.7...v0.16.8

## [0.16.7] (2023-12-31)

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/495
* Warn the user on Python 3.12 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/496


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.6...v0.16.7

## [0.16.6] (2023-12-24)

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/488
* Add parameter for specifying `--heap-size-hint` on spawned Julia processes by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/493


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.5...v0.16.6

## [0.16.5] (2023-12-14)

### What's Changed
* Add more piecewise operators by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/486


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.4...v0.16.5

## [0.16.4] (2023-12-13)

### What's Changed
* Requesting addition of paper to research examples by @tmengel in https://github.com/MilesCranmer/PySR/pull/415
* Incorporate pre-commit hooks by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/425
* Refactor sympy and export functionality by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/427
* Refactor utility functions in `sr.py` by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/428
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/431
* Add paper "Discovery of a Planar Black Hole Mass Scaling Relation for Spiral Galaxies" by @ZehaoJin in https://github.com/MilesCranmer/PySR/pull/437
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/440
* Added "min" and "max" sympy mapping by @tanweer-mahdi in https://github.com/MilesCranmer/PySR/pull/473
* Added "round" operator in the Sympy mappings by @tanweer-mahdi in https://github.com/MilesCranmer/PySR/pull/474
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/MilesCranmer/PySR/pull/446
* Automated update to backend: v0.22.5 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/482

### New Contributors
* @tmengel made their first contribution in https://github.com/MilesCranmer/PySR/pull/415
* @ZehaoJin made their first contribution in https://github.com/MilesCranmer/PySR/pull/437
* @tanweer-mahdi made their first contribution in https://github.com/MilesCranmer/PySR/pull/473

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.3...v0.16.4

## [0.16.3] (2023-08-21)

### What's Changed
* Automated update to backend: v0.22.4 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/413
  * Fixes world age issue


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.2...v0.16.3

## [0.16.2] (2023-08-17)

### What's Changed
* Automated update to backend: v0.22.3 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/409

### Backend changes

- CompatHelper: bump compat for DynamicExpressions to 0.13, (keep existing compat) by @github-actions in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/250
- Fix type stability of deterministic mode by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/251
- Faster random sampling of nodes by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/252
- Faster copying of MutationWeights by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/253
- Hotfix for breaking change in Optim.jl by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/256

**Backend changes**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.22.2...v0.22.3

**Frontend changes**: https://github.com/MilesCranmer/PySR/compare/v0.16.1...v0.16.2

## [0.16.1] (2023-08-10)

### What's Changed
* Automated update to backend: v0.22.2 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/404


### Backend changes

- Expand aqua test suite (MilesCranmer/SymbolicRegression.jl#246) (@MilesCranmer)
- Return more descriptive errors for poorly defined operators (MilesCranmer/SymbolicRegression.jl#247) (@MilesCranmer)

**Backend Changelog**: [Diff since v0.22.1](https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.22.1...v0.22.2)
**PySR Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.16.0...v0.16.1

## [0.16.0] (2023-08-07)

### What's Changed
* Backend version update in https://github.com/MilesCranmer/PySR/pull/400. Includes:
  * Algorithmic improvements to batching
  * Code quality improvements (some method ambiguities, old exports)

### Backend changes

* (**Algorithm modification**) Evaluate on fixed batch when building per-population hall of fame in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/243
  * This only affects searches that use `batching=true`. It results in improved searches on large datasets, as the "winning expression" is not biased towards an expression that landed on a lucky batch.
  * Note that this only occurs within an iteration. Evaluation on the entire dataset still happens at the end of an iteration and those loss measurements are used for absolute comparison between expressions.
* (**Algorithm modification**) Deprecates the `fast_cycle` feature in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/243. Use of this parameter will have no effect.
  * Was removed to ease maintenance burden and because it doesn't have a use. This feature was created early on in development as a way to get parallelism within a population. It is no longer useful as you can parallelize across populations.
* Add Aqua.jl to test suite in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/245 for code quality control
* CompatHelper: bump compat for DynamicExpressions to 0.12, (keep existing compat) in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/242
  * Is able to avoids method invalidations when using operators to construct expressions manually by modifying a global constant mapping of operator => index, rather than `@eval`-ing new operators.
  * This only matters if you were using operators to build trees, like `x1 + x2`. All internal search code uses `Node()` explicitly to build expressions, so did not rely on method invalidation at any point.


**Backend Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.21.5...v0.22.1

**PySR Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.15.4...v0.16.0

## [0.15.4] (2023-08-04)

### What's Changed
* Warn user when using power laws by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/399
  * This seems like the most common configuration mistake in PySR: using the `^` operator without setting `constraints`, leading to extremely complex expressions with poor generalization properties. Thus, this warning will let the user know about it if they set up `^` without constraints.


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.15.3...v0.15.4

## [0.15.3] (2023-08-02)

### What's Changed
* Use unicode in printing without needing to decode by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/398


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.15.2...v0.15.3

## [0.15.2] (2023-08-01)

### What's Changed
* Ensure files are read as utf-8 on all operating systems by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/396


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.15.1...v0.15.2

## [0.15.1] (2023-07-30)

### What's Changed
* Fix compat with old scikit-learn versions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/393


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.15.0...v0.15.1

## [0.15.0] (2023-07-28)

### What's Changed

* Backend version update in https://github.com/MilesCranmer/PySR/pull/389. Includes:
  * Dimensional analysis (see docs examples page)
  * Printing improvements
  * Many misc changes (see below)

### Backend Changes

* https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228 and https://github.com/MilesCranmer/SymbolicRegression.jl/pull/230 and https://github.com/MilesCranmer/SymbolicRegression.jl/pull/231 and https://github.com/MilesCranmer/SymbolicRegression.jl/pull/235
    - **Dimensional analysis** ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
        - Allows you to (softly) constrain discovered expressions to those that respect physical dimensions
        - Specify `X_units` and `y_units` (see https://astroautomata.com/PySR/examples/#10-dimensional-constraints)
    - **Printing improvements** ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
      - By default, only 5 significant digits are now printed, rather than the entire float. You can change this with the `print_precision` option.
      - In the default printed equations, `x₁` is used rather than `x1`.
      - `y = ` is printed at the start (or `y₁ = ` for multi-output). With units this becomes, for example, `y[kg] =`.
    - **Misc**
      - Easier to convert from MLJ interface to SymbolicUtils (via `node_to_symbolic(::Node, ::AbstractSRRegressor)`) ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
      - Improved precompilation ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
      - Various performance and type stability improvements ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
      - Inlined the recording option to speedup compilation ([230](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/230))
      - Updated Julia tutorials to use MLJ rather than low-level interface ([228](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/228))
      - Moved JSON3.jl to extension ([231](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/231))
      - Use PackageExtensionsCompat.jl over Requires.jl ([231](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/231))
      - Require LossFunctions.jl to be 0.10 ([231](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/231))
      - Batching inside optimization loop + batching support for custom objectives by ([235](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/235))
      - Update docker defaults: Julia=1.9.1; Python=3.10.11 in https://github.com/MilesCranmer/PySR/pull/371

**Backend Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.20.0...v0.21.0

**PySR Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.14.3...v0.15.0

## [0.14.3] (2023-07-04)

### What's Changed
* Self-repairing PyCall installation to lower entrance barrier for new users by @MilesCranmer and @mkitti in https://github.com/MilesCranmer/PySR/pull/363

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.14.2...v0.14.3

## [0.14.2] (2023-06-20)

### What's Changed
* Recommend user install with `--enable-shared` by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/352
* Automated update to backend: v0.19.1 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/355

### Backend

[Diff since v0.19.0](https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.19.0...v0.19.1)

**Merged pull requests on backend:**
- CompatHelper: bump compat for StatsBase to 0.34, (keep existing compat) (MilesCranmer/SymbolicRegression.jl#202) (@github-actions[bot])
- (Soft deprecation) change `varMap` to `variable_names` (MilesCranmer/SymbolicRegression.jl#219) (@MilesCranmer)
- (Soft deprecation) rename `EquationSearch` to `equation_search` (MilesCranmer/SymbolicRegression.jl#222) (@MilesCranmer)
- Fix equation splitting for unicode variables (MilesCranmer/SymbolicRegression.jl#223) (@MilesCranmer)


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.14.1...v0.14.2

## [0.14.1] (2023-05-28)

### What's Changed
* Automated update to backend: v0.19.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/340
  * ~30% faster startup time on first search (https://github.com/MilesCranmer/SymbolicRegression.jl/releases/tag/v0.19.0)
* Let user know when compilation is taking place by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/341


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.14.0...v0.14.1

## [0.14.0] (2023-05-20)

### What's Changed
* Added CLI to run pysr.install() to install Julia dependencies by @w2ll2am in https://github.com/MilesCranmer/PySR/pull/298
  * Let's you install PySR with `python -m pysr install` rather than `python -c 'import pysr; pysr.install()'`
  * This CLI also has other options available (precompilation, Julia project name, etc.)

### New Contributors
* @w2ll2am made their first contribution in https://github.com/MilesCranmer/PySR/pull/298

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.13.0...v0.14.0

## [0.13.0] (2023-05-12)

### What's Changed
* Test Julia 1.9 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/329
* Automated update to backend: v0.18.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/331


### Backend changes

[Diff since v0.17.1](https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v0.17.1...v0.18.0)


- Overload ^ if user passes explicitly (MilesCranmer/SymbolicRegression.jl#201) (@MilesCranmer)
- Upgrade DynamicExpressions to 0.8; LossFunctions to 0.10 (MilesCranmer/SymbolicRegression.jl#206) (@github-actions[bot])
- Show expressions evaluated per second (MilesCranmer/SymbolicRegression.jl#209) (@MilesCranmer)
- Cache complexity of expressions whenever possible (MilesCranmer/SymbolicRegression.jl#210) (@MilesCranmer)


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.12.3...v0.13.0

## [0.12.3] (2023-04-27)

### What's Changed
* Highlight contributors by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/301
* Automated update to backend: v0.17.1 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/320


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.12.2...v0.12.3

## [0.12.2] (2023-04-22)

### What's Changed
* Add paper 'Electron Transfer Rules of Minerals under Pressure…' by @GCaptainNemo in https://github.com/MilesCranmer/PySR/pull/288
* Fix colab notebook example by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/295
* Add paper: "Data-Driven Equation Discovery of a Cloud Cover Parameterization" by @agrundner24 in https://github.com/MilesCranmer/PySR/pull/302
* Pass through `enable_autodiff` parameter by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/316

### New Contributors
* @GCaptainNemo made their first contribution in https://github.com/MilesCranmer/PySR/pull/288
* @agrundner24 made their first contribution in https://github.com/MilesCranmer/PySR/pull/302

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.12.1...v0.12.2

## [0.12.1] (2023-03-25)

### What's Changed
* Allow user to specify full objective functions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/276


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.12.0...v0.12.1

## [0.12.0] (2023-03-22)

### What's Changed
* Complex-valued expressions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/281
* Various fixes in backend (see https://github.com/MilesCranmer/SymbolicRegression.jl/releases/tag/v0.16.0)


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.17...v0.12.0

## [0.11.17] (2023-03-07)

### What's Changed
* Update backend version with warm start fix by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/271
    * This means that you can change the dataset or loss function, and `warm_start=True` will still work, and the losses will be re-computed.

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.16...v0.11.17

## [0.11.16] (2023-03-01)

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.15...v0.11.16

## [0.11.15] (2023-02-18)

### What's Changed
* Bump backend version with data race fix by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/268
  * Incorporates depth check into constraints, rather than in mutation step.
  * Fixes one instance of a data race (appears to be remaining issues, however)


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.14...v0.11.15

## [0.11.14] (2023-02-13)

### What's Changed
* Update backend with constraints fix by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/265


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.13...v0.11.14

## [0.11.13] (2023-02-09)

### What's Changed
* Fix latex_table assertion for multi-output by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/253
* Make precompilation optional by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/263


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.12...v0.11.13

## [0.11.12] (2023-01-16)

### What's Changed
* Make docker build multi-stage by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/235
* Create interactive API reference page by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/247
* Bump backend version with stream fix; fixes #250 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/252


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.11...v0.11.12

## [0.11.11] (2022-11-22)

### What's Changed
* Make Julia startup options configurable; set optimize=3 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/228


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.10...v0.11.11

## [0.11.10] (2022-11-21)

### What's Changed
* Clean up dockerfile by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/223
* Update backend version with improved resource monitoring by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/227


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.9...v0.11.10

## [0.11.9] (2022-11-05)

### What's Changed
* Refactor testing suite to have CLI by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/221


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.8...v0.11.9

## [0.11.8] (2022-11-04)

### What's Changed
* Fix PyCall not giving traceback by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/218
* Fixed safe operators; make progress bar print to stderr by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/219


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.7...v0.11.8

## [0.11.7] (2022-11-04)

### What's Changed
* Expand nightly conda-forge tests to other Python versions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/212
* Clean up parameter groupings in docs by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/214
* Add optimization-as-mutation, and adaptive parsimony by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/217


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.6...v0.11.7

## [0.11.6] (2022-10-31)

### What's Changed
* Speed up evaluation with `turbo` parameter by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/208

https://user-images.githubusercontent.com/7593028/199054602-7ad19e87-19ff-4440-aa09-da6d7b6175d5.mp4

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.5...v0.11.6

## [0.11.5] (2022-10-24)

### What's Changed
* 30-50% Faster evaluation, and perform explicit version assertion for backend by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/205


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.4...v0.11.5

## [0.11.4] (2022-10-10)

### What's Changed
* Fix conda forge installs by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/202


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.3...v0.11.4

## [0.11.3] (2022-10-06)

### What's Changed

- Faster evaluation for constant sub-expressions ([SymbolicRegression.jl#129](https://github.com/MilesCranmer/SymbolicRegression.jl/pull/129))
- Will now check variable names for spaces and other non-alphanumeric characters, aside from underscores. Before this would only raise an issue after a search, when trying to pickle the saved data.


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.2...v0.11.3

## [0.11.2] (2022-09-28)

(Fix for conda-forge build)

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.1...v0.11.2

## [0.11.1-1] (2022-09-26)

### What's Changed
* Added [Customization](https://astroautomata.com/PySR/backend/) page in the docs for tweaking the backend's loss function and constraints.
* Adding two entries to papers.yml by @JayWadekar in https://github.com/MilesCranmer/PySR/pull/192
* Explicitly deprecate Julia <= 1.5 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/194
* Allow custom shared projects for `julia_project` by @MilesCranmer @mkitti in https://github.com/MilesCranmer/PySR/pull/197
  * e.g., this would allow you to run with `@my-project` and it will set up a shared Julia project under `my-project` (in the environments dir)


### New Contributors
* @JayWadekar made their first contribution in https://github.com/MilesCranmer/PySR/pull/192

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.11.0...v0.11.1-1

## [0.11.0] (2022-09-11)

### What's Changed
* Update backend https://github.com/MilesCranmer/PySR/pull/191
  * Includes high-precision constants when `precision=64`
  * Enables datasets with zero variance (to allow fitting a constant)
  * Changes, e.g., `abs(x)^y` to `x^y`, with expressions avoided altogether for invalid input. This is because the former would sometimes give weird functional forms by exploiting the cusp at `x=0`. Thanks to @johanbluecreek.

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.10.4...v0.11.0

## [0.10.4-1] (2022-09-08)

### What's Changed
* Fix install for Julia <=1.6 by @MilesCranmer @mkitti  in https://github.com/MilesCranmer/PySR/pull/188
  * PyJulia will now launch directly into the shared `pysr-{version}` environment, rather than activating it later.

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.10.3...v0.10.4

## [0.10.3] (2022-09-06)

### What's Changed
* Displays a warning message when PyTorch is imported *before* PyJulia starts. See https://github.com/pytorch/pytorch/issues/78829. The only current solution is to start Julia beforehand.
* New [docs](https://astroautomata.com/PySR/)! Using Material-Mkdocs:
<img width="1445" alt="Screen Shot 2022-09-06 at 6 06 49 PM" src="https://user-images.githubusercontent.com/7593028/188748940-e6e0262b-3567-4819-9169-efecc174c59c.png">

## [0.10.2] (2022-09-06)

### What's Changed
* Set JULIA_PROJECT, use Pkg.add once by @mkitti in https://github.com/MilesCranmer/PySR/pull/186


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.10.1...v0.10.2

## [0.10.1] (2022-09-06)

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.10.0...v0.10.1

## [0.10.0] (2022-08-14)

### What's Changed

* Easy loading from auto-generated checkpoint files by @MilesCranmer w/ review @tttc3 @Pablo-Lemos in https://github.com/MilesCranmer/PySR/pull/167
  * Use `.from_file` to load from the auto-generated `.pkl` file.
* LaTeX table generator by @MilesCranmer w/ review @tttc3 @kazewong in https://github.com/MilesCranmer/PySR/pull/156
  * Generate a LaTeX table of discovered equations with `.latex_table()`
* Improved default model selection strategy by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/177
  * Old strategy is available as `model_selection="score"`
* Add opencontainers image-spec to `Dockerfile` by @SauravMaheshkar w/ review @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/166
* Switch to comma-based csv format by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/176

### Bug fixes

* Fixed conversions to torch and JAX when a rational number appears in the sympy expression (https://github.com/MilesCranmer/PySR/commit/17c9b1a1762efbd8e021d275491f75cc6dcea8f1, https://github.com/MilesCranmer/PySR/commit/f119733698e4517e34cc902c78dcb95d450c0c80)
* Fixed pickle saving when trained with multi-output (https://github.com/MilesCranmer/PySR/commit/3da0df512ee295f446ceb0ae6e2c39fb0e380618)
* Fixed pickle saving when using custom operators with defined sympy -> jax/torch/numpy mappings
* Backend fix avoids use of Julia's `cp` which is buggy for some file systems (e.g., EOS)

### New Contributors
* @SauravMaheshkar made their first contribution in https://github.com/MilesCranmer/PySR/pull/166

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.9.0...v0.10.0

## [0.9.0] (2022-06-04)

### What's Changed
* Refactor of PySRRegressor by @tttc3 in https://github.com/MilesCranmer/PySR/pull/146
  * PySRRegressor is now completely compatible with scikit-learn.
  * PySRRegressor can be stored in a pickle file, even after fitting, and then be reloaded and used with `.predict()`
  * `PySRRegressor.equations` -> `PySRRegressor.equations_`

### New Contributors
* @tttc3 made their first contribution in https://github.com/MilesCranmer/PySR/pull/146

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.8.7...v0.9.0

## [0.8.5] (2022-05-20)

### What's Changed
* Custom complexities for operators, constants, and variables (https://github.com/MilesCranmer/PySR/pull/138)
* Early stopping conditions (https://github.com/MilesCranmer/PySR/pull/134)
  * Based on a certain loss value being achieved
  * Max number of evaluations (for theoretical studies of genetic algorithms, rather than anything practical).
* Work with specified expression rather than the one given by `model_selection`, by passing `index` to the function you wish to use (e.g,. `model.predict(X, index=5)` would use the 5th equation.).

**Full Changelog since v0.8.1**: https://github.com/MilesCranmer/PySR/compare/v0.8.1...v0.8.5

## [0.8.1] (2022-05-08)

### What's Changed
* Enable distributed processing with ClusterManagers.jl from https://github.com/MilesCranmer/PySR/pull/133


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v0.8.0...v0.8.1

## [0.8.0] (2022-05-08)

This new release updates the entire set of default PySR parameters according to the ones presented in https://github.com/MilesCranmer/PySR/discussions/115. These parameters have been tuned over nearly 71,000 trials. See the discussion for further info.

Additional changes:

- Nested constraints implemented. For example, you can now prevent `sin` and `cos` from being repeatedly nested, by using the argument: `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}`. This argument states that within a `sin` operator, you can only have a max depth of 0 for other `sin` or `cos`. The same is done for `cos`. The argument `nested_constraints={"^": {"+": 2, "*": 1, "^": 0}}` states that within a pow operator, you can only have 2 things added, or 1 use of multiplication (i.e., no double products), and zero other pow operators. This helps a lot with finding interpretable expressions!
- New parsimony algorithm (backend change). This seems to help searches quite a bit, especially when one is searching for more complex expressions. This is turned on by `use_frequency_in_tournament` which is now the default.
- Many backend improvements: speed, bug fixes, etc.
- Improved stability of multi-processing (backend change). Thanks to @CharFox1.
- Auto-differentiation implemented (backend change). This isn't used by default in any instances right now, but could be used by optimization later. Thanks to @kazewong.
- Improved testing coverage of weird edge cases.
- All parameters to PySRRegressor have been cleaned up to be in snake_case rather than CamelCase. The backend is also now almost entirely snake_case for internal functions. +Other readability improvements. Thanks to @bstollnitz and @patrick-kidger for the suggestions.

## [0.6.0] (2021-06-01)

PySR Version 0.6.0

Large changes:

- Exports to JAX, PyTorch, NumPy. All exports have a similar interface. JAX and PyTorch allow the equation parameters to be trained (e.g., as part of some differentiable model). Read https://pysr.readthedocs.io/en/latest/docs/options/#callable-exports-numpy-pytorch-jax for details. Thanks Patrick Kidger for the PyTorch export.
- Multi-output `y` input is allowed, and the backend will efficiently batch over each output. A list of dataframes is returned by pysr for these cases. All `best_*` functions return a list as well.
- BFGS optimizer introduced + more stable parameter search due to back tracking line search.

Smaller changes since 0.5.16:

- Expanded tests, coverage calculation for PySR
- Improved (pre-processing) feature selection with random forest
- New default parameters for search:
  - annealing=False (no annealing works better with the new code. This is equivalent to alpha=infinity)
  - useFrequency=True (deals with complexity in a smarter way)
  - npopulations = 20 ~~procs*4~~
  - progress=True (show a progress bar)
  - optimizer_algorithm="BFGS"
  - optimizer_iterations=10
  - optimize_probability=1
  - binary_operators default = ["+", "-", "/", "*"]
  - unary_operators default = []
- Warnings:
  - Using maxsize > 40 will trigger a warning mentioning how it will be slow and use a lot of memory. Will mention to turn off `useFrequency`, and perhaps also use `warmupMaxsizeBy`.
- Deprecated nrestarts -> optimizer_nrestarts
- Printing fixed in Jupyter

## [0.4.0] (2021-02-01)

With versions v0.4.0/v0.4.0, SymbolicRegression.jl and PySR have now been completely disentangled: PySR is 100% Python code (with some Julia meta-programming), and SymbolicRegression.jl is 100% Julia code.

PySR now works by activating a Julia env that has SymbolicRegression.jl as a dependency, and making calls to it! By default it will set up a Julia project inside the pip install location, and install requirements at the user's confirmation, though you can pass an arbitrary project directory as well (e.g., if you want to use PySR but also tweak the backend). The nice thing about this is that for Python users, all you need to do is install a Julia binary somewhere, and they should be good to go. And for Julia users, you never need to touch the Python side.

The SymbolicRegression.jl backend also sets up workers automatically & internally now, so one never needs to call `@everywhere` when setting things up. The same is true even with locally-defined functions - these get passed to workers!

With PySR importing the latest Julia code, this also means it gets new simplification routines powered by SymbolicUtils.jl, which seem to help improve the equations discovered.

## [0.3.8] (2020-09-27)

Populations don't block eachother, which gives a large speedup especially for large numbers of populations. This was fixed by using RemoteChannel() in Julia.

Some populations happen to take longer than others - perhaps they have very complex equations - and can therefore block others that have finished early. This lets the processor work on the next population to be finished.

## [0.3.5] (2020-09-27)

Uses equation from Cranmer et al. (2020) https://arxiv.org/abs/2006.11287 to score equations, and prints this alongside MSE. This makes symbolic regression more robust to noise.

## [0.2] (2020-09-21)
