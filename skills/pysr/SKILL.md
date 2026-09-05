---
name: pysr
description: Use when fitting equations to data with PySR or SymbolicRegression.jl, when a user wants an interpretable formula, symbolic model, scaling law, or empirical relation discovered from numeric data, or when debugging a PySR search that is slow, stuck, or giving poor equations.
---

# Using PySR Effectively

PySR discovers symbolic expressions (readable equations) that fit data, using an evolutionary search over expression trees with a Julia backend (SymbolicRegression.jl). You get a Pareto front of equations trading accuracy against complexity, not a single black-box model.

This guide is distilled from the PySR documentation and several hundred real user threads, checked against PySR 2.0.0. Full docs: https://ai.damtp.cam.ac.uk/pysr/

## Quick start

```python
# pip install pysr  (Julia is downloaded automatically on first import; no separate install)
import numpy as np
from pysr import PySRRegressor

X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

model = PySRRegressor(
    operators={1: ["cos"], 2: ["+", "-", "*", "/"]},
    niterations=100,
)
model.fit(X, y)
print(model)            # Pareto front: complexity, loss, equation
model.predict(X)        # uses the auto-selected "best" equation
model.sympy()           # SymPy expression
model.latex()           # LaTeX string
```

`model.equations_` is a pandas DataFrame with `complexity`, `loss`, `score`, `equation`, `sympy_format`, `lambda_format`. Pass a row index to `predict`, `sympy`, `latex`, `jax`, `pytorch` to use any equation on the front, not just the selected one.

## Critical performance fact: keep the process alive

Julia is JIT-compiled. The first `.fit()` in a Python process pays Julia startup plus compilation (roughly a minute or two); every later fit in the *same process* starts almost instantly. The single most common agent mistake is running each experiment as a fresh `python script.py`, paying the full compile cost every time.

Instead:

- Keep one long-lived Python session (a background IPython/Jupyter kernel, a REPL tool, or any mechanism you already have) and run successive experiments in it. Creating a new `PySRRegressor` per experiment is cheap; the expensive state is the Julia runtime, which lives per-process.
- If you must use scripts, structure iteration so parameter tweaks happen inside one process rather than via repeated relaunches.
- Reuse an existing Python environment that already has `pysr` installed rather than creating a fresh environment per task (a new environment re-resolves and precompiles the Julia packages).
- Do not misdiagnose the first-run compilation pause as a hang. "Compiling Julia backend..." taking a minute or two is normal.

Running plain `python` scripts works fine; this is an optimization, not a requirement.

## Recommended workflow

1. **Subsample the data.** Symbolic regression rarely needs more than a few thousand rows; ~1,000 to 5,000 representative rows often suffice even when millions are available. More rows help with many features, heavy noise, or rare regimes. PySR 2.0 uses automatic batching for larger datasets by default. Fewer rows still mean proportionally faster search.
2. **Choose the minimal operator set.** Only operators plausible for the domain. Redundant operators (e.g. `pow` alongside `square` and `cube`, or `-` alongside `neg`) blow up the search space. Fewer operators is better. If the target is a polynomial, use `["+", "-", "*"]` and skip `/` and `^` entirely.
3. **Start from defaults otherwise.** The default hyperparameters (populations, parsimony, mutation weights, `ncycles_per_iteration`) were tuned by large-scale search in 2024-2025. Do not copy hyperparameter recipes from old forum threads or papers; most predate the retuning.
4. **Short runs first.** Debug the setup with a few-minute run: check operators, loss, and that sensible equations appear. Then do one long run for the real search.
5. **Long final run.** Evolution does not converge in the usual sense; a search that looks stalled can jump to a new expression family hours later. Set `niterations` very large and control the budget with `timeout_in_seconds` (wall clock), `max_evals` (cap on total expression evaluations, for compute-matched comparisons; note `niterations * populations * population_size` is NOT an evaluation count), or set `early_stop_condition` (e.g. `"stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"`, a Julia function string of `(loss, complexity)`). Results stream to `outputs/<run_id>/hall_of_fame.csv`, so you can monitor progress and read partial results at any time. In IPython, pressing `q` then Enter stops the search gracefully.
6. **Inspect the whole Pareto front, not just the auto-selected row.** See "Choosing an equation" below.

`model_selection="best"` picks the equation with highest `score` among those with loss within 1.5x of the best loss; `"accuracy"` picks the lowest-loss row. The `score` column is the negative log-loss slope per unit complexity: a large score marks the "kink" where accuracy jumps for little added complexity. These are heuristics for `predict`'s default; report and examine the full table.

## Choosing an equation

Print `model.equations_` and look at where loss drops sharply as complexity increases; that kink is usually the interesting equation. Evaluate the top few candidates on held-out data (`model.predict(X_test, index=i)`) when overfitting is plausible. Check limiting behavior (x -> 0, x -> inf) against domain expectations. Prefer presenting 2-3 candidates with the tradeoff to silently picking one.

## Losses and weights

Default is MSE. Common expert moves:

- Robust to outliers: `elementwise_loss="L1DistLoss()"` (median-seeking rather than mean-seeking).
- Known per-point uncertainty: `model.fit(X, y, weights=1/sigma**2)`; the built-in losses apply weights automatically. Custom weighted form: `elementwise_loss="myloss(x, y, w) = w * abs(x - y)^2"`.
- Target spans many orders of magnitude: MSE is dominated by the largest values. Use a log-space loss, e.g.

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1e-20) / (abs(target)+1e-20)))
    sign_loss = 10 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end"""
```

- Percentage/relative error: divide by the *target*, never the prediction (dividing by the prediction lets evolution win by sending predictions to infinity), and beware targets near zero.
- Any loss from LossFunctions.jl works as a string: `"HuberLoss(1.0)"`, `"LPDistLoss{3}()"`, etc.
- Binary classification: encode targets as +1/-1 and use a margin loss such as `"L2MarginLoss()"`; pass predictions through a sigmoid yourself afterwards if you need probabilities.
- Known asymptotic or boundary behavior (y -> 0 as x -> inf, exact value at a boundary, a known limit): add a few synthetic data points along the asymptote or at the boundary with very large `weights`. This is the standard trick and usually beats a custom objective. For strict enforcement, numerically estimate the limit inside a `loss_function` and add a graded penalty.
- Losses must be deterministic (results are cached) and non-negative unless `loss_scale="linear"`, which permits negative losses (e.g. log-likelihoods).

`elementwise_loss` receives scalars `(prediction, target)` or `(prediction, target, weight)`. Never sum or broadcast inside it. Objectives needing the whole prediction vector or the expression tree use `loss_function` instead (see below).

## Scaling and feature count

- Normalization is optional. Constants are sampled near N(0,1) and mutated multiplicatively, so wildly scaled features can slow the search, but normalizing inserts nuisance constants into the final equation and hides physical meaning. Prefer natural units; rescale only if the search visibly struggles.
- Up to roughly 10 features: no special handling.
- Tens of features: raise `maxsize`, provide more rows, and let the search select features itself; it is reasonably good at this. Automatic batching handles large datasets. An equation forced to contain all of 30+ features would need `maxsize` well above 100.
- More than ~50 features: the primary fix is smarter features or structure. Engineer aggregate features from domain knowledge, or use a template expression over a sensible decomposition. For structured data (fields, images, sequences, graphs), a naive one-column-per-pixel tabular encoding is usually the wrong move; build physically meaningful features, or train a neural network with the right inductive bias and symbolically distill its components (see arXiv:2006.11287). If there is no smarter representation available, `select_k_features=k` (gradient-boosting pre-selection) is the fallback.
- If the search omits a variable the user expected: PySR only optimizes accuracy and simplicity, so omission means the variable did not reduce loss enough to pay for its complexity. Forcing inclusion requires a custom loss that penalizes its absence.

## PDE discovery (spatiotemporal data)

To discover `u_t = f(u, u_x, u_xx, ...)` from field data `u(x, t)`, turn it into normal regression: every grid point is one sample, the columns are the field and its spatial derivatives, and the target is the time derivative. Unlike a fixed candidate library, PySR finds products like `u*u_x` itself.

- Spatial derivatives: spectral differentiation on periodic clean grids; otherwise Savitzky-Golay, e.g. `savgol_filter(U, 21, 3, deriv=1, delta=dx, axis=-1)`. Plain finite differences are only safe on clean data, and spectral derivatives amplify noise catastrophically.
- Target: central differences of the snapshots. If snapshots are sparse or noisy in time, smooth along the time axis first or sample more finely. Trim stencil margins on non-periodic grids.
- Worth trying: `complexity_of_variables=[1, 2, 3, ...]`, one entry per feature in column order, so `u` costs 1, `u_x` costs 2, `u_xx` costs 3; this biases the front toward low-order terms. Units (`X_units=["m/s", "s^-1", ...]`) similarly prune dimensionally impossible terms.
- Read the whole Pareto front for the elbow, not just the best loss. Fitted constants are approximate under noise; validate by simulating the discovered PDE forward from a held-out initial condition.

## Template expressions: use when structure is known

Plain search is the default. Escalate to `TemplateExpressionSpec` when the user knows structure that a free-form tree search would have to rediscover (or would violate). Users very often under-use this feature; suggest it whenever you see:

- A known outer form: y = sin(f(x1, x2)) + g(x3), a rational function, a known envelope like x*(1-x)*g(x).
- Per-category constants: same formula, different coefficients per class/object/condition (`parameters`).
- Shared subexpressions across several outputs, or coupled multi-output problems.
- Derivative or integral relations (differential operator `D`).
- Hard requirements on which variables may appear where.

```python
from pysr import PySRRegressor, TemplateExpressionSpec

spec = TemplateExpressionSpec(
    combine="sin(f(x1, x2)) + g(x3)^2",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
model = PySRRegressor(expression_spec=spec, operators={2: ["+", "-", "*", "/"]})
model.fit(X, y)
```

With per-category parameters (pass the category as a column of X; Julia indexing is 1-based, so pass `category + 1` if starting from 0):

```python
spec = TemplateExpressionSpec(
    combine="p[class] * f(x1, x2) + q[class]",
    expressions=["f"],
    variable_names=["x1", "x2", "class"],
    parameters={"p": 3, "q": 3},   # 3 categories
)
```

The combine string is arbitrary Julia: multiple statements, reuse of a subexpression (`fx = f(x); fx + fx^2`), evaluating the same f at different arguments (`f(x1) - f(x2)`), derivatives (`df = D(f, 1); df(x)`). Multi-output/vector problems: put the extra targets in X as columns, return the per-row residual from the template, fit against dummy y with `elementwise_loss="(p, t) -> p"` (the template output is then the loss itself).

Template caveats (PySR 2.0):

- No `.sympy()`, `.latex()`, `.jax()`, `.pytorch()` export; `combine` can contain arbitrary Julia, so read component strings from `model.equations_` and reassemble manually. Component arguments print as `#1, #2` (argument positions, since f may be called at different inputs).
- Values inside the combine string are `ValidVector`s: raw data in `.x`, validity flag in `.valid`. Ordinary arithmetic propagates validity automatically; custom manipulations must unwrap and rebuild (`ValidVector(raw, valid)`).
- Write Float32-safe literals in the combine string (`0.5f0`, not `0.5`) or convert explicitly; a bare Float64 literal can break type stability.
- Combining a template with a custom objective requires `loss_function_expression` (not `loss_function`).
- Use `TemplateExpressionSpec` with `parameters` for learnable parameters. The pre-1.4 template API (`function_symbols`, lambda-style combine) is deprecated.

## Custom value types with `TypeSpec`

`TypeSpec` makes expression-tree nodes hold generated Julia structs. It is independent of `expression_spec`, so custom values can also use a fixed structure.

Python values supply fields in declaration order. Here `((1.0, 2.0, 3.0), np.uint32(7))` supplies `triple` and `code`; the inner tuple remains one field. Keep `X` two-dimensional and `y` one-dimensional; assign object cells individually so NumPy does not create extra axes.

```python
import numpy as np
from pysr import PySRRegressor, TypeSpec

value = ((1.0, 2.0, 3.0), np.uint32(7))
X = np.empty((1, 1), dtype=object)
y = np.empty(1, dtype=object)
X[0, 0] = value
y[0] = value

packet = TypeSpec(
    "Packet",
    fields={"triple": "NTuple{3,Float64}", "code": "UInt32"},
    sample="rng -> Packet(ntuple(_ -> randn(rng), 3), rand(rng, UInt32))",
    mutate="""function (rng, value::Packet, temperature::Float64)  # note annotation isn't required; shown for documentation purposes
        triple = ntuple(i -> value.triple[i] + temperature * randn(rng), 3)
        bit = UInt32(1) << rand(rng, 0:31)
        Packet(triple, xor(value.code, bit))
    end""",
    scalar_constants="value -> collect(value.triple)",
    with_scalar_constants=(
        "(value, c) -> Packet((c[1], c[2], c[3]), value.code)"
    ),
    is_valid="value -> all(isfinite, value.triple)",
)

operators = {
    1: [
        "negate_packet(x::Packet)::Packet = "
        "Packet(ntuple(i -> -x.triple[i], 3), x.code)"
    ],
    2: [
        "combine_packets(x::Packet, y::Packet)::Packet = "
        "Packet(ntuple(i -> x.triple[i] + y.triple[i], 3), "
        "xor(x.code, y.code))"
    ],
}
loss = """
packet_loss(p::Packet, y::Packet)::Float64 =
    sum((p.triple[i] - y.triple[i])^2 for i in 1:3) +
    Float64(count_ones(xor(p.code, y.code)))
"""

model = PySRRegressor(
    type_spec=packet,
    operators=operators,
    elementwise_loss=loss,
)
```

`sample(rng) -> Packet` creates constant leaves. `mutate(rng, value, temperature) -> Packet` perturbs continuous fields and evolves the discrete field. `scalar_constants(Packet) -> Vector{Float64}` exposes only optimizable scalars; `with_scalar_constants(Packet, Vector{Float64}) -> Packet` rebuilds the value after optimization and must preserve non-optimized fields. `is_valid(Packet) -> Bool` rejects invalid intermediate values. The extraction and reconstruction hooks must be supplied together.

Optional hooks: `init` is a Julia callable `() -> Packet` for initialization; `string` is `Packet -> AbstractString` for printing; `preamble` is Julia source evaluated before the generated type and hooks; `loss_type` names the concrete `AbstractFloat` returned by a full custom objective. Elementwise losses cannot take `loss_type`; their return type is inferred, so the example annotates `packet_loss` as `Float64` to keep it concrete.

When adapting this pattern, check that Python values match the declared field count and order; each field converts to its Julia type; every searched operator is type-stable and returns `Packet`; and the loss returns one real scalar per pair; reserve `Inf` for invalid evaluations.

## Custom operators

Pass Julia definitions as strings, and always provide the SymPy mapping (with *SymPy* functions, never numpy/scipy, or export and `predict` will break):

```python
model = PySRRegressor(
    operators={
        1: ["inv(x) = 1/x", "gauss(x) = exp(-x^2)"],
        2: ["+", "*"],
    },
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "gauss": lambda x: sympy.exp(-x**2),
    },
)
```

Rules that prevent most operator bugs:

- The operator must accept *any* real input (PySR probes far outside your data range). Return a typed NaN for invalid inputs: `my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)`. Candidates producing NaN anywhere on the data are discarded with infinite loss, which is exactly how domain restrictions should be handled. Built-ins (`sqrt`, `log`, `acosh`, ...) are already protected this way; prefer them over hand-rolled versions.
- Preserve the input type: write constants as `T(2.5)` with a `where {T}` signature, or `2.5f0` for Float32 (the default precision). A bare `2.5` or `0` is Float64/Int64 and breaks Float32 pipelines.
- One or two scalar arguments in, one scalar out. For three or more arguments, use the arity-keyed `operators` dictionary or a template.
- For a function from a Julia package: `from pysr import jl; jl.seval("import Pkg; Pkg.add(\"SpecialFunctions\")"); jl.seval("using SpecialFunctions")`, wrap it safely, then use it as an operator. Anything importable in Julia works.
- Operators with no closed sympy form: map to a `class myop(sympy.Function): pass` placeholder; export then works symbolically, though `predict` needs a numeric mapping (call `model.refresh()` after changing mappings).

## Custom objectives (loss_function)

When the loss needs the whole prediction vector, the expression tree, derivatives, or auxiliary data, write a full Julia objective:

```python
objective = """
function my_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    residuals = prediction .- dataset.y
    return sum(abs2, residuals) / dataset.n
end
"""
model = PySRRegressor(loss_function=objective, operators={2: ["+", "*", "-"]})
```

Hard-won rules from the issue tracker:

- **Always check the `completed` flag** from `eval_tree_array` before using predictions; on failure the array contains garbage.
- Return `L(Inf)` for invalid evaluations, but use *graded finite penalties* for structural preferences (e.g. `L(1e6 * n_violations)`), so evolution gets a gradient toward compliance. All-or-nothing `Inf` for structure makes the target unreachable, because intermediate mutations must survive.
- `dataset.X` is features x samples (transposed relative to Python!). `dataset.n` is the number of samples. `dataset.y` is a vector.
- With automatic batching, a three-argument objective receives the active-batch dataset. Use `(tree, full_dataset, options, idx)` when the objective needs the full dataset and selected row indices.
- Performance: this function runs millions of times. Keep it type-stable (no untyped globals; make globals `const` or interpolate values into the string), no printing, no Python callbacks (the GIL serializes everything), vectorize, and prefer `Distances.jl`/standard packages. Diagnose with `@code_warntype` or BenchmarkTools in a separate Julia session. Do not name it `eval_loss` (collides with an internal function).
- Auxiliary data (extra targets, derivative observations, group ids): append as columns of X and slice inside the objective, or interpolate literal arrays into the objective string. Note appended columns are visible to the search as features unless the search cannot use them (template) or you penalize their use.
- Derivatives of candidates: `eval_diff_tree_array` (one feature) / `eval_grad_tree_array` (all features), useful for monotonicity penalties and physics-informed losses. For templates use `D(f, i)` inside the combine string instead.
- With a custom objective that reinterprets the tree (split subtrees, recursion), the *printed* equation is the raw tree, not your interpretation; you must decode it yourself.
- Custom objectives that manipulate the tree symbolically generally break `.sympy()`/`.predict()`; extract what you need manually.

### Symbolic constraints by walking the tree

Any structural rule that `constraints`/`nested_constraints` cannot express (required features, forbidden variable placements, restrictions on constant values, operator-specific rules) is implemented by traversing the expression tree inside the objective. Trees support Julia's standard collection functions: `any(f, tree)`, `all(f, tree)`, `count(f, tree)`, `sum(f, tree)`, `foreach(f, tree)`, `collect(tree)`, and `for node in tree` (depth-first). Prefer the functional forms; `for node in tree` allocates a stack while `count`/`sum`/`any` traverse directly. Node fields:

- `node.degree`: 0 = leaf, 1 = unary, 2 = binary
- `node.l`, `node.r`: children (subtrees, themselves traversable)
- `node.constant` (leaves only): constant vs variable; `node.val`: the constant's value; `node.feature`: 1-based feature index
- `node.op`: 1-based index into the operator list *you passed*, per arity

Worked example: allow `^` only with a lone constant exponent in [0, 1]:

```python
objective = """
function constrained_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    idx_pow = 3   # position of ^ in operators[2] below (1-indexed)
    n_bad = count(tree) do node
        node.degree == 2 && node.op == idx_pow &&
            any(c -> !(c.degree == 0 && c.constant && 0 <= c.val <= 1), node.r)
    end
    n_bad > 0 && return L(10_000 * n_bad)
    prediction, valid = eval_tree_array(tree, dataset.X, options)
    !valid && return L(Inf)
    return sum(i -> abs2(prediction[i] - dataset.y[i]), eachindex(prediction)) / dataset.n
end
"""
model = PySRRegressor(operators={2: ["+", "*", "^"]}, loss_function=objective)
```

The count-then-penalize shape (violations counted, multiplied by a large finite constant, returned *before* evaluation) is the canonical pattern: cheap structural check first, graded penalty so evolution has a direction, `Inf` reserved for failed numerical evaluation. Requiring a feature is the mirror image: `any(n -> n.degree == 0 && !n.constant && n.feature == 2, tree) || return L(big)`. For template expressions, get the tree of a component via `get_tree(ex)` inside `loss_function_expression`. More traversal tools (`tree_mapreduce`, `NodeSampler`, node construction): https://ai.damtp.cam.ac.uk/dynamicexpressions/stable/examples/base_operations/

## Constraints and complexity shaping

- `constraints={"pow": (9, 1)}`: max complexity of each argument (here: any base, exponent must be a lone constant/variable). `-1` means unlimited. Strongly recommended whenever `^` is included; unconstrained exponentiation searches terribly.
- `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}`: forbids nested trig. General form: how many times each inner operator may appear inside each outer one.
- `complexity_of_operators={"exp": 3}`: make disliked operators expensive. `complexity_of_constants=2`: discourage free constants. `complexity_of_variables`: per-feature or global cost.
- These constraints apply to every intermediate expression during evolution, not just final answers; overly tight settings silently make the target unreachable. Leave slack (e.g. want final size 30, set `maxsize=35`).
- `maxsize` counts every node (operator, constant, variable). A 7-feature linear model is already complexity ~29. The default 30 is too small for anything with many terms.
- `warmup_maxsize_by=0.5` (fraction of the run over which maxsize ramps up): useful when the search dives into complex expressions early and gets stuck.
- `parsimony` and `adaptive_parsimony_scaling` are already well-tuned by default in 1.5+; old advice to set them manually is mostly obsolete.
- Known constants (pi, G, ...): just let it fit floats and recognize them afterwards, or pass the constant as an extra constant-valued feature if it must appear exactly.
- Integer-only constants: pass candidate integers as constant features and raise `complexity_of_constants`, or round inside a custom operator; do not expect the continuous optimizer to land on integers.

## Dimensional constraints

Physical units, checked during search:

```python
model.fit(X, y, X_units=["Constants.M_sun", "kg", "m"], y_units="kg * m / s^2")
```

- Uses DynamicQuantities.jl notation; `"1"` means explicitly dimensionless.
- Violations are softly penalized via `dimensional_constraint_penalty` (default 1000). Do not crank it to 1e9; a graded penalty is what lets evolution route through slightly-wrong intermediates.
- Fitted constants get wildcard units (printed `[⋅]` or `[?]`), so a lone constant can absorb any units and make an expression valid. Set `dimensionless_constants_only=True` to forbid that.
- Radians count as dimensionless in SI, so units cannot force a variable to appear only inside trig.

## Parallelism

- Default `parallelism="multithreading"` is right for laptops and single nodes. Thread count is set at Julia startup: set the env var `PYTHON_JULIACALL_THREADS=<n>` *before* importing pysr (`JULIA_NUM_THREADS` is not the right variable under juliacall).
- Keep `populations` at ~2-3x the number of threads/cores so workers always have work (default `populations=31` already covers typical machines).
- If the coordinating thread is saturated on many-core machines, raise `ncycles_per_iteration`; workers then communicate less often.
- `parallelism="multiprocessing"` (with `procs=n`) has much higher startup cost per fit but can run faster steady-state and spans multiple nodes with `cluster_manager="slurm"` (or run SymbolicRegression.jl natively with SlurmClusterManager.jl, see https://ai.damtp.cam.ac.uk/symbolicregression/dev/slurm/). Only worth it for very long runs. Launch the script once, on one node, and let it spawn workers; do not wrap it in `srun`. Custom Julia packages needed on workers go in `worker_imports`.
- Full reproducibility requires `deterministic=True, random_state=<seed>, parallelism="serial"`; parallel seeded runs are not deterministic, and even serial results can differ slightly across CPUs (use `precision=64` to reduce this).

## Saving, resuming, exporting

- By default, each non-temporary fit writes `outputs/<run_id>/hall_of_fame.csv` (updated continuously; safe to read mid-run) and `checkpoint.pkl`. Reload with `PySRRegressor.from_file(run_directory=...)`. Pickles are version-locked: load with the same PySR version that wrote them. The CSV plus your construction code is the durable artifact; custom operators need their `extra_sympy_mappings` supplied again on reload.
- `warm_start=True` continues evolution from the previous call's populations on the next `.fit()` in the same process. Search-space parameters (operators, `maxsize`, `expression_spec`, precision, feature count/order) must stay fixed; you can change the loss or weights between warm-started fits to implement staged objectives.
- Exports: `model.sympy(i)`, `model.latex(i)`, `model.latex_table()`, `model.jax(i)` (returns `{'callable', 'parameters'}`, differentiable), `model.pytorch(i)` (trainable module). Custom operators need `extra_jax_mappings`/`extra_torch_mappings` for those backends. A common pattern: pick an equation, export to JAX/PyTorch, and fine-tune its constants by gradient descent on the full dataset.
- Loss printed by Julia can differ from a Python recomputation (32-bit default vs numpy 64-bit); `precision=64` if it matters. Data with values beyond ~1e19 or below ~1e-19 also needs `precision=64` (Float32 overflow).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First `import pysr` or first `.fit` takes minutes | Normal: Julia download (first install) and JIT compile (each process). Keep the process alive. |
| "Compiling Julia backend" apparently forever | On Apple Silicon: x86 Python under Rosetta; install a native arm64 Python env. In notebooks/embedded shells: stdin monitoring can wedge; pass `input_stream="devnull"`. |
| Hang at startup with a `lock.pid` mentioned | Stale juliapkg lock from a killed process: verify nothing is installing, delete the lock file, `import pysr` once in a fresh process. Shared filesystems: pre-initialize once before launching many jobs. |
| `UnicodeDecodeError` spam in Jupyter | Old PythonCall bug; set `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no` before import, or upgrade. |
| Cannot interrupt search in Jupyter | Known limitation; `q`+Enter works in IPython/terminal, not notebooks. Use `timeout_in_seconds` or run from IPython. |
| Search finds nothing sensible | Loss mismatched to data scale (try log-space loss), operators missing or redundant, `maxsize` too small for the true equation, or constraints exclude it. Check in that order before adding compute. |
| All equations are tiny/trivial | `maxsize` too small, or huge dynamic range with MSE (largest points dominate). |
| `DomainError` from an operator | Custom operator not defined on all reals; add the typed-NaN guard. |
| Fit OK but `.predict`/`.sympy` fails | Missing/wrong `extra_sympy_mappings` (must be sympy functions, not numpy). |
| Equation uses a "forbidden" value like division by ~0 | Any-NaN-anywhere invalidates a candidate, so surviving equations are finite on *your data*; they can still blow up elsewhere in the domain. |
| `ProcessExitedException` wall of text on early stop | Harmless worker teardown noise under multiprocessing. |
| Memory grows across a long run | Mostly fixed in recent Julia; if hit, set `heap_size_hint_in_bytes` (multiprocessing) or upgrade Julia. |
| Results differ run to run | Expected; evolution is stochastic. See determinism recipe under Parallelism. |

## Dropping to Julia

Everything above also exists natively in SymbolicRegression.jl (`SRRegressor` via MLJ), which is preferable when the whole pipeline is Julia or you need deep customization (custom expression types, mutation operators, per-component constraints). From Python, `from pysr import jl` gives the live Julia runtime: `jl.seval(...)` runs arbitrary code, and installed Julia packages can back custom operators and losses. The backend source is readable and small; `src/Options.jl` and `src/CheckConstraints.jl` are the usual extension points, and a dev checkout can be wired in via `pysr/juliapkg.json`.

## Version notes

Written against PySR 2.0.0 (backend SymbolicRegression.jl 2.x). Compared with 1.5.x, 2.0 adds n-ary operators via `operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp"]}`, `guesses` for seeding initial equations, custom value types through `TypeSpec`, automatic batching, and autodiff plugins. Defaults also changed: `batching="auto"`, `batch_size=None`, `annealing=True`, `crossover_probability=0.2`, and a new mutation mix. Start from 2.0 defaults rather than copying 1.x tuning.

Deprecated API you may know or find in old examples; write the current form instead:

| Old | Current |
|---|---|
| `pysr.install()`, `python -m pysr install`, PyCall/`from julia import Main` | none needed; Julia installs automatically on import |
| `pysr(...)`, `best()`, `get_hof()` | `PySRRegressor`, `.fit()`, `model.equations_`, `get_best()` |
| `multithreading=True/False`, `procs=0` | `parallelism="multithreading"/"multiprocessing"/"serial"` |
| `full_objective=` | `loss_function` (`loss_function_expression` for templates) |
| `loss=` | `elementwise_loss` |
| `equation_file=` | `output_directory=` + `run_id=` |
| `npop`, `ncyclesperiteration`, camelCase args | `population_size`, `ncycles_per_iteration`, snake_case |
| positional `TemplateExpressionSpec` or `function_symbols=...` | explicit `combine=`, `expressions=`, and `variable_names=` keywords |
