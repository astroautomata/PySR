---
name: pysr
description: Use when fitting equations to data with PySR or SymbolicRegression.jl, when a user wants an interpretable formula, symbolic model, scaling law, or empirical relation discovered from numeric data, or when debugging a PySR search that is slow, stuck, or giving poor equations.
---

# Using PySR Effectively

PySR evolves expression trees through the Julia backend SymbolicRegression.jl, returning readable equations on an accuracy/complexity Pareto front. This guide distills the documentation and several hundred real user threads. Full docs: https://pysr.ai/

## Quick start

```python
# pip install pysr  (Julia is downloaded automatically on first import; no separate install)
import numpy as np
from pysr import PySRRegressor

X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos"],
    niterations=100,
)
model.fit(X, y)
print(model)            # Pareto front: complexity, loss, equation
model.predict(X)        # uses the auto-selected "best" equation
model.sympy()           # SymPy expression
model.latex()           # LaTeX string
```

`model.equations_` is a pandas DataFrame containing `complexity`, `loss`, `score`, `equation`, `sympy_format`, `lambda_format`. Select any front row by index in `predict`, `sympy`, `latex`, `jax`, `pytorch`; without an index, methods use the auto-selected equation. Indexing lets you inspect and export a specific Pareto candidate instead of only the default row.

## Keep the process alive

The first `.fit()` pays Julia startup/JIT, roughly 1-2 minutes; subsequent same-process fits start almost instantly. Repeated fresh processes repay startup and compilation for every experiment, a common agent failure.

- Keep a long-lived IPython/Jupyter kernel or REPL. New `PySRRegressor` instances are cheap; Julia runtime state lives per-process.
- Scripts work. Iterate parameters inside one process instead of repeatedly relaunching `python script.py`. Process reuse is an optimization rather than a requirement.
- Reuse an environment containing `pysr`; fresh environments re-resolve and precompile Julia packages.
- Expect “Compiling Julia backend...” initially. Avoid treating normal compilation as a hang.

## Candidate equations with `guesses`

Inject knowledge through modifiable/discardable `guesses` or mandatory template structure. Combine both using component/parameter guesses. Constructor strings require Julia syntax, input variable names, and enabled operators:

```python
model = PySRRegressor(
    operators={2: ["+", "*"]},
    guesses=["x0 + 0.5 * x1", "x0 * (x1 + 1.0)"],
)
```

Single output: candidate list. Multiple outputs: lists in target-column order, `guesses=[["x0 + x1"], ["x0 * x1"]]`. Templates: dictionaries keyed by component/parameter names; `#1`, `#2`, etc. identify component arguments.

Guesses enter populations throughout search. `should_optimize_constants=True` optimizes constants/template parameters before insertion. `fraction_replaced_guesses`, default `0.001`, controls the cycle-end replacement fraction.

Guesses are pushed into the hall of fame before generation one. Inspect accepted seeds at their complexity in `model.equations_` or immediate `hall_of_fame.csv`; confirm insertion and optimized constants. Oversize seeds produce only Julia warning `Guess expression '...' has complexity 23 > maxsize (10)`, using default `x1, x2, ...` names regardless of supplied names. They remain unrecorded but still migrate; descendants become recordable after mutations satisfy `maxsize`. Violations of `constraints`, `nested_constraints`, or `maxdepth` disappear silently. Lower-loss candidates replace seeds at the same complexity: only one expression survives per complexity.

Keep constraints compatible and set `maxsize` slightly above the largest seed; use the warning's complexity. Cost rises steeply, PySR warns above 40, and several-times-seed sizes can exhaust the budget without completing one generation despite appearing alive. Use `warmup_maxsize_by` against early bloat.

Set guesses on the estimator, never `.fit()`. Continue with `model.set_params(guesses=[...], warm_start=True)`, replacing guesses while keeping data representation/search space fixed.

## Recommended workflow

1. **Subsample:** usually 1,000-5,000 representative rows suffice even from millions. Many features, heavy noise, or rare regimes benefit from more. PySR 2.0 automatically batches larger datasets; `batching=False` forces full-data search evaluations. Batching still reevaluates hall-of-fame candidates fully. Fewer rows proportionally accelerate search.
2. **Minimize operators:** select domain-plausible operators. Redundant `pow`/`square`/`cube` or `-`/`neg` enlarge search. Polynomials: `["+", "-", "*"]`, omit `/` and `^`.
3. **Use defaults:** populations, parsimony, mutation weights, and `ncycles_per_iteration` were tuned by large-scale search in 2024-2025. Avoid older forum/paper recipes.
4. **Debug briefly:** check operators, loss, and sensible equations in a few-minute run before one long search.
5. **Budget the final run:** evolution lacks conventional convergence and can jump families after hours. Set large `niterations`; control wall-clock with `timeout_in_seconds`, total evaluations with `max_evals`, or stopping with `early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"` (Julia `(loss, complexity)` function). `max_evals` supports compute-matched comparisons; `niterations * populations * population_size` does not count evaluations. Monitor partial `outputs/<run_id>/hall_of_fame.csv`; stop gracefully with `q` then Enter in IPython.
6. **Inspect the entire front:** validate candidates and report tradeoffs.

## Choosing an equation

`model_selection="best"` maximizes `score` within 1.5x minimum loss; `"accuracy"` minimizes loss. Score is negative log-loss slope per complexity unit; a large score marks a sharp accuracy gain for little added complexity. These heuristics control default `predict`.

Print `model.equations_`; inspect sharp loss drops, held-out `model.predict(X_test, index=i)` when overfitting is plausible, and limits x -> 0/inf against domain expectations. Present 2-3 candidates with tradeoffs.

## Losses and weights

Default: MSE.

- MAE (mean absolute error): `elementwise_loss="L1DistLoss()"` works with or without weights. For unweighted data, the equivalent custom loss is `elementwise_loss="my_loss(x, y) = abs(x - y)"`. Useful for outliers, heavy-tailed noise, or costs proportional to absolute error; targets the conditional median rather than the mean.
- Uncertainty: `model.fit(X, y, weights=1/sigma**2)`; built-ins apply weights automatically. Custom: `elementwise_loss="myloss(x, y, w) = w * abs(x - y)^2"`.
- Wide target range: largest values dominate MSE; use log-space loss:

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1e-20) / (abs(target)+1e-20)))
    sign_loss = 10 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end"""
```

- Relative/percentage error: divide by target; prediction denominators reward infinite predictions. Guard near-zero targets.
- LossFunctions.jl strings work: `"HuberLoss(1.0)"`, `"LPDistLoss{3}()"`.
- Binary classification: +1/-1 targets, margin loss `"L2MarginLoss()"`; apply sigmoid yourself for probabilities.
- Asymptotes/boundaries, including y -> 0 as x -> inf, exact boundary values, and known limits: add synthetic points with very large `weights`, usually preferable to custom objectives. Strict enforcement: estimate limits numerically inside `loss_function`, add graded penalties.
- Keep losses deterministic for caching and non-negative; `loss_scale="linear"` permits negatives, including log-likelihoods.

`elementwise_loss` receives scalar `(prediction, target)` or `(prediction, target, weight)`; never sum/broadcast. Whole-vector/tree objectives require `loss_function`.

## Scaling and features

- Normalization is optional: constants sample near N(0,1), mutate multiplicatively; extreme scales slow search. Prefer natural units, avoiding nuisance constants and hidden physical meaning. Rescale visibly struggling searches.
- Roughly 10 features or fewer need no special handling.
- Tens: increase `maxsize` and rows; search selects features reasonably well, automatic batching handles volume. Requiring all 30+ features needs `maxsize` well above 100.
- Above ~50: engineer domain aggregates or template decompositions. Fields/images/sequences/graphs need meaningful features rather than pixel columns, or appropriately biased neural networks followed by symbolic component distillation, arXiv:2006.11287. Fallback: gradient-boosting preselection, `select_k_features=k`.
- Omitted variables failed to justify complexity through accuracy; force inclusion through an absence-penalizing custom loss.

## PDE discovery
Regress `u_t = f(u, u_x, u_xx, ...)` from `u(x, t)`: grid points become samples, field/spatial derivatives become features, time derivative becomes target; PySR discovers products like `u*u_x` without a fixed library.
- Spatial derivatives: spectral differentiation for clean periodic grids, otherwise Savitzky-Golay, `savgol_filter(U, 21, 3, deriv=1, delta=dx, axis=-1)`; finite differences require clean data, spectral derivatives catastrophically amplify noise.
- Target: central snapshot differences; smooth noisy/sparse time samples along time or sample finer. Trim non-periodic stencil margins.
- Bias low-order terms with column-ordered `complexity_of_variables=[1, 2, 3, ...]`, costing `u`, `u_x`, `u_xx` as 1, 2, 3; prune dimensional impossibilities with `X_units=["m/s", "s^-1", ...]`. Inspect the front elbow; noisy coefficients are approximate. Validate forward simulation from held-out initial conditions.
## Template expressions
Default to plain search; suggest `TemplateExpressionSpec` for known structure that free-form trees would rediscover or violate:
- Outer forms: `sin(f(x1, x2)) + g(x3)`, rational functions, envelopes `x*(1-x)*g(x)`.
- Per-category/class/object/condition coefficients through `parameters`; shared subexpressions/coupled outputs; derivative/integral relations through differential operator `D`; hard variable-placement requirements.
```python
from pysr import PySRRegressor, TemplateExpressionSpec
spec = TemplateExpressionSpec(
    combine="sin(f(x1, x2)) + g(x3)^2",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
model = PySRRegressor(expression_spec=spec, binary_operators=["+", "-", "*", "/"])
model.fit(X, y)
```
Pass category as an X column; increment zero-based IDs for Julia's 1-based indexing:
```python
spec = TemplateExpressionSpec(
    combine="p[class] * f(x1, x2) + q[class]",
    expressions=["f"],
    variable_names=["x1", "x2", "class"],
    parameters={"p": 3, "q": 3},
)
model = PySRRegressor(
    expression_spec=spec,
    operators={2: ["+", "*"]},
    guesses=[{
        "f": "#1 + #2",
        "p": [5.0, 10.0, 0.8],
        "q": [0.0, 0.0, 0.0],
    }],
)
```
`parameters={"p": 3, "q": 3}` declares vector lengths; guesses initialize vectors and `f`. Declare disjoint component/parameter names and every guess key; supply every component formula. Match parameter lengths using Python lists/1D NumPy arrays; omit vectors for normal initialization. Values remain learnable, are copied without modifying caller arrays, and use model precision. Parameter guesses require SymbolicRegression.jl ≥2.3.0. `#1`/`#2` denote passed arguments, here `x1`/`x2`; with `combine="f(x, p[1])"`, `#2` denotes `p[1]`.
`combine` accepts arbitrary Julia/multiple statements: reuse `fx = f(x); fx + fx^2`, change arguments `f(x1) - f(x2)`, differentiate `df = D(f, 1); df(x)`. Vector/multi-output problems: append extra targets to X, return per-row residuals, fit dummy y with `elementwise_loss="(p, t) -> p"` so output itself is loss.
PySR 2.0 caveats: arbitrary Julia prevents `.sympy()`, `.latex()`, `.jax()`, `.pytorch()` export; manually reassemble component strings from `model.equations_`. Arguments print positionally as `#1, #2` because components may receive different inputs. Combine values are `ValidVector`s (`.x` data, `.valid` flag); arithmetic propagates validity, custom manipulation must unwrap/rebuild `ValidVector(raw, valid)`. Use Float32-safe `0.5f0` or explicit conversion; bare Float64 `0.5` risks type instability. Custom objectives require `loss_function_expression`. Learnable parameters use `TemplateExpressionSpec(parameters=...)`; pre-1.4 `function_symbols`/lambda-style combine API is removed in 2.0.

## Custom value types: `TypeSpec`
`TypeSpec` generates Julia structs for nodes, independently of `expression_spec`, including fixed structures. Supply fields in declaration order: `triple`, then `code`; the nested tuple remains one field. Keep X 2D/y 1D and assign object cells individually to prevent extra NumPy axes:
```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec, TypeSpec
value = ((1.0, 2.0, 3.0), np.uint32(7))
X = np.empty((1, 1), dtype=object)
y = np.empty(1, dtype=object)
X[0, 0] = value
y[0] = value
packet = TypeSpec(
    "Packet",
    fields={"triple": "NTuple{3,Float64}", "code": "UInt32"},
    sample="rng -> Packet(ntuple(_ -> randn(rng), 3), rand(rng, UInt32))",
    mutate="""function (rng, value::Packet, temperature::Float64)
        triple = ntuple(i -> value.triple[i] + temperature * randn(rng), 3)
        bit = UInt32(1) << rand(rng, 0:31)
        Packet(triple, xor(value.code, bit))
    end""",
    scalar_constants="value -> collect(value.triple)",
    with_scalar_constants="(value, c) -> Packet((c[1], c[2], c[3]), value.code)",
    is_valid="value -> all(isfinite, value.triple)",
)
```
- `sample(rng) -> Packet`: constant leaves; `mutate(rng, value, temperature) -> Packet`: continuous/discrete mutation; annotations are optional.
- Supply `scalar_constants(Packet) -> Vector{Float64}` and `with_scalar_constants(Packet, Vector{Float64}) -> Packet` together; expose optimizable scalars, preserve other fields when rebuilding.
- `is_valid(Packet) -> Bool`: reject invalid intermediates; optional `init`: `() -> Packet`; `string`: `Packet -> AbstractString`.
- `preamble`: Julia before generated type/hooks; `definitions`: Julia immediately after type, including constructors/type-specific methods.
- `loss_type`: concrete `AbstractFloat` for full objectives; unavailable for elementwise losses, whose return type is inferred.
Configure `PySRRegressor(type_spec=packet, operators=operators, elementwise_loss=loss)`. In arity-keyed `operators`, unary `negate_packet(x::Packet)::Packet = Packet(ntuple(i -> -x.triple[i], 3), x.code)` preserves code; binary `combine_packets(x::Packet, y::Packet)::Packet = Packet(ntuple(i -> x.triple[i] + y.triple[i], 3), xor(x.code, y.code))` sums triples/XORs codes. Include binary `choose_parameter(a::Packet, b::Packet) = a` and `choose_parameter(a::Packet, b::ValidVector) = ValidVector(map(_ -> a, b.x), b.valid)`. Define `loss` as `packet_loss(p::Packet, y::Packet)::Float64 = sum((p.triple[i] - y.triple[i])^2 for i in 1:3) + Float64(count_ones(xor(p.code, y.code)))`; annotation keeps return type concrete.
Typed constructors work in `ordinary_guess = ["combine_packets(x0, Packet((1.0, 2.0, 3.0), UInt32(7)))"]` and `template_component_guess = [{"f": "combine_packets(#1, Packet((1.0, 2.0, 3.0), UInt32(7)))"}]`. Parsing uses the generated configuration module with Packet/operators defined; no parser closure/import needed.
Template parameters share the data's `TypeSpec`. Define `packet_template = TemplateExpressionSpec(combine="choose_parameter(p[1], f(x))", expressions=["f"], variable_names=["x"], parameters={"p": 1})`. Preserve one logical Packet with `packet_guess_from_list = [{"f": "#1", "p": [value]}]`, or assign `packet_parameter_array = np.empty(1, dtype=object)` and `packet_parameter_array[0] = value`, then `packet_guess_from_array = [{"f": "#1", "p": packet_parameter_array}]`. Configure the estimator above with additional `expression_spec=packet_template`, `guesses=packet_guess_from_array`.
Verify field count/order, conversion to declared Julia types, type-stable Packet-returning operators, and one real scalar per loss pair; reserve `Inf` for invalid evaluations.

## Custom operators

Supply Julia strings and SymPy mappings using SymPy functions, never numpy/scipy; incorrect mappings break export/prediction:

```python
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x", "gauss(x) = exp(-x^2)"],
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "gauss": lambda x: sympy.exp(-x**2),
    },
)
```

- Accept all reals, including probes beyond data. Guard domains: `my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)`; any data-point NaN discards candidates with infinite loss. Prefer protected built-ins `sqrt`, `log`, `acosh`.
- Preserve types: `T(2.5)` with `where {T}`, or default-Float32 `2.5f0`; bare `2.5`/`0` introduce Float64/Int64 and break Float32 pipelines.
- One/two scalar inputs, scalar output; ≥3 arguments require arity-keyed `operators` or templates.
- Julia packages: `from pysr import jl; jl.seval("import Pkg; Pkg.add(\"SpecialFunctions\")"); jl.seval("using SpecialFunctions")`; safely wrap any importable function.
- No closed SymPy form: map to `class myop(sympy.Function): pass`; symbolic export works, `predict` requires numeric mapping. Call `model.refresh()` after mapping changes.

## Custom objectives: `loss_function`

Use full objectives for vectors, trees, derivatives, auxiliary data:

```python
objective = """
function my_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    residuals = prediction .- dataset.y
    return sum(abs2, residuals) / dataset.n
end
"""
model = PySRRegressor(loss_function=objective, binary_operators=["+", "*", "-"])
```

- Check `completed` before accessing predictions; failure leaves garbage. Return `L(Inf)` for numerical invalidity; use graded finite structural penalties such as `L(1e6 * n_violations)` so intermediate mutations survive. All-or-nothing structural rejection can make targets unreachable.
- `dataset.X`: features × samples, transposed from Python; `dataset.y`: vector; `dataset.n`: sample count. Three-argument objectives receive active batches; `(tree, full_dataset, options, idx)` exposes full data/selected indices.
- Millions of calls: type-stable, vectorized, no printing/Python callbacks (GIL serialization). Use `const` globals or string interpolation, Distances.jl/standard packages; diagnose with `@code_warntype`/BenchmarkTools in separate Julia. Avoid internal-name collision `eval_loss`.
- Append auxiliary targets/derivatives/group IDs to X and slice, or interpolate literal arrays. Appended columns remain searchable unless templates exclude them or penalties discourage use.
- Candidate derivatives: `eval_diff_tree_array` for one feature, `eval_grad_tree_array` for all; support monotonicity/physics-informed losses. Templates use `D(f, i)`.
- Reinterpreted trees, including split subtrees/recursion, print raw equations requiring manual decoding; symbolic manipulation generally breaks `.sympy()`/`.predict()`, requiring manual extraction.

### Structural tree constraints

Before writing a traversal, check whether the rule is already an option. Operator placement rules are `nested_constraints`: "no sin inside cos and no cos inside sin" is `nested_constraints={"sin": {"cos": 0}, "cos": {"sin": 0}}`; "exponent must be a bare constant or variable" is `constraints={"^": (-1, 1)}` (keys are the names used in the operator lists, so `"pow"` if that is the name given). Both reject a violating mutation before it is evaluated, which is cheaper than a loss penalty and never lets a violator into the hall of fame. Use traversal only for what they cannot express: required features, constant value ranges, rules that depend on position or on several operators at once.

Implement those by traversal. Supported: `any(f, tree)`, `all(f, tree)`, `count(f, tree)`, `sum(f, tree)`, `foreach(f, tree)`, `collect(tree)`, depth-first `for node in tree`. Prefer functional traversal: loops allocate stacks; `count`/`sum`/`any` traverse directly.

- `node.degree`: leaf 0/unary 1/binary 2; `node.l`, `node.r`: traversable children.
- Leaves: `node.constant` distinguishes constants/variables; `node.val`: constant value; `node.feature`: 1-based feature index.
- `node.op`: 1-based supplied operator-list index, per arity.

Allow `^` only with a lone constant exponent in [0,1]:

```python
objective = """
function constrained_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    idx_pow = 3   # position of ^ in binary_operators below (1-indexed)
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
model = PySRRegressor(binary_operators=["+", "*", "^"], loss_function=objective)
```

Count violations before evaluation; multiply by a large finite penalty to guide evolution. Require feature 2: `any(n -> n.degree == 0 && !n.constant && n.feature == 2, tree) || return L(big)`. Templates: obtain component trees using `get_tree(ex)` inside `loss_function_expression`. `tree_mapreduce`, `NodeSampler`, construction: https://ai.damtp.cam.ac.uk/dynamicexpressions/stable/examples/base_operations/

## Constraints and complexity

- `constraints={"pow": (9, 1)}` caps argument complexity: base 9, lone constant/variable exponent; `-1` unlimited. Strongly constrain `^`; unrestricted exponentiation searches poorly.
- `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}` prohibits nested trig; values cap inner occurrences within outer operators.
- `complexity_of_operators={"exp": 3}`, `complexity_of_constants=2` discourage operators/constants; `complexity_of_variables` accepts global/per-feature costs.
- Constraints affect every intermediate; excessive tightness silently excludes targets. Allow slack: final size 30, `maxsize=35`.
- `maxsize` counts operator/constant/variable nodes or `complexity_mapping` units; seven-feature linear models cost ~29, default 30 excludes many-term formulas.
- `warmup_maxsize_by=0.5` ramps maxsize over that run fraction, helping avoid early complex traps.
- `parsimony`/`adaptive_parsimony_scaling`: defaults well-tuned in 1.5+.
- Known pi/G: recognize fitted floats afterwards, or supply exact constant-valued features.
- Integers: constant features plus increased `complexity_of_constants`, or custom-operator rounding; continuous optimization does not guarantee integers.

### `complexity_mapping`

Replace fixed operator prices/node counts with a Julia expression-to-`Int` function for emitted characters, evaluation cost, tokens, or depth. Pricing applies to front, `maxsize`, parsimony, `constraints`.

```python
complexity_mapping = """
function source_chars(ex)
    tree_mapreduce(
        leaf -> leaf.constant ? length(string(leaf.val)) : 2,   # "x0" is 2 chars
        branch -> branch.degree == 1 ? 6 : 5,                    # "sin(" + ")" / " + " with parens
        (parent, children...) -> parent + sum(children),
        get_tree(ex),
    )
end
"""
model = PySRRegressor(
    binary_operators=["+", "-", "*"],
    complexity_mapping=complexity_mapping,
    maxsize=280,
)
```

This mapping tracked every test-row `len(equation)` within offset 2 for outer parentheses; rank by actual cost and calibrate against emitted candidates.

- Input: `AbstractExpression`; `get_tree(ex)` exposes `degree`, `constant`, `val`, `feature`, `op`. `tree_mapreduce`/`get_tree` are in scope.
- Avoid `length(string_tree(ex))`: internal rendering uses `binary_operator[3](...)` and `x1, x2` placeholders, unrelated to exported length.
- Set real-budget `maxsize`; default 30 with character pricing leaves `equations_` empty when nothing fits. Hall of fame allocates one slot/unit through maxsize; filter long fine-grained fronts.
- Every candidate incurs computation: cheap, type-stable, no Python callbacks.
- `guesses` use identical pricing/acceptance rules.

## Dimensional constraints

Check physical units during search:

```python
model.fit(X, y, X_units=["Constants.M_sun", "kg", "m"], y_units="kg * m / s^2")
```

- DynamicQuantities.jl notation; `"1"` explicitly dimensionless.
- Soft `dimensional_constraint_penalty`, default 1000; avoid 1e9, preserve graded paths through imperfect intermediates.
- Fitted constants have wildcard `[⋅]`/`[?]` units; a lone constant can absorb any dimensions. Forbid with `dimensionless_constants_only=True`.
- SI radians are dimensionless; units cannot restrict variables to trig.

## Parallelism

- Default `parallelism="multithreading"` suits laptops/single nodes. Thread count is fixed at Julia startup: set `PYTHON_JULIACALL_THREADS=<n>` before importing pysr; `JULIA_NUM_THREADS` is inappropriate under juliacall.
- Keep `populations` ~2-3x threads/cores for worker utilization; default `populations=31` covers typical machines.
- Saturated coordinating thread: increase `ncycles_per_iteration`, reducing communication.
- Long runs: `parallelism="multiprocessing"`, `procs=n` costs substantially more startup/fit, potentially faster steady-state. Multiple nodes: `cluster_manager="slurm"` or native SlurmClusterManager.jl, https://julia.pysr.ai/dev/slurm/. Launch once on one node, let it spawn workers, avoid `srun`; put worker packages in `worker_imports`.
- Reproducibility: `deterministic=True, random_state=<seed>, parallelism="serial"`; parallel seeds remain nondeterministic. CPU differences can persist serially; reduce with `precision=64`.

## Saving, resuming, exporting

- Non-temporary fits write continuously readable `outputs/<run_id>/hall_of_fame.csv` and `checkpoint.pkl`. Reload `PySRRegressor.from_file(run_directory=...)` with identical PySR version; pickles are version-locked. Preserve CSV/construction code durably; resupply custom `extra_sympy_mappings`.
- `warm_start=True` resumes previous populations on next same-process `.fit()`. Fix operators, `maxsize`, `expression_spec`, precision, feature count/order; loss/weights may change for staged objectives.
- Export `model.sympy(i)`, `model.latex(i)`, `model.latex_table()`, differentiable `model.jax(i)` returning `{'callable', 'parameters'}`, trainable `model.pytorch(i)`. Supply custom `extra_jax_mappings`/`extra_torch_mappings`; fine-tune selected equations' constants by full-data gradient descent.
- Julia losses can differ from NumPy recomputation through default 32-bit versus 64-bit precision. Use `precision=64` when important or values exceed ~1e19/fall below ~1e-19, risking Float32 overflow.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First import/fit takes minutes | Initial Julia download; per-process JIT. Retain process. |
| Compilation hangs | Apple Silicon: replace Rosetta x86 Python with native arm64. Notebook/embedded stdin wedge: `input_stream="devnull"`. |
| Startup `lock.pid` | Killed-process juliapkg lock: verify no installation, delete lock, import once freshly. Shared filesystems: pre-initialize before parallel jobs. |
| Jupyter `UnicodeDecodeError` | Old PythonCall bug: upgrade or set `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no` before import. |
| Cannot interrupt notebook | Known limitation; IPython/terminal `q`+Enter only. Set `timeout_in_seconds` or use IPython. |
| Nonsensical equations | Check scale/loss (log-space), missing/redundant operators, insufficient `maxsize`, excluding constraints, in order before more compute. |
| Tiny equations | Insufficient `maxsize` or MSE dominated by largest points. |
| `DomainError` | Guard custom operators with typed NaN. |
| Prediction/SymPy fails | Correct `extra_sympy_mappings`; use SymPy functions. |
| Division near zero | Any-NaN-anywhere rejects candidates; survivors remain finite only on supplied data, potentially diverging elsewhere. |
| Early-stop `ProcessExitedException` | Harmless multiprocessing worker-teardown noise. |
| Growing memory | Mostly fixed in recent Julia; upgrade or set multiprocessing `heap_size_hint_in_bytes`. |
| Different results | Stochastic evolution; apply serial determinism recipe. |

## Dropping to Julia

Use native SymbolicRegression.jl, `SRRegressor` via MLJ, for Julia pipelines/deep customization: expression types, mutation operators, per-component constraints. `from pysr import jl` exposes the runtime; `jl.seval(...)` executes arbitrary Julia, installed packages support operators/losses. Small readable backend extension points: `src/Options.jl`, `src/CheckConstraints.jl`; connect dev checkouts through `pysr/juliapkg.json`.

## Version notes

Target: PySR 2.0.0, SymbolicRegression.jl 2.x. Relative to 1.5.x: n-ary `operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp"]}`, `guesses`, `TypeSpec`, automatic batching, autodiff plugins. Defaults: `batching="auto"`, `batch_size=None`, `annealing=True`, `crossover_probability=0.2`, new mutation mix; use 2.0 defaults rather than 1.x tuning.

| Deprecated | Current |
|---|---|
| `pysr.install()`, `python -m pysr install`, PyCall/`from julia import Main` | Automatic Julia installation on import |
| `pysr(...)`, `best()`, `get_hof()` | `PySRRegressor`, `.fit()`, `model.equations_`, `get_best()` |
| `multithreading=True/False`, `procs=0` | `parallelism="multithreading"/"multiprocessing"/"serial"` |
| `full_objective=` | `loss_function`; templates: `loss_function_expression` |
| `loss=` | `elementwise_loss` |
| `equation_file=` | `output_directory=` + `run_id=` |
| `npop`, `ncyclesperiteration`, camelCase args | `population_size`, `ncycles_per_iteration`, snake_case |
| Positional `TemplateExpressionSpec`, `function_symbols=...` | Explicit `combine=`, `expressions=`, `variable_names=` |
