# Changelog

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
