# Expression specifications

## Expression specifications

Use an expression specification when part of the equation is known in advance.
It fixes the structure around learnable subexpressions while preserving PySR's ordinary
fitting and prediction interfaces. Inspect template components through `model.equations_`;
the arbitrary Julia `combine` string does not provide the usual SymPy or LaTeX exports.
Choose `TemplateExpressionSpec` when the outer form is known and the inner functions remain unknown.

### Template Expressions

Write the outer form in `combine` and name each learnable hole in `expressions`.
`variable_names` supplies the column names available to `combine`, and each
hole receives the arguments written in its call. Here `f` receives `x_1` and `x_2`,
while `g` receives `x_3`, so the search keeps those input roles separate. The target
form is:

$$ y = \sin(f(x_1, x_2)) + g(x_3) $$

The code below creates the data, declares the template, and fits the model:

```python
import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec

# Create data
X = np.random.randn(1000, 3)
y = np.sin(X[:, 0] + X[:, 1]) + X[:, 2]**2

# Define template: we want sin(f(x1, x2)) + g(x3)
template = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
    combine="sin(f(x1, x2)) + g(x3)",
)

model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X, y)
```

### Parametric Expressions

Add learnable parameter vectors to the template when categories share a formula
but require different coefficients. This example uses category-specific scale and offset
values for the target:

$$ y = \alpha \sin(x_1) + \beta $$

Here $\alpha$ is the category-specific scale and $\beta$ is the offset. The
example has $3$ categories, with $\alpha \in \{1.0, 2.0, 0.5\}$ and
$\beta \in \{0.1, 1.5, -0.5\}$.

```python
import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec

# Create data with 2 features and 3 categories
X = np.random.uniform(-3, 3, (1000, 2))
category = np.random.randint(0, 3, 1000)

# Parameters for each category
offsets = [0.1, 1.5, -0.5]
scales = [1.0, 2.0, 0.5]

# y = scale[category] * sin(x1) + offset[category]
y = np.array([
    scales[c] * np.sin(x1) + offsets[c]
    for x1, c in zip(X[:, 0], category)
])
```

Declare the parameter-vector lengths, optimize their entries during the search,
and use the selected values in the template. The following block defines that
parametric expression:

```python
from pysr import TemplateExpressionSpec

template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": 3, "p2": 3},  # One parameter per category
    combine="f(x1, x2, p1[category], p2[category])"
)
```

Append the category as the column listed in `variable_names`; the template indexes
that column when it selects parameter entries. Python labels start at zero, whereas
Julia arrays start at one, so shift the labels before stacking them.

```python
import numpy as np

category_p_one = category + 1
X_with_category = np.column_stack([X, category_p_one])
```

Fit the model with the augmented feature matrix, and retain this category column
for any later `predict` call. The template uses it to choose the corresponding
parameter entry:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X_with_category, y)

# Evaluate the fitted model on the same feature/category rows:
model.predict(X_with_category)
```

See [Expression Specifications](/api/#expression-specifications) for the complete
reference to this API. The same pattern extends to several template holes and to parameter
vectors whose entries vary by category.

### Learning multiple outputs jointly

`TemplateExpressionSpec` can couple several scalar holes through one scalar
residual. Put the observed components in `X`, let the template construct each
prediction, and return the combined error. This is useful when the components share
a known outer term.

The target consists of $3$ components. Each contains the shared term $\exp(x_1)$
and its own second term:

$$\begin{align*}
y_1 &= \exp(x_1) + x_2^2 \\
y_2 &= \exp(x_1) + \sin(x_3) \\
y_3 &= \exp(x_1) + x_1 \cdot x_2
\end{align*}$$

The data and their noise are generated in the following block:

```python
import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec

n = 200
rstate = np.random.RandomState(0)
x1 = rstate.uniform(-2, 2, n)
x2 = rstate.uniform(-2, 2, n)
x3 = rstate.uniform(-2, 2, n)

# True model with shared component exp(x1):
y1 = np.exp(x1) + x2**2
y2 = np.exp(x1) + np.sin(x3)
y3 = np.exp(x1) + x1 * x2

# Add some noise
y1 += 0.05 * rstate.randn(n)
y2 += 0.05 * rstate.randn(n)
y3 += 0.05 * rstate.randn(n)
```

`X` contains the three inputs followed by the three observed target components,
in the same order listed in `variable_names`. The template can therefore use the
target columns while building its residual:

```python
import numpy as np

X = np.column_stack([x1, x2, x3, y1, y2, y3])
```

The template returns the sum of squared errors across the three components for each row.

These sums constrain the predicted components, while the individual `shared` and `f*` terms are not uniquely identifiable. A common function can be added to `shared` and subtracted from every `f*` without changing the residual, so the values in the evaluation block are conditional on recovering the intended decomposition.

```python
from pysr import TemplateExpressionSpec

spec = TemplateExpressionSpec(
    expressions=["f1", "f2", "f3", "shared"],
    variable_names=["x1", "x2", "x3", "y1", "y2", "y3"],
    combine="""
        v = shared(x1, x2, x3)
        y1_predicted = v + f1(x1, x2, x3)
        y2_predicted = v + f2(x1, x2, x3)
        y3_predicted = v + f3(x1, x2, x3)

        residuals = (
            abs2(y1 - y1_predicted) +
            abs2(y2 - y2_predicted) +
            abs2(y3 - y3_predicted)
        )

        residuals
    """
)
```

The fit uses a dummy target and an `elementwise_loss` that returns the template
value, because `combine` already returns a row-wise squared residual. The estimator
still receives `dummy_y` to satisfy the fit interface; those zeros do not enter the
residual:

```python
import numpy as np

from pysr import PySRRegressor

model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "sin"],
    maxsize=20,
    niterations=50,
    elementwise_loss="(pred, target) -> pred",
)

dummy_y = np.zeros(n)
model.fit(X, dummy_y)
```

Inspect the fitted named expressions through their Julia objects:

```python
# Simply get the expression with the highest score:
idx = model.equations_.score.idxmax()

# Extract the Julia object:
julia_expr = model.equations_.loc[idx, 'julia_expression']

# Access individual subexpressions:
for name in ['f1', 'f2', 'f3', 'shared']:
    tree = getattr(julia_expr.trees, name)
    print(f"{name}: {tree}")
```

The named trees can also be evaluated directly through the Julia bridge:

```python
import numpy as np

from pysr import jl
from pysr.julia_helpers import jl_array

SR = jl.SymbolicRegression

# Get individual trees
f1_tree = julia_expr.trees.f1
shared_tree = julia_expr.trees.shared

# Evaluate one point (x1=1, x2=2, x3=3). `eval_tree_array` expects
# features by rows, so this is a 3-by-1 array.
test_inputs = jl_array(np.array([[1.0], [2.0], [3.0]]))
f1_result, _ = SR.eval_tree_array(f1_tree, test_inputs, model.julia_options_)
shared_result, _ = SR.eval_tree_array(shared_tree, test_inputs, model.julia_options_)

print(f"f1 at (1,2,3): {f1_result[0]}")  # Intended target term: x2^2 = 4.0
print(f"shared at (1,2,3): {shared_result[0]}")  # Intended shared term: exp(1) ≈ 2.718
```

## Recovering a magnetic field from force measurements

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D12.mp4"></video>

The force law is fixed in this example, while its coefficient functions are unknown.
A charged particle in a viscous medium follows

$$ F = -\eta(T)\,v + v \times B(t), $$

The data provide $(t, v, T)$ and the resulting force. The template fixes the drag and
cross-product assembly and leaves, as learnable holes, the three components of $B(t)$
and the temperature-dependent drag scale $\eta(T)$.

Each target is a three-component value, so the example defines a `Force` type with three
fields. Pairing `TypeSpec` with `TemplateExpressionSpec` lets `PySRRegressor` pass these
custom values through the fit while the template constructs each force from the known law.
Here the template reads the `.x` field from each returned hole; the other fields remain
part of the `Force` representation.

The complete runnable script is `examples/magnetic_field.py`; the blocks below expose its
data and model definitions in the same order.

<details>
<summary>Data generation code</summary>

Each scalar input is embedded as a diagonal `Force`, with the same value in each of the
three fields. The output remains a full three-component force. This lets the template recover
scalar inputs from `.x` while keeping one custom value type for both `X` and `y`:

`TypeSpec` converts each tuple in these object arrays into the generated `Force` using the
declared field order (`x`, `y`, `z`), so Python does not need to call `Force(...)` directly.

```python
import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec, TypeSpec

OMEGA = 2 * np.pi


def experiments(n, seed):
    rng = np.random.default_rng(seed)
    T = 298.15 + 0.5 * rng.random(n)
    t = 10.0 * rng.random(n)
    v = 2.0 * rng.random((n, 3)) - 1.0
    B = np.stack([np.sin(OMEGA * t), np.cos(OMEGA * t), np.exp(-t / 10.0)], axis=1)
    F = -1e-5 * np.sqrt(T)[:, None] * v + np.cross(v, B)
    return np.column_stack([t, v[:, 0], v[:, 1], v[:, 2], T]), F


def as_forces(rows):
    """An (n, 3) float array as a column of n Force values."""
    out = np.empty(len(rows), dtype=object)
    out[:] = [tuple(map(float, row)) for row in rows]
    return out


def on_diagonal(columns):
    out = np.empty(columns.shape, dtype=object)
    for j in range(columns.shape[1]):
        out[:, j] = as_forces(np.repeat(columns[:, j, None], 3, axis=1))
    return out


INPUTS, FORCES = experiments(1000, seed=0)
X = on_diagonal(INPUTS)
y = as_forces(FORCES)
variable_names = ["t", "v_x", "v_y", "v_z", "T"]

HELD_OUT_INPUTS, HELD_OUT_FORCES = experiments(500, seed=12345)
HELD_OUT_X = on_diagonal(HELD_OUT_INPUTS)
```

</details>

`FORCE` declares the three fields and the hooks that control sampling, constant
optimization, string output, and failed evaluation. `sample` draws a value for each
field. The paired `scalar_constants` and `with_scalar_constants` hooks expose and restore
all three components, so every `Force` constant is a vector. `init_invalid` supplies a
`Force` with `NaN` in every field. No explicit `is_valid` is given, so the
default finite-constant check rejects that value:

```python
from pysr import TypeSpec

FORCE = TypeSpec(
    "Force",
    fields={"x": "Float64", "y": "Float64", "z": "Float64"},
    sample="rng -> Force(randn(rng, 3)...)",
    definitions="""
        Base.sin(a::Force)::Force = Force(sin(a.x), sin(a.y), sin(a.z))
        Base.cos(a::Force)::Force = Force(cos(a.x), cos(a.y), cos(a.z))
        function Base.sqrt(a::Force)::Force
            f(v) = v < 0 ? NaN : sqrt(v)
            return Force(f(a.x), f(a.y), f(a.z))
        end
        Base.exp(a::Force)::Force = Force(exp(a.x), exp(a.y), exp(a.z))
        Base.:+(a::Force, b::Force)::Force = Force(a.x + b.x, a.y + b.y, a.z + b.z)
        Base.:-(a::Force, b::Force)::Force = Force(a.x - b.x, a.y - b.y, a.z - b.z)
        Base.:*(a::Force, b::Force)::Force = Force(a.x * b.x, a.y * b.y, a.z * b.z)
        Base.:/(a::Force, b::Force)::Force = Force(a.x / b.x, a.y / b.y, a.z / b.z)
    """,
    scalar_constants="value -> [value.x, value.y, value.z]",
    with_scalar_constants="(value, c) -> Force(c[1], c[2], c[3])",
    string="value -> sprint(show, (value.x, value.y, value.z); context = :compact => true)",
    init_invalid="() -> Force(NaN, NaN, NaN)",
)
```

`definitions` runs after the `Force` type is generated, so its Julia methods can return
`Force` values. The guarded `sqrt` returns `NaN` for negative fields instead of throwing.
Register the methods by arity and define a loss over all three force components:

```python
OPERATORS = {
    1: [
        "Base.sin",
        "Base.cos",
        "Base.sqrt",
        "Base.exp",
    ],
    2: [
        "Base.:+",
        "Base.:-",
        "Base.:*",
        "Base.:/",
    ],
}

FORCE_LOSS = "force_loss(a::Force, b::Force)::Float64 = (a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2"
```

Inside a template `combine`, PySR's backend wraps input columns and named-hole results in
`ValidVector` values. The wrapper exposes `.x` for raw values and `.valid` for evaluation
status; `ValidVector(raw, valid)` constructs a wrapper for a new result. This type is supplied
by the template runtime rather than defined in this example.

`PHYSICS` checks the hole results for validity, then assembles one `Force` per row
using the fixed cross product and drag law:

```python
from pysr import TemplateExpressionSpec

PHYSICS = r"""
begin
    _B_x = B_x(t)
    _B_y = B_y(t)
    _B_z = B_z(t)
    _F_d_scale = F_d_scale(T)
    if !(_B_x.valid && _B_y.valid && _B_z.valid && _F_d_scale.valid)
        return ValidVector(_B_x.x, false)
    end
    F = map(_B_x.x, _B_y.x, _B_z.x, _F_d_scale.x,
            v_x.x, v_y.x, v_z.x) do bx, by, bz, fd, ux, uy, uz
        b1, b2, b3 = bx.x, by.x, bz.x
        u1, u2, u3 = ux.x, uy.x, uz.x
        s = fd.x
        Force(u2 * b3 - u3 * b2 + s * u1,
              u3 * b1 - u1 * b3 + s * u2,
              u1 * b2 - u2 * b1 + s * u3)
    end
    ValidVector(F, true)
end
"""

STRUCTURE = TemplateExpressionSpec(
    combine=PHYSICS,
    expressions=["B_x", "B_y", "B_z", "F_d_scale"],
    variable_names=variable_names,
)
```

The field holes receive only `t`; `F_d_scale` receives only `T`. Only the `.x` fields
of their `Force` values enter the numerical residual. The `.y` and `.z` fields can still
affect whether intermediate values remain valid.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=FORCE,
    expression_spec=STRUCTURE,
    operators=OPERATORS,
    elementwise_loss=FORCE_LOSS,
    niterations=100,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    random_state=0,
)

model.fit(X, y, variable_names=variable_names)
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```

An example fitted expression is:

    B_x = sin(#1 / (0.159155, 6.43746, -0.55788))
    B_y = cos(#1 / (0.159155, 6.43746, -0.55788))
    B_z = exp(#1 * (-0.1, -0.334652, -1.00446))
    F_d_scale = -0.000172745, 0.262294, 0.471951

`#1` is the first argument passed to each hole: `t` for the field components and
`T` for the drag scale. Constants print three entries because they are full `Force`
values. The field expressions match the sinusoidal and exponential forms used to
generate the data; the constant drag scale only approximates the temperature-dependent law.

The script validates force predictions on independently generated experiments with
`held_out_error(model)`. Accurate force predictions alone do not establish exact
recovery of every coefficient function. Expect an expensive search: a run took about
1.5 hours in the recorded full-vector configuration.
