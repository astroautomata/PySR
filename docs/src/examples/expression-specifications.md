# Expression specifications

## Preamble

```python
import numpy as np

from pysr import *
```

## Expression specifications

Expression specifications let you define a structured equation while retaining
normal prediction and export behavior. Use `TemplateExpressionSpec` when the
outer form is known and one or more inner expressions must be learned.

### Template Expressions

`TemplateExpressionSpec` allows you to define a specific structure for the equation.
For example, let's say we want to learn an equation of the form:

$$ y = \sin(f(x_1, x_2)) + g(x_3) $$

We can do this as follows:

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

When your data has categories with shared equation structure but different parameters,
you can use the `parameters` argument of `TemplateExpressionSpec` to specify learned category-specific parameters.

For example, let's say we want to learn an equation of the form:

$$ y = \alpha \sin(x_1) + \beta $$

where $\alpha$ and $\beta$ are different for each category.

Further, let's say we have 3 categories,
with $\alpha \in \{0.1, 1.5, -0.5\}$ and $\beta \in \{1.0, 2.0, 0.5\}$.

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

Now, let's define our parametric expression:

```python
template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": 3, "p2": 3},  # One parameter per category
    combine="f(x1, x2, p1[category], p2[category])"
)
```

Next, we pass the category as a _column_ in `X`
corresponding to the index we defined in `variable_names`.

**Note that because Julia is 1-indexed, we need to add 1 to the category index.**

```python
category_p_one = category + 1
X_with_category = np.column_stack([X, category_p_one])
```

Now, we can fit our model:

```python
model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X_with_category, y)

# Predicting on new data
# model.predict(X_test_with_category)
```

See [Expression Specifications](/api/#expression-specifications) for more details.

You can use this approach for more complex cases,
where you have multiple expressions in the template and parameters that vary by category.

### Learning multiple outputs jointly

You can use `TemplateExpressionSpec` to learn several scalar expressions jointly
and compare their combined predictions with a vector target. This is useful when
the outputs share a known outer structure. Each learned expression still operates
on scalar values; the template combines their predictions and computes a scalar
residual.

For example, say we have 3-dimensional vectors where each component
follows a pattern with a shared term. Say the true model is:

$$\begin{align*}
y_1 &= \exp(x_1) + x_2^2 \\
y_2 &= \exp(x_1) + \sin(x_3) \\
y_3 &= \exp(x_1) + x_1 \cdot x_2
\end{align*}$$

Let's set this up:

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

Now, we put everything in `X`; BOTH features and targets:

```python
X = np.column_stack([x1, x2, x3, y1, y2, y3])
```

Now, we can define our template expression:

```python
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

Now, we can fit our model using this template. Since
we already computed the per-row squared error inside the template,
we can pass a dummy `y` to the `fit` method, and also define
an `elementwise_loss` that simply returns the residuals (which get
summed over the data):

```python
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

After running, PySR should find both the shared component (`exp(x1)`) as well as individual components (`square(x2)`, `sin(x3)`, and `x1 * x2`).

You can access the individual expressions through the Julia objects:

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

We can also evaluate individual expressions:

```python
from pysr import jl
from pysr.julia_helpers import jl_array

SR = jl.SymbolicRegression

# Get individual trees
f1_tree = julia_expr.trees.f1
shared_tree = julia_expr.trees.shared

# Evaluate at specific points (x1=1, x2=2, x3=3)
test_inputs = jl_array(np.array([[1.0], [2.0], [3.0]]))
f1_result, _ = SR.eval_tree_array(f1_tree, test_inputs, model.julia_options_)
shared_result, _ = SR.eval_tree_array(shared_tree, test_inputs, model.julia_options_)

print(f"f1 at (1,2,3): {f1_result[0]}")  # Should be ~4.0 for x2^2
print(f"shared at (1,2,3): {shared_result[0]}")  # Should be ~2.718 for exp(1)
```

## Recovering a magnetic field from force measurements

Sometimes you know the physics and only the coefficient functions are missing. A charged
particle in a viscous medium obeys

$$ F = -\eta(T)\,v + v \times B(t), $$

and an experiment records only the total force alongside the inputs $(t, v, T)$. The cross
product and the drag term are known; the three components of $B(t)$ and the drag scale
$\eta(T)$ are not. This is exactly the shape a `TemplateExpressionSpec` is for: we write the
fixed physics once, and PySR searches only inside the holes we leave. The film clip shows
this search filling in a rotating field.

Each measurement is a three-component vector, so we define a `Force` value type and let it
flow through the expressions. Note that a custom struct-valued output type works through
`PySRRegressor`: combining a `TypeSpec` with a `TemplateExpressionSpec` carries the custom
output element type through the scikit-learn-side validation, which is the load-bearing part
of this example.

<details>
<summary>Data generation code</summary>

Every value in the problem is a `Force`, so the scalar inputs $t$, $v_i$, and $T$ are embedded
on the diagonal (the same number in all three slots):

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

The type spec declares the three fields and the hooks PySR needs to sample and optimize
constants. The one-argument constructor is the diagonal embedding of a scalar, and it also
gives the evaluator the `Force(Inf)` sentinel it substitutes for an invalid value. A
constant is a whole vector: `sample` draws all three slots and `scalar_constants` hands all
three to BFGS, so the same spec still works when an expression has to return a direction
rather than a magnitude:

```python
FORCE = TypeSpec(
    "Force",
    fields={"x": "Float64", "y": "Float64", "z": "Float64"},
    sample="""begin
        Force(u::Real) = Force(u, u, u)
        rng -> Force(randn(rng, 3)...)
    end""",
    scalar_constants="value -> [value.x, value.y, value.z]",
    with_scalar_constants="(value, c) -> Force(c[1], c[2], c[3])",
    string="value -> sprint(show, (value.x, value.y, value.z); context = :compact => true)",
)
```

The operators extend Julia's own `Base` functions to this type, so we pass each definition
followed by the name PySR should use to look it up. All of them are type-stable through the
`::Force` return annotation, and `sqrt` returns `NaN` rather than throwing on negative input:

```python
def _method(body, name):
    return body + "\n" + name


OPERATORS = {
    1: [
        _method(
            "Base.sin(a::Force)::Force = Force(sin(a.x), sin(a.y), sin(a.z))",
            "Base.sin",
        ),
        _method(
            "Base.cos(a::Force)::Force = Force(cos(a.x), cos(a.y), cos(a.z))",
            "Base.cos",
        ),
        _method(
            """function Base.sqrt(a::Force)::Force
            f(v) = v < 0 ? NaN : sqrt(v)
            return Force(f(a.x), f(a.y), f(a.z))
            end""",
            "Base.sqrt",
        ),
        _method(
            "Base.exp(a::Force)::Force = Force(exp(a.x), exp(a.y), exp(a.z))",
            "Base.exp",
        ),
    ],
    2: [
        _method(
            "Base.:+(a::Force, b::Force)::Force = Force(a.x + b.x, a.y + b.y, a.z + b.z)",
            "Base.:+",
        ),
        _method(
            "Base.:-(a::Force, b::Force)::Force = Force(a.x - b.x, a.y - b.y, a.z - b.z)",
            "Base.:-",
        ),
        _method(
            "Base.:*(a::Force, b::Force)::Force = Force(a.x * b.x, a.y * b.y, a.z * b.z)",
            "Base.:*",
        ),
        _method(
            "Base.:/(a::Force, b::Force)::Force = Force(a.x / b.x, a.y / b.y, a.z / b.z)",
            "Base.:/",
        ),
    ],
}

FORCE_LOSS = "force_loss(a::Force, b::Force)::Float64 = (a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2"
```

Now the template itself. `combine` receives the four sub-expressions as callables and the
input columns as `ValidVector`s. We evaluate each hole once, propagate invalidity if any of
them failed, and then assemble the force from the known law. Because the scalar inputs are
diagonal, we read the `x` slot of each `Force` to get the underlying number, then build one
true three-component vector per row:

```python
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

Each named expression sees only the variables it is called with, so `B_x`, `B_y`, and `B_z`
can depend on time alone and `F_d_scale` on temperature alone. The cross product is never
searched over; it is arithmetic we already trust.

We set `deterministic=True`, `parallelism="serial"`, and a `random_state` so the run is
reproducible; drop the first two if you want the search to use all your cores.

```python
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

On PySR 2.1.0 all 5 of 5 seeds recover the field exactly and reproduce the forces on unseen
experiments to machine precision. The simplest exact member of the front is complexity 13 on
four seeds and 15 on the fifth, and it prints as

    B_x = sin(#1 / (0.159155, 6.43746, -0.55788))
    B_y = cos(#1 / (0.159155, 6.43746, -0.55788))
    B_z = exp(#1 * (-0.1, -0.334652, -1.00446))
    F_d_scale = -0.000172745, 0.262294, 0.471951

against the truth `1/(2*pi) = 0.159155`, `-1/10`, and `1e-5*sqrt(298.4) = 1.727e-4`. Two
details of that output are worth reading carefully. `#1` is the first argument of the
sub-expression, here `t` for the field components and `T` for the drag, since a named
sub-expression prints its own arguments positionally. And each constant shows three numbers
because a `Force` constant is a whole vector, while the template reads only the `x` slot of
each hole: the second and third components are never seen by the loss, so they keep whatever
the sampler drew and carry no meaning. That freedom is not free. Handing BFGS three
parameters per constant where one is read costs a factor of about three in time against a
diagonal-only spec, 4877 to 5507 seconds per seed against 1544 to 2106, which makes this the
slowest example here at roughly an hour and a half for a single run.

The full runnable script is `examples/magnetic_field.py`.
