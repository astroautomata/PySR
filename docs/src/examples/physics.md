# Physics and units

## Preamble

```python
import numpy as np

from pysr import *
```

## Dimensional constraints

One other feature we can exploit is dimensional analysis.
Say that we know the physical units of each feature and output,
and we want to find an expression that is dimensionally consistent.

We can do this as follows, using `DynamicQuantities.jl` to assign units,
passing a string specifying the units for each variable.
First, let's make some data on Newton's law of gravitation, using
astropy for units:

```python
import numpy as np
from astropy import units as u, constants as const

M = (np.random.rand(100) + 0.1) * const.M_sun
m = 100 * (np.random.rand(100) + 0.1) * u.kg
r = (np.random.rand(100) + 0.1) * const.R_earth
G = const.G

F = G * M * m / r**2
```

We can see the units of `F` with `F.unit`.

Now, let's create our model.
Since this data has such a large dynamic range,
let's also create a custom loss function
that looks at the error in log-space:

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))
    sign_loss = 10 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end
"""
```

Now let's define our model:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    elementwise_loss=elementwise_loss,
    complexity_of_constants=2,
    maxsize=25,
    niterations=100,
    populations=50,
    # Amount to penalize dimensional violations:
    dimensional_constraint_penalty=10**5,
)
```

and fit it, passing the unit information.
To do this, we need to use the format of [DynamicQuantities.jl](https://symbolicml.org/DynamicQuantities.jl/dev/#Usage).

```python
# Get numerical arrays to fit:
X = pd.DataFrame(dict(
    M=M.to("M_sun").value,
    m=m.to("kg").value,
    r=r.to("R_earth").value,
))
y = F.value

model.fit(
    X,
    y,
    X_units=["Constants.M_sun", "kg", "Constants.R_earth"],
    y_units="kg * m / s^2"
)
```

You can observe that all expressions with a loss under
our penalty are dimensionally consistent!
(The `"[⋅]"` indicates free units in a constant, which can cancel out other units in the expression.)
For example,

```julia
"y[m s⁻² kg] = (M[kg] * 2.6353e-22[⋅])"
```

would indicate that the expression is dimensionally consistent, with
a constant `"2.6353e-22[m s⁻²]"`.

Note that this expression has a large dynamic range so may be difficult to find. Consider searching with a larger `niterations` if needed.

Note that you can also search for exclusively dimensionless constants by settings
`dimensionless_constants_only` to `true`.

## Using differential operators

As part of the [`TemplateExpressionSpec`](/examples/expression-specifications),
you can also use differential operators within the template.
The operator for this is `D` which takes an expression as the first argument,
and the argument _index_ we are differentiating as the second argument.
This lets you compute integrals via evolution.

For example, let's say we wish to find the integral of $\frac{1}{x^2 \sqrt{x^2 - 1}}$
in the range $x > 1$.
We can compute the derivative of a function $f(x)$, and compare that
to numerical samples of $\frac{1}{x^2\sqrt{x^2-1}}$. Then, by extension,
$f(x)$ represents the indefinite integral of it with some constant offset!

```python
import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec

x = np.random.uniform(1, 10, (1000,))  # Integrand sampling points
y = 1 / (x**2 * np.sqrt(x**2 - 1))     # Evaluation of the integrand

expression_spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x"],
    combine="df = D(f, 1); df(x)",
)

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    expression_spec=expression_spec,
    maxsize=20,
)
model.fit(x[:, np.newaxis], y)
```

If everything works, you should find something that simplifies to $\frac{\sqrt{x^2 - 1}}{x}$.

Here, we write out a full function in Julia.

## Discovering a PDE

Suppose we have data in the form of a field `u(x, t)`: measurements of
some quantity on a grid of positions, repeated over time. We can
discover the PDE `u_t = f(u, u_x, u_xx, ...)` by turning this into a
normal regression problem: every grid point is one sample, the input
features are the field and its spatial derivatives, and the target is
the time derivative.

Let's simulate the viscous Burgers equation,
`u_t = -u*u_x + 0.1*u_xx`, on a periodic domain
(in practice you would use measured data instead):

<details>
<summary>Data generation code</summary>

```python
import numpy as np

L, nx, nu = 2 * np.pi, 128, 0.1
x = np.linspace(0.0, L, nx, endpoint=False)
dx = L / nx
k = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)

def d_dx(u, order=1):
    return np.real(np.fft.irfft((1j * k) ** order * np.fft.rfft(u), n=nx))

def rhs(u):
    return -u * d_dx(u) + nu * d_dx(u, order=2)

u = -np.sin(x)
snapshots = [u.copy()]
dt = 1e-3
for i in range(1, 4001):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if i % 100 == 0:
        snapshots.append(u.copy())

U = np.array(snapshots)   # shape (41, 128)
t = np.arange(U.shape[0]) * 0.1
```

</details>

Now we build the feature matrix and target. The spatial derivatives are
computed with a Savitzky-Golay filter, which smooths the data while
differentiating it; the time derivative is a finite difference:

```python
from scipy.signal import savgol_filter

ux = savgol_filter(U, 21, 3, deriv=1, delta=dx, axis=-1)
uxx = savgol_filter(U, 21, 3, deriv=2, delta=dx, axis=-1)
ut = np.gradient(U, t, axis=0)

X = np.stack([U, ux, uxx], axis=-1).reshape(-1, 3)
y = ut.reshape(-1)
```

Now let's fit. The `complexity_of_variables` list makes higher
derivatives cost more, which biases the search toward low-order terms:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*"],
    complexity_of_variables=[1, 2, 3],
    maxsize=20,
    niterations=100,
)
model.fit(X, y, variable_names=["u", "u_x", "u_xx"])
print(model)
```

If all goes well, you should see an equation like `0.1 * u_xx - u * u_x`
on the Pareto front at low complexity, which is the PDE we put in!
This setup also survives noise well: at 1% Gaussian noise,
finite-difference features fail completely, but the Savitzky-Golay
features above still recover the right terms (coefficients ~13% low),
and even at 5% noise the structure survives. Before trusting the
constants, simulate the discovered PDE forward from a held-out initial
condition.

Finally, if you know the physical units, pass them to `fit` with
`X_units=["m/s", "s^-1", "m^-1*s^-1"]` and `y_units="m*s^-2"`.
