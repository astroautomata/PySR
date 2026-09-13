# Physics and units

## Dimensional constraints

Dimensional constraints let PySR use physical units during symbolic search. When the units of the features and target are known, pass a unit expression for each feature and one for the target so candidates with mismatched dimensions receive a penalty.

The fit interface follows the unit syntax of `DynamicQuantities.jl`. The following example uses Astropy to sample a mass $M$, a test mass $m$, and a separation $r$, then evaluates Newton's inverse-square law $F = GMm/r^2$.

```python
import numpy as np
from astropy import units as u, constants as const

M = (np.random.rand(100) + 0.1) * const.M_sun
m = 100 * (np.random.rand(100) + 0.1) * u.kg
r = (np.random.rand(100) + 0.1) * const.R_earth
G = const.G

F = G * M * m / r**2
```

The force values span a wide range. This loss compares their magnitudes in logarithmic space and penalizes a wrong sign:

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))
    sign_loss = 10 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end
"""
```

`dimensional_constraint_penalty` adds a cost when a candidate fails dimensional analysis:

```python
from pysr import PySRRegressor

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

Pass numeric values to `fit`, with matching units in column order through `X_units` and the target unit through `y_units`. Unit strings follow the Julia syntax documented by [DynamicQuantities.jl](https://symbolicml.org/DynamicQuantities.jl/dev/#Usage).

```python
import pandas as pd

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

The loss is nonnegative, and the dimensional penalty is additive. A candidate with total loss below that penalty therefore did not receive the dimensional-violation charge and is dimensionally consistent for the supplied units. The marker `"[⋅]"` denotes a constant whose units remain free. The unit-aware form below illustrates this notation: `M[kg]` supplies mass, so the constant supplies the acceleration units needed for the target:

```julia
"y[m s⁻² kg] = (M[kg] * 2.6353e-22[⋅])"
```

Set `dimensionless_constants_only=True` to restrict fitted constants to dimensionless quantities.

## Using differential operators

A [`TemplateExpressionSpec`](/examples/expression-specifications) can include differential operators in its `combine` string. `D` takes the expression to differentiate and an argument index. Here, `D(f, 1)` differentiates the learned function `f` with respect to its first argument, and `df(x)` evaluates that derivative. Matching those values to data turns antiderivative discovery into regression.

The regression target is the integrand $1/(x^2\sqrt{x^2 - 1})$ on $x > 1$. The template searches for a function $f$ whose derivative matches samples of the integrand, leaving an additive constant undetermined.

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

The target antiderivative is $f(x) = \frac{\sqrt{x^2 - 1}}{x}$, up to an additive constant.

## Discovering a PDE

For a field $u(x,t)$ sampled on a space-time grid, each point becomes one regression row: its field value and spatial derivatives are features, and its time derivative is the target. PySR then searches for the right-hand side in $u_t = f(u, u_x, u_{xx}, \ldots)$.

The following example simulates the viscous Burgers equation $u_t = -u\,u_x + 0.1\,u_{xx}$ on a periodic domain. Measurements of the field can replace the simulation when they are arranged into the same feature and target arrays.

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

Estimate spatial derivatives with the smoothing differentiator `savgol_filter` and time derivatives with `np.gradient`. Supply the coordinate spacings and keep each space-time point aligned across the feature columns and target:

```python
import numpy as np

from scipy.signal import savgol_filter

ux = savgol_filter(U, 21, 3, deriv=1, delta=dx, axis=-1)
uxx = savgol_filter(U, 21, 3, deriv=2, delta=dx, axis=-1)
ut = np.gradient(U, t, axis=0)

X = np.stack([U, ux, uxx], axis=-1).reshape(-1, 3)
y = ut.reshape(-1)
```

`complexity_of_variables=[1, 2, 3]` makes higher spatial derivatives more expensive, biasing the search toward lower-order terms when candidate fits compete. A successful low-complexity search should place $0.1\,u_{xx} - u\,u_x$ on the Pareto front, matching the right-hand side that generated the snapshots.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*"],
    complexity_of_variables=[1, 2, 3],
    maxsize=20,
    niterations=100,
)
model.fit(X, y, variable_names=["u", "u_x", "u_xx"])
print(model)
```

The regression uses estimated derivatives, so simulate the discovered PDE forward from a held-out initial condition before relying on its coefficients.

If the field $u$ has units $\mathrm{m\,s^{-1}}$, then $u_x$ has units $\mathrm{s^{-1}}$, $u_{xx}$ has units $\mathrm{m^{-1}\,s^{-1}}$, and $u_t$ has units $\mathrm{m\,s^{-2}}$. For this feature order, pass the corresponding strings through `X_units=["m/s", "s^-1", "m^-1*s^-1"]` and `y_units="m*s^-2"`.
