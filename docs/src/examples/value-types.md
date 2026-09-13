# Value types

## Complex numbers

To fit complex-valued expressions, pass feature and target arrays with a complex NumPy dtype.

The code constructs a complex cosine target and restricts the search to `+`, `-`, `*`, and `cos`:

```python
import numpy as np

from pysr import PySRRegressor

X = np.random.randn(100, 1) + 1j * np.random.randn(100, 1)
y = (1 + 2j) * np.cos(X[:, 0] * (0.5 - 0.2j))

model = PySRRegressor(
    binary_operators=["+", "-", "*"], unary_operators=["cos"], niterations=100,
)

model.fit(X, y)
```

The fitted constants belong to the complex-valued expression domain. Use `model.sympy()` to inspect the selected equation in SymPy form:

```python
model.sympy()
```

Pass an equation-row index to `predict`; `-1` selects the final row of `equations_`:

```python
model.predict(X, -1)
```

## Julia packages and types

PySR delegates its symbolic-regression search to the pure Julia package [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl). A custom operator can call functionality from another Julia package through that backend.

Load the package that supplies the prime lookup before defining the operator. The target in this example is

$$ y = p_{3x + 1} - 5, $$

where $p_i$ denotes the $i$th prime and $x$ is the input feature. The lookup comes from [Primes.jl](https://github.com/JuliaMath/Primes.jl).

First, import the handle to PySR's Julia runtime:

```python
from pysr import jl
```

`jl` is the juliacall runtime object, and `jl.seval` evaluates Julia source from Python. Use it to add `Primes.jl` to the active PySR environment:

```python
from pysr import jl

jl.seval("""
import Pkg
Pkg.add("Primes")
""")
```

Adding a package and importing its namespace are separate steps. Import `Primes` before defining `p`:

```python
from pysr import jl

jl.seval("import Primes")
```

With the package available, define the unary operator used by the search:

```python
from pysr import jl

jl.seval("""
function p(i::T) where T
    if (0.5 < i < 1000)
        return T(Primes.prime(round(Int, i)))
    else
        return T(NaN)
    end
end
""")
```

The operator rounds inputs within `0.5 < i < 1000` to prime indices and preserves the input type `T`. Outside that domain it returns `T(NaN)` to reject invalid evaluations.

Call the Julia function through `jl.p` to build the prime values used by the synthetic data:

```python
from pysr import jl

primes = {i: jl.p(i*1.0) for i in range(1, 999)}
```

Use the lookup table to generate noisy samples of the target:

```python
import numpy as np

X = np.random.randint(0, 100, 100)[:, None]
y = [primes[3*X[i, 0] + 1] - 5 + np.random.randn()*0.001 for i in range(100)]
```

Register `p` with the regular PySR operator interface. The `sympy_p` subclass and `extra_sympy_mappings` give SymPy a symbolic form for the custom function:

```python
from pysr import PySRRegressor
import sympy

class sympy_p(sympy.Function):
    pass

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["p"],
    niterations=100,
    extra_sympy_mappings={"p": sympy_p}
)
```

Fit the model on the generated pairs:

```python
model.fit(X, y)
```

Since `p` rounds its argument, different real offsets can produce the same integer prime index, so the offset in a fitted expression need not equal `1` exactly. Inspect the selected equation in SymPy with:

```python
model.sympy()
```

## Custom value types

`TypeSpec` declares a custom Julia value that can flow through an expression tree. Vectors, tensors, strings, structs, and other Julia value types can serve as payloads when their behavior is described by the type's hooks, operators, and loss.

A model configured with `type_spec=TypeSpec(...)` needs three pieces:

1. Define hooks that sample and change values. For continuously optimized constants, `scalar_constants` exposes scalar coordinates and `with_scalar_constants` rebuilds the value. A custom `mutate` hook handles changes that need a discrete or structural operation.
2. Choose exactly one of `elementwise_loss`, `loss_function`, or `loss_function_expression`. The selected loss must accept the custom values and return a real-valued score. Elementwise-loss return types are inferred; a full objective requires an explicit `loss_type`.
3. Supply operators through `operators={...}`, grouped by arity. Each operator must accept and return the declared value type so the type-stability check succeeds. Add an explicit return annotation such as `::Vec2` when Julia cannot infer it.

The examples below combine these pieces for vector values, strings as discrete constants, and payloads with several shapes.

### Vector-valued expression trees

This example searches for an expression whose values are two-dimensional vectors:

$$
y = \operatorname{rotate90}(x_1) + 2x_2 +
\begin{bmatrix}0.5 \\ -1.0\end{bmatrix}.
$$

The 128-by-2 DataFrame has two vector-valued features per row, and each entry of `y` is one target vector. The table dimensions describe samples and features, while each cell of the object array holds a single logical vector value.

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

rng = np.random.default_rng(0)
x1 = [rng.normal(size=2) for _ in range(128)]
x2 = [rng.normal(size=2) for _ in range(128)]
X = pd.DataFrame({"x1": x1, "x2": x2})
y = np.empty(128, dtype=object)
offset = np.array([0.5, -1.0])
y[:] = [np.array([-a[1], a[0]]) + 2 * b + offset for a, b in zip(x1, x2)]
```

Declare `Vec2` with one `Vector{Float64}` field. `sample` creates a random payload of length two, while `scalar_constants` and `with_scalar_constants` expose and rebuild the entries used by continuous constant optimization:

```python
from pysr import TypeSpec

type_spec = TypeSpec(
    "Vec2",
    fields={"data": "Vector{Float64}"},
    sample="rng -> Vec2(randn(rng, 2))",
    scalar_constants="value -> value.data",
    with_scalar_constants=(
        "(value, scalar_constants) -> Vec2(scalar_constants)"
    ),
)
```

PySR derives how values are initialized, mutated, validated, counted, packed, and unpacked from these hooks. Add `init` or `mutate` for specialized search behavior, or `string` for custom display, when the derived defaults are insufficient.

`TypeSpec` generates the Julia value type from that declaration. Its equivalent Julia struct has one vector field:

```julia
struct Vec2
    data::Vector{Float64}
end
```

Define operators that consume and return `Vec2` values:

```python
operators = {
    1: [
        "rotate90(a::Vec2) = Vec2([-a.data[2], a.data[1]])",
        "double(a::Vec2) = Vec2(2a.data)",
    ],
    2: ["add_vectors(a::Vec2, b::Vec2) = Vec2(a.data + b.data)"],
}
```

The elementwise loss compares vector payloads by summing squared differences, which gives the real-valued score the search requires:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=type_spec,
    operators=operators,
    elementwise_loss="vector_loss(a::Vec2, b::Vec2) = sum(abs2, a.data - b.data)",
    niterations=40,
    populations=4,
    maxsize=10,
)

model.fit(X, y)
print(model.equations_)
```

The search space can represent the target as `add_vectors(add_vectors(rotate90(x1), double(x2)), Vec2([0.5, -1.0]))`. The vector constant carries both offset components, which the scalar hook pair exposes for joint optimization.


<details>
<summary>String-valued expressions and discrete constants</summary>

This example treats each string as a value and chooses a separator from a finite set during mutation. It lowercases the first input, uppercases the second, and joins them with the selected separator:

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

X = pd.DataFrame(
    {
        "first": ["Py", "symbolic", "hello", "left"],
        "second": ["SR", "regression", "world", "right"],
    }
)
y = np.array(
    [f"{a.lower()}-{b.upper()}" for a, b in X.itertuples(index=False)],
    dtype=object,
)

type_spec = TypeSpec(
    "StringValue",
    fields={"data": "String"},
    sample='rng -> StringValue(rand(rng, ("", "-", "_")))',
    mutate="""
    mutate_string(rng, value, temperature) = StringValue(rand(rng, ("", "-", "_")))
    """,
)

model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: [
            "string_lowercase(x::StringValue) = StringValue(lowercase(x.data))",
            "string_uppercase(x::StringValue) = StringValue(uppercase(x.data))",
        ],
        2: [
            "string_concat(a::StringValue, b::StringValue) = StringValue(a.data * b.data)"
        ],
    },
    elementwise_loss="""
    string_loss(a::StringValue, b::StringValue) = Float64(Base.editdistance(a.data, b.data))
    """,
    niterations=40,
)

model.fit(X, y)
print(model.equations_)
```

The `sample` and `mutate` hooks choose discrete separator constants. Without a
`scalar_constants` and `with_scalar_constants` pair, BFGS leaves them unchanged.
Edit distance provides the real-valued loss for string outputs.

</details>

<details>
<summary>Advanced: recovering a neural network with tensor constants</summary>

A `TypeSpec` can place scalar, vector, and matrix payloads in the same Julia value type. Its scalar-constant hooks flatten each payload for BFGS and rebuild its original shape.

The advanced example searches for this two-layer neural network:

$$ y = W_2\operatorname{relu}(W_1x + b_1) + b_2 $$

The targets are vectors. `safe_matmul` and `safe_add` return `NaN` for incompatible shapes, so expressions using those helpers receive an invalid value instead of a dimension error:

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

preamble = """
const NNPayload = Union{Float64, Vector{Float64}, Matrix{Float64}}

safe_matmul(a::Matrix{Float64}, b::Vector{Float64}) =
    size(a, 2) == length(b) ? a * b : NaN
safe_matmul(::NNPayload, ::NNPayload) = NaN

safe_add(a::Float64, b::Float64) = a + b
safe_add(a::T, b::T) where {T<:Union{Vector{Float64}, Matrix{Float64}}} =
    size(a) == size(b) ? a + b : NaN
safe_add(::NNPayload, ::NNPayload) = NaN

# Constants sample a random rank, generating only the payload that was chosen:
function random_nn_payload(rng)
    rank = rand(rng, 0:2)
    rank == 0 ? randn(rng) : rank == 1 ? randn(rng, 2) : randn(rng, 2, 2)
end
"""

type_spec = TypeSpec(
    "NNValue",
    fields={"data": "NNPayload"},
    sample="rng -> NNValue(random_nn_payload(rng))",
    # Mutations usually perturb every scalar in the payload, but occasionally
    # resample a fresh rank:
    mutate="""
    (rng, value, temperature) -> if rand(rng) < 0.1
        NNValue(random_nn_payload(rng))
    else
        NNValue(value.data .+ temperature .* randn(rng, size(value.data)...))
    end
    """,
    scalar_constants="""
    function scalar_constants(value)
        return value.data isa Float64 ? [value.data] : vec(value.data)
    end
    """,
    with_scalar_constants="""
    function with_scalar_constants(value, scalar_constants)
        data = value.data isa Float64 ? scalar_constants[1] :
            reshape(collect(scalar_constants), size(value.data))
        return NNValue(data)
    end
    """,
    preamble=preamble,
)
```

`preamble` makes the payload union and helpers available before the generated type is
defined. Mutation can change a constant's rank; continuous optimization preserves its
shape through the scalar hook pair.

Generate vector-valued targets from fixed weights and biases:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x_values = rng.normal(size=(64, 2))
W1 = np.array([[1.2, -0.7], [0.5, 1.1]])
b1 = np.array([0.3, -0.2])
W2 = np.array([[0.8, -1.0], [1.3, 0.4]])
b2 = np.array([-0.4, 0.2])
y_values = (W2 @ np.maximum(x_values @ W1.T + b1, 0).T).T + b2

X = pd.DataFrame({"x": list(x_values)})
y = pd.Series(list(y_values), dtype=object)
```

Register the neural-network operations and score matching vector outputs with mean squared error. The loss penalizes other output shapes:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: ["nn_relu(a) = NNValue(max.(a.data, 0.0))"],
        2: [
            "nn_matmul(a, b) = NNValue(safe_matmul(a.data, b.data))",
            "nn_add(a, b) = NNValue(safe_add(a.data, b.data))",
        ],
    },
    elementwise_loss="""
    function nn_mse(a, b)::Float64
        valid = a.data isa Vector && b.data isa Vector && size(a.data) == size(b.data)
        return valid ? sum(abs2, a.data .- b.data) / length(a.data) : 1.0e6
    end
    """,
    niterations=100,
    populations=4,
    maxsize=11,
)

model.fit(X, y)
print(model.equations_)
```

A recovered expression can have the form `nn_matmul(W2, nn_add(b, nn_relu(nn_matmul(W1, nn_add(x, c)))))`. The inner constant `c` represents the first bias through $b_1 = W_1c$, and the outer constant `b` represents the second through $b_2 = W_2b$. Each displayed constant is an `NNValue` containing its fitted scalar, vector, or matrix payload.

</details>

The same custom-value interface can seed a search with `guesses`. In the vector example above,
`Vec2` is the generated constructor from its `TypeSpec`, `add_vectors` is the declared binary
operator, and `x1` is the first named feature column. A matching guess is
`guesses=["add_vectors(x1, Vec2([1.0, 2.0]))"]`.
