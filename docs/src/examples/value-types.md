# Value types

## Preamble

```python
import numpy as np

from pysr import *
```

## Complex numbers

PySR can also search for complex-valued expressions. Simply pass
data with a complex datatype (e.g., `np.complex128`),
and PySR will automatically search for complex-valued expressions:

```python
import numpy as np

X = np.random.randn(100, 1) + 1j * np.random.randn(100, 1)
y = (1 + 2j) * np.cos(X[:, 0] * (0.5 - 0.2j))

model = PySRRegressor(
    binary_operators=["+", "-", "*"], unary_operators=["cos"], niterations=100,
)

model.fit(X, y)
```

You can see that all of the learned constants are now complex numbers.
We can get the sympy version of the best equation with:

```python
model.sympy()
```

We can also make predictions normally, by passing complex data:

```python
model.predict(X, -1)
```

to make predictions with the most accurate expression.
## Julia packages and types

PySR uses [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl)
as its search backend. This is a pure Julia package, and so can interface easily with any other
Julia package.
For some tasks, it may be necessary to load such a package.

For example, let's say we wish to discovery the following relationship:

$$ y = p_{3x + 1} - 5, $$

where $p_i$ is the $i$th prime number, and $x$ is the input feature.

Let's see if we can discover this using
the [Primes.jl](https://github.com/JuliaMath/Primes.jl) package.

First, let's get the Julia backend:

```python
from pysr import jl
```

`jl` stores the Julia runtime.

Now, let's run some Julia code to add the Primes.jl
package to the PySR environment:

```python
jl.seval("""
import Pkg
Pkg.add("Primes")
""")
```

This imports the Julia package manager, and uses it to install
`Primes.jl`. Now let's import `Primes.jl`:

```python
jl.seval("import Primes")
```

Now, we define a custom operator:

```python
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

We have created a a function `p`, which takes an arbitrary number as input.
`p` first checks whether the input is between 0.5 and 1000.
If out-of-bounds, it returns `NaN`.
If in-bounds, it rounds it to the nearest integer, compures the corresponding prime number, and then
converts it to the same type as input.

Next, let's generate a list of primes for our test dataset.
Since we are using juliacall, we can just call `p` directly to do this:

```python
primes = {i: jl.p(i*1.0) for i in range(1, 999)}
```

Next, let's use this list of primes to create a dataset of $x, y$ pairs:

```python
import numpy as np

X = np.random.randint(0, 100, 100)[:, None]
y = [primes[3*X[i, 0] + 1] - 5 + np.random.randn()*0.001 for i in range(100)]
```

Note that we have also added a tiny bit of noise to the dataset.

Finally, let's create a PySR model, and pass the custom operator. We also need to define the sympy equivalent, which we can leave as a placeholder for now:

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

We are all set to go! Let's see if we can find the true relation:

```python
model.fit(X, y)
```

if all works out, you should be able to see the true relation (note that the constant offset might not be exactly 1, since it is allowed to round to the nearest integer).
You can get the sympy version of the best equation with:

```python
model.sympy()
```

## Custom value types

`TypeSpec` lets you specify a custom type for values flowing through your expressions.
Vectors, tensors, strings, structs, or anything else,
can be defined with a `TypeSpec` and enable PySR to search for expressions
that are compatible with that type.

Searches with a `type_spec=TypeSpec(...)` specified requires a few additional arguments
to be set:

1. There are various hooks that must be defined as part of the type spec. These tell PySR how to sample and mutate your type, and also how to unpack it into a vector of constants (for optimization).
2. You must define a custom loss function that can handle your type and produce a real-valued loss. This can be done with `elementwise_loss`, `loss_function`, or `loss_function_expression`. (You can declare the return type as, e.g., `loss_type=Float64` in your type spec, or this will be inferred automatically).
3. You must define custom operators that accept and return your type. PySR will perform a check that the operators are type-stable, and will raise an error if they are not. Add an explicit return annotation such as `::Vec2` when Julia cannot infer it.

We will look at some examples below.

### Vector-valued expression trees

This example searches for a program over two-dimensional vectors:

$$
y = \operatorname{rotate90}(x_1) + 2x_2 +
\begin{bmatrix}0.5 \\ -1.0\end{bmatrix}.
$$

Each cell of `X` and `y` contains one vector. We are not treating this as a higher-dimensional array, but rather as a normal 2D/1D array with vectors _as values_.

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

PySR will each vector in a private Julia `Vec2` type that we define below.
`scalar_constants` extracts the continuous values from this type (to be optimized by BFGS),
and `with_scalar_constants` rebuilds the constant after optimization. PySR derives
initialization, mutation, validation, counting, packing, and unpacking from
this pair:

```python
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

This is the minimal set of hooks needed. There are a few others that you could optionally define, necessary to get faster speeds, such as `init` and `mutate` (and if we want to customize the printing, `string`). PySR will try to derive these automatically from the provided `sample` hook, but it will not be as fast.

In Julia, this would give us a type definition of

```julia
struct Vec2
    data::Vector{Float64}
end
```

Once you have defined your type, you need to define the operators that accept and return this type. For example, we can define a `rotate90` operator that rotates a vector by 90 degrees, and a `double` operator that doubles the vector. We also define an `add_vectors` operator that adds two vectors together:

```python
operators = {
    1: [
        "rotate90(a::Vec2) = Vec2([-a.data[2], a.data[1]])",
        "double(a::Vec2) = Vec2(2a.data)",
    ],
    2: ["add_vectors(a::Vec2, b::Vec2) = Vec2(a.data + b.data)"],
}
```

```python
model = PySRRegressor(
    type_spec=type_spec,
    elementwise_loss="vector_loss(a::Vec2, b::Vec2) = sum(abs2, a.data - b.data)",
    niterations=40,
    populations=4,
    maxsize=10,
)

model.fit(X, y)
print(model.equations_)
```

The target can be represented as
`add_vectors(add_vectors(rotate90(x1), double(x2)), [0.5, -1.0])`, including a
learned vector-valued constant. PySR searches over both the program structure
and the two components of that constant.


<details>
<summary>String-valued expressions and discrete constants</summary>

Strings demonstrate a value type whose constants are evolved discretely rather
than optimized with BFGS. This search learns to join two transformed strings
with a sampled separator:

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

Because this type has no scalar-constant hook pair, PySR does not run BFGS on
its constants. The explicit `mutate` hook resamples separators during
evolution.

</details>

<details>
<summary>Advanced: recovering a neural network with tensor constants</summary>

`TypeSpec` can place scalar, vector, and matrix constants in one Julia value
type. The scalar-constant hooks flatten each constant for BFGS and rebuild its
original shape.

Here we recover a two-layer neural network

$$ y = W_2\operatorname{relu}(W_1x + b_1) + b_2 $$

from vector-valued data. Safe operators return an invalid value for shape
mismatches, so arbitrary expressions from the search cannot throw dimension
errors:

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

Generate training data from fixed $2\times2$ weights and two-element biases:

```python
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

Search with matrix multiplication, elementwise ReLU, and addition. BFGS is the
default constant optimizer:

```python
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

The search recovers a two-layer form such as
`nn_matmul(W2, nn_add(b, nn_relu(nn_matmul(W1, nn_add(x, c)))))`. Both biases
are absorbed into the fitted constants, through $b_1 = W_1c$ and $b_2 = W_2b$;
each displayed constant contains the fitted matrix or vector payload.

</details>

`TypeSpec` can also be used with `guesses`. Write them with the TypeSpec
constructor, for example `guesses=["add_vectors(x0, Vec2([1.0, 2.0]))"]`.
