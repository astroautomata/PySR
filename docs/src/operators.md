# Operators

## Pre-defined

First, note that pretty much any valid Julia function which
takes one or two scalars as input, and returns on scalar as output,
is likely to be a valid operator[^1].
A selection of these and other valid operators are stated below.

Also, note that it's a good idea to not use too many operators, since
it can exponentially increase the search space.

### Unary Operators

| Basic      | Exp/Log    | Trig      | Hyperbolic | Special   | Rounding   |
|------------|------------|-----------|------------|-----------|------------|
| `neg`      | `exp`      | `sin`     | `sinh`     | `erf`     | `round`    |
| `square`   | `log`      | `cos`     | `cosh`     | `erfc`    | `floor`    |
| `cube`     | `log10`    | `tan`     | `tanh`     | `gamma`   | `ceil`     |
| `cbrt`     | `log2`     | `asin`    | `asinh`    | `relu`    |            |
| `sqrt`     | `log1p`    | `acos`    | `acosh`    | `sinc`    |            |
| `abs`      |            | `atan`    | `atanh`    |           |            |
| `sign`     |            |           |            |           |            |
| `inv`      |            |           |            |           |            |

### Binary Operators

| Arithmetic    | Comparison | Logic    |
|--------------|------------|----------|
| `+`          | `max`      | `logical_or`[^2] |
| `-`          | `min`      | `logical_and`[^3]|
| `*`          | `>`[^4]    |                 |
| `/`          | `>=`       |                 |
| `^`          | `<`        |                 |
|              | `<=`       |                 |
|              | `cond`[^5] |                 |
|              | `mod`      |                 |

### Higher Arity Operators

| Ternary      |
|--------------|
| `clamp`      |
| `fma` / `muladd` |
| `max`        |
| `min`        |

<!--TODO: | `ifelse`[^6] | -->

Note that to use operators with arity 3 or more, you must use the `operators` parameter instead of the `*ary_operators` parameters, and pass operators as a dictionary with the arity as key:

```python
operators={
    1: ["sin"], 2: ["+", "-", "*"], 3: ["clamp"]
},
```

## Custom

Instead of passing a predefined operator as a string,
you can just define a custom function as Julia code. For example:

```python
    PySRRegressor(
        ...,
        unary_operators=["myfunction(x) = x^2"],
        binary_operators=["myotherfunction(x, y) = x^2*y"],
        extra_sympy_mappings={
            "myfunction": lambda x: x**2,
            "myotherfunction": lambda x, y: x**2 * y,
        },
    )
```


Make sure that it works with
`Float32` as a datatype (for default precision, or `Float64` if you set `precision=64`). That means you need to write `1.5f3`
instead of `1.5e3`, if you write any constant numbers, or simply convert a result to `Float64(...)`.

PySR expects that operators not throw an error for any input value over the entire real line from `-3.4e38` to `+3.4e38`.
Thus, for invalid inputs, such as negative numbers to a `sqrt` function, you may simply return a `NaN` of the same type as the input. For example,

```julia
my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)
```

would be a valid operator. The genetic algorithm
will preferentially selection expressions which avoid
any invalid values over the training dataset.

[^1]: However, you will need to define a sympy equivalent in `extra_sympy_mapping` if you want to use a function not in the above list.
[^2]: `logical_or` is equivalent to `(x, y) -> (x > 0 || y > 0) ? 1 : 0`
[^3]: `logical_and` is equivalent to `(x, y) -> (x > 0 && y > 0) ? 1 : 0`
[^4]: `>` is equivalent to `(x, y) -> x > y ? 1 : 0`
[^5]: `cond` is equivalent to `(x, y) -> x > 0 ? y : 0`

<!-- [^6]: `ifelse` is equivalent to `(x, y, z) -> x > 0 ? y : z` -->
