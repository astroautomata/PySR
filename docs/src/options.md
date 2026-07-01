# Features and Options

Some configurable features and options in `PySR` which you
may find useful include:

- [Selecting from the accuracy-complexity curve](#model-selection)
- [Operators](#operators)
- [Number of outer search iterations](#iterations)
- [Number of inner search iterations](#cycles-per-iteration)
- [Multi-processing](#processors)
- [Populations](#populations)
- [Data weighting](#weighted-data)
- [Max complexity and depth](#max-size)
- [Mini-batching](#batching)
- [Variable names](#variable-names)
- [Constraining use of operators](#constraining-use-of-operators)
- [Custom complexities](#custom-complexity)
- [LaTeX and SymPy](#latex-and-sympy)
- [Exporting to numpy, pytorch, and jax](#exporting-to-numpy-pytorch-and-jax)
- [Loss functions](#loss)
- [Model loading](#model-loading)

These are described below.
Also check out the [tuning page](tuning.md) for workflow tips.

The program will output a pandas DataFrame containing the equations
to `PySRRegressor.equations` containing the loss value
and complexity.

It will also dump to a csv
at the end of every iteration,
which is `.hall_of_fame_{date_time}.csv` by default.
It also prints the equations to stdout.

## Model selection

By default, `PySRRegressor` uses `model_selection='best'`
which selects an equation from `PySRRegressor.equations_` using
a combination of accuracy and complexity.
You can also select `model_selection='accuracy'`.

By printing a model (i.e., `print(model)`), you can see
the equation selection with the arrow shown in the `pick` column.

## Operators

A list of operators can be found on the [operators page](operators.md).
One can define custom operators in Julia by passing a string:

```python
PySRRegressor(niterations=100,
    binary_operators=["mult", "plus", "special(x, y) = x^2 + y"],
    extra_sympy_mappings={'special': lambda x, y: x**2 + y},
    unary_operators=["cos"])
```

Now, the symbolic regression code can search using this `special` function
that squares its left argument and adds it to its right. Make sure
all passed functions are valid Julia code, and take one (unary)
or two (binary) float32 scalars as input, and output a float32. This means if you
write any real constants in your operator, like `2.5`, you have to write them
instead as `2.5f0`, which defines it as `Float32`.
Operators are automatically vectorized.

One should also define `extra_sympy_mappings`,
so that the SymPy code can understand the output equation from Julia,
when constructing a useable function. This step is optional, but
is necessary for the `lambda_format` to work.

## Iterations

This is the total number of generations that `pysr` will run for.
I usually set this to a large number, and exit when I am satisfied
with the equations.

## Cycles per iteration

Each cycle considers every 10-equation subsample (re-sampled for each individual 10,
unless `fast_cycle` is set in which case the subsamples are separate groups of equations)
a single time, producing one mutated equation for each.
The parameter `ncycles_per_iteration` defines how many times this
occurs before the equations are compared to the hall of fame,
and new equations are migrated from the hall of fame, or from other populations.
It also controls how slowly annealing occurs. You may find that increasing
`ncycles_per_iteration` results in a higher cycles-per-second, as the head
worker needs to reduce and distribute new equations less often, and also increases
diversity. But at the same
time, a smaller number it might be that migrating equations from the hall of fame helps
each population stay closer to the best current equations.

## Processors

One can adjust the number of workers used by Julia with the
`procs` option. You should set this equal to the number of cores
you want `pysr` to use.

## Populations

By default, `populations=15`, but you can set a different
number of populations with this option.
More populations may increase
the diversity of equations discovered, though will take longer to train.
However, it is usually more efficient to have `populations>procs`,
as there are multiple populations running
on each core.

## Weighted data

Here, we assign weights to each row of data
using inverse uncertainty squared. We also use 10 processes for the search
instead of the default.

```python
sigma = ...
weights = 1/sigma**2

model = PySRRegressor(procs=10)
model.fit(X, y, weights=weights)
```

## Max size

`maxsize` controls the maximum size of equation (number of operators,
constants, variables). `maxdepth` is by default not used, but can be set
to control the maximum depth of an equation. These will make processing
faster, as longer equations take longer to test.

One can warm up the maxsize from a small number to encourage
PySR to start simple, by using the `warmupMaxsize` argument.
This specifies that maxsize increases every `warmupMaxsize`.

## Batching

One can turn on mini-batching, with the `batching` flag,
and control the batch size with `batch_size`. This will make
evolution faster for large datasets. Equations are still evaluated
on the entire dataset at the end of each iteration to compare to the hall
of fame, but only on a random subset during mutations and annealing.

## Variable Names

You can pass a list of strings naming each column of `X` with
`variable_names`. Alternatively, you can pass `X` as a pandas dataframe
and the columns will be used as variable names. Make sure only
alphabetical characters and `_` are used in these names.

## Constraining use of operators

One can limit the complexity of specific operators with the `constraints` parameter.
There is a "maxsize" parameter to PySR, but there is also an operator-level
"constraints" parameter. One supplies a dict, like so:

```python
constraints={'pow': (-1, 1), 'mult': (3, 3), 'cos': 5}
```

What this says is that: a power law $x^y$ can have an expression of arbitrary (-1) complexity in the x, but only complexity 1 (e.g., a constant or variable) in the y. So $(x_0 + 3)^{5.5}$ is allowed, but $5.5^{x_0 + 3}$ is not.
I find this helps a lot for getting more interpretable equations.
The other terms say that each multiplication can only have sub-expressions
of up to complexity 3 (e.g., $5.0 + x_2$) in each side, and cosine can only operate on
expressions of complexity 5 (e.g., $5.0 + x_2 exp(x_3)$).

## Custom complexity

By default, all operators, constants, and instances of variables
have a complexity of 1. The sum of the complexities of all terms
is the total complexity of an expression.
You may change this by configuring the options:

- `complexity_of_operators` - pass a dictionary of `<str>: <int>` pairs
  to change the complexity of each operator. If an operator is not
  specified, it will have the default complexity of 1.
- `complexity_of_constants` - supplying an integer will make all constants
  have that complexity.
- `complexity_of_variables` - supplying an integer will make all variables
  have that complexity.

## LaTeX and SymPy

After running `model.fit(...)`, you can look at
`model.equations` which is a pandas dataframe.
The `sympy_format` column gives sympy equations,
and the `lambda_format` gives callable functions.
You can optionally pass a pandas dataframe to the callable function,
if you called `.fit` on a pandas dataframe as well.

There are also some helper functions for doing this quickly.

- `model.latex()` will generate a TeX formatted output of your equation.
  - `model.latex_table(indices=[2, 5, 8])` will generate a formatted LaTeX table including all the specified equations.
- `model.sympy()` will return the SymPy representation.
- `model.jax()` will return a callable JAX function combined with parameters (see below)
- `model.pytorch()` will return a PyTorch model (see below).

## Exporting to numpy, pytorch, and jax

By default, the dataframe of equations will contain columns
with the identifier `lambda_format`.
These are simple functions which correspond to the equation, but executed
with numpy functions.
You can pass your `X` matrix to these functions
just as you did to the `model.fit` call. Thus, this allows
you to numerically evaluate the equations over different output.

Calling `model.predict` will execute the `lambda_format` of
the best equation, and return the result. If you selected
`model_selection="best"`, this will use an equation that combines
accuracy with simplicity. For `model_selection="accuracy"`, this will just
look at accuracy.

One can do the same thing for PyTorch, which uses code
from [sympytorch](https://github.com/patrick-kidger/sympytorch),
and for JAX, which uses code from
[sympy2jax](https://github.com/MilesCranmer/sympy2jax).

Calling `model.pytorch()` will return
a PyTorch module which runs the equation, using PyTorch functions,
over `X` (as a PyTorch tensor). This is differentiable, and the
parameters of this PyTorch module correspond to the learned parameters
in the equation, and are trainable.

```python
torch_model = model.pytorch()
torch_model(X)
```

**Warning: If you are using custom operators, you must define `extra_torch_mappings` or `extra_jax_mappings` (both are `dict` of callables) to provide an equivalent definition of the functions.** (At any time you can set these parameters or any others with `model.set_params`.)

For JAX, you can equivalently call `model.jax()`
This will return a dictionary containing a `'callable'` (a JAX function),
and `'parameters'` (a list of parameters in the equation).
You can execute this function with:

```python
jax_model = model.jax()
jax_model['callable'](X, jax_model['parameters'])
```

Since the parameter list is a jax array, this therefore lets you also
train the parameters within JAX (and is differentiable).

## `loss`

The default loss is mean-square error, and weighted mean-square error.
One can pass an arbitrary Julia string to define a custom loss, using,
e.g., `elementwise_loss="myloss(x, y) = abs(x - y)^1.5"`. For more details,
see the
[Losses](https://milescranmer.github.io/SymbolicRegression.jl/dev/losses/)
page for SymbolicRegression.jl.

Here are some additional examples:

abs(x-y) loss

```python
PySRRegressor(..., elementwise_loss="f(x, y) = abs(x - y)^1.5")
```

Note that the function name doesn't matter:

```python
PySRRegressor(..., elementwise_loss="loss(x, y) = abs(x * y)")
```

With weights:

```python
model = PySRRegressor(..., elementwise_loss="myloss(x, y, w) = w * abs(x - y)")
model.fit(..., weights=weights)
```

Weights can be used in arbitrary ways:

```python
model = PySRRegressor(..., weights=weights, elementwise_loss="myloss(x, y, w) = abs(x - y)^2/w^2")
model.fit(..., weights=weights)
```

Built-in loss (faster) (see [losses](https://ai.damtp.cam.ac.uk/symbolicregression/dev/losses/)).
This one computes the L3 norm:

```python
PySRRegressor(..., elementwise_loss="LPDistLoss{3}()")
```

Can also uses these losses for weighted (weighted-average):

```python
model = PySRRegressor(..., weights=weights, elementwise_loss="LPDistLoss{3}()")
model.fit(..., weights=weights)
```

## Model loading

PySR will automatically save a pickle file of the model state
when you call `model.fit`, once before the search starts,
and again after the search finishes. The filename will
have the same base name as the input file, but with a `.pkl` extension.
You can load the saved model state with:

```python
model = PySRRegressor.from_file(pickle_filename)
```

If you have a long-running job and would like to load the model
before completion, you can also do this. In this case, the model
loading will use the `csv` file to load the equations, since the
`csv` file is continually updated during the search. Once
the search completes, the model including its equations will
be saved to the pickle file, overwriting the existing version.
