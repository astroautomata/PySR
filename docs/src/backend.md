# Customization

## Backend selection

PySR uses the Julia backend by default:

```python
PySRRegressor(backend="julia")
```

An experimental Rust backend adapter is also available:

```python
PySRRegressor(backend="rust")
```

The Rust backend is intended for vanilla symbolic regression with builtin
operators and faster Julia-free startup. It requires an optional Python package
exposing `symbolic_regression_rs.search`.
Install it with:

```bash
pip install "pysr[rust]"
```

| Feature | `backend="julia"` | `backend="rust"` |
| --- | --- | --- |
| Builtin operators | Yes | Yes |
| Custom Julia operators/losses | Yes | No |
| Template and parametric expression specs | Yes | No |
| Operator and nested constraints | Yes | No |
| Units and dimensional constraints | Yes | No |
| Cluster managers and Julia extensions | Yes | No |

Use `backend="julia"` when you need the full PySR feature set.

## Julia backend customization

If you have explored the [options](options.md) and [PySRRegressor reference](api.md), and still haven't figured out how to specify a constraint or objective required for your problem, you might consider editing the backend.
The backend of PySR is written as a pure Julia package under the name [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl).
This package is accessed with [`juliacall`](https://github.com/JuliaPy/PythonCall.jl), which allows us to transfer objects back and forth between the Python and Julia runtimes.

PySR gives you access to everything in SymbolicRegression.jl, but there are some specific use-cases which require modifications to the backend itself.
Generally you can do this as follows:

## 1. Check out the source code

Clone a copy of the backend as well as PySR:

```bash
git clone https://github.com/astroautomata/SymbolicRegression.jl
git clone https://github.com/astroautomata/PySR
```

You may wish to check out the specific versions, which you can do with:

```bash
cd PySR
git checkout <version>

# You can see the current backend version in `pysr/juliapkg.json`
cd ../SymbolicRegression.jl
git checkout <backend_version>
```

## 2. Edit the source to your requirements

The main search code can be found in `src/SymbolicRegression.jl`.

Here are some tips:

-  The documentation for the backend is given [here](https://ai.damtp.cam.ac.uk/symbolicregression/dev/).
- Throughout the package, you will often see template functions which typically use a symbol `T` (such as in the string `where {T<:Real}`). Here, `T` is simply the datatype of the input data and stored constants, such as `Float32` or `Float64`. Writing functions in this way lets us write functions generic to types, while still having access to the specific type specified at compilation time.
- Expressions are stored as binary trees, using the `Node{T}` type, described [here](https://ai.damtp.cam.ac.uk/symbolicregression/dev/types/#SymbolicRegression.CoreModule.EquationModule.Node).
- For reference, the main loop itself is found in the `equation_search` function inside [`src/SymbolicRegression.jl`](https://github.com/astroautomata/SymbolicRegression.jl/blob/master/src/SymbolicRegression.jl).
- Parts of the code which are typically edited by users include:
    - [`src/CheckConstraints.jl`](https://github.com/astroautomata/SymbolicRegression.jl/blob/master/src/CheckConstraints.jl), particularly the function `check_constraints`. This function checks whether a given expression satisfies constraints, such as having a complexity lower than `maxsize`, and whether it contains any forbidden nestings of functions.
        - Note that all expressions, *even intermediate expressions*, must comply with constraints. Therefore, make sure that evolution can still reach your desired expression (with one mutation at a time), before setting a hard constraint. In other cases you might want to instead put in the loss function.
    - [`src/Options.jl`](https://github.com/astroautomata/SymbolicRegression.jl/blob/master/src/Options.jl), as well as the struct definition in [`src/OptionsStruct.jl`](https://github.com/astroautomata/SymbolicRegression.jl/blob/master/src/OptionsStruct.jl). This file specifies all the options used in the search: an instance of `Options` is typically available throughout every function in `SymbolicRegression.jl`. If you add new functionality to the backend, and wish to make it parameterizable (including from PySR), you should specify it in the options.

## 3. Let PySR use the modified backend

Once you have made your changes, you should edit the `pysr/juliapkg.json` file
in the PySR repository to point to this local copy.
Do this by removing the `"version"` key and adding a `"dev"` and `"path"` key:

```json
    ...
    "packages": {
        "SymbolicRegression": {
            "uuid": "8254be44-1295-4e6a-a16d-46603ac705cb",
            "dev": true,
            "path": "/path/to/SymbolicRegression.jl"
        },
    ...
```

You can then install PySR with this modified backend by running:

```bash
cd PySR
pip install .
```

For more information on `juliapkg.json`, see [`pyjuliapkg`](https://github.com/JuliaPy/pyjuliapkg).

## Additional notes

If you get comfortable enough with the backend, you might consider using the Julia package directly: the API is given on the [SymbolicRegression.jl documentation](https://ai.damtp.cam.ac.uk/symbolicregression/dev/).

If you make a change that you think could be useful to other users, don't hesitate to open a pull request on either the PySR or SymbolicRegression.jl repositories! Contributions are very appreciated.
