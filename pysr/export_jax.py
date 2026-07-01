import numpy as np  # noqa: F401
import sympy  # type: ignore
from sympy.codegen.cfunctions import log2, log10  # type: ignore

# Special since need to reduce arguments.
MUL = 0
ADD = 1

_jnp_func_lookup = {
    sympy.Mul: MUL,
    sympy.Add: ADD,
    sympy.div: "jnp.div",
    sympy.Abs: "jnp.abs",
    sympy.sign: "jnp.sign",
    # Note: May raise error for ints.
    sympy.ceiling: "jnp.ceil",
    sympy.floor: "jnp.floor",
    sympy.log: "jnp.log",
    log2: "jnp.log2",
    log10: "jnp.log10",
    sympy.exp: "jnp.exp",
    sympy.sqrt: "jnp.sqrt",
    sympy.cos: "jnp.cos",
    sympy.acos: "jnp.acos",
    sympy.sin: "jnp.sin",
    sympy.asin: "jnp.asin",
    sympy.tan: "jnp.tan",
    sympy.atan: "jnp.atan",
    sympy.atan2: "jnp.atan2",
    # Note: Also may give NaN for complex results.
    sympy.cosh: "jnp.cosh",
    sympy.acosh: "jnp.acosh",
    sympy.sinh: "jnp.sinh",
    sympy.asinh: "jnp.asinh",
    sympy.tanh: "jnp.tanh",
    sympy.atanh: "jnp.atanh",
    sympy.Pow: "jnp.power",
    sympy.re: "jnp.real",
    sympy.im: "jnp.imag",
    sympy.arg: "jnp.angle",
    # Note: May raise error for ints and complexes
    sympy.erf: "jsp.erf",
    sympy.erfc: "jsp.erfc",
    sympy.LessThan: "jnp.less",
    sympy.GreaterThan: "jnp.greater",
    sympy.And: "jnp.logical_and",
    sympy.Or: "jnp.logical_or",
    sympy.Not: "jnp.logical_not",
    sympy.Max: "jnp.max",
    sympy.Min: "jnp.min",
    sympy.Mod: "jnp.mod",
    sympy.Heaviside: "jnp.heaviside",
    sympy.core.numbers.Half: "(lambda: 0.5)",
    sympy.core.numbers.One: "(lambda: 1.0)",
}


def sympy2jaxtext(expr, parameters, symbols_in, extra_jax_mappings=None):
    if issubclass(expr.func, sympy.Float):
        parameters.append(float(expr))
        return f"parameters[{len(parameters) - 1}]"
    elif issubclass(expr.func, sympy.Rational) or issubclass(
        expr.func, sympy.NumberSymbol
    ):
        return f"{float(expr)}"
    elif issubclass(expr.func, sympy.Integer):
        return f"{int(expr)}"
    elif issubclass(expr.func, sympy.Symbol):
        return (
            f"X[:, {[i for i in range(len(symbols_in)) if symbols_in[i] == expr][0]}]"
        )
    if extra_jax_mappings is None:
        extra_jax_mappings = {}
    try:
        _func = {**_jnp_func_lookup, **extra_jax_mappings}[expr.func]
    except KeyError:
        raise KeyError(
            f"Function {expr.func} was not found in JAX function mappings."
            "Please add it to extra_jax_mappings in the format, e.g., "
            "{sympy.sqrt: 'jnp.sqrt'}."
        )
    args = [
        sympy2jaxtext(
            arg, parameters, symbols_in, extra_jax_mappings=extra_jax_mappings
        )
        for arg in expr.args
    ]
    if _func == MUL:
        return " * ".join(["(" + arg + ")" for arg in args])
    if _func == ADD:
        return " + ".join(["(" + arg + ")" for arg in args])
    return f'{_func}({", ".join(args)})'


jax_initialized = False
jax = None
jnp = None
jsp = None


def _initialize_jax():
    global jax_initialized
    global jax
    global jnp
    global jsp

    if not jax_initialized:
        import jax as _jax
        from jax import numpy as _jnp
        from jax.scipy import special as _jsp

        jax = _jax
        jnp = _jnp
        jsp = _jsp


def sympy2jax(expression, symbols_in, selection=None, extra_jax_mappings=None):
    """Returns a function f and its parameters;
    the function takes an input matrix, and a list of arguments:
            f(X, parameters)
    where the parameters appear in the JAX equation.

    # Examples:

        Let's create a function in SymPy:
        ```python
        x, y = symbols('x y')
        cosx = 1.0 * sympy.cos(x) + 3.2 * y
        ```
        Let's get the JAX version. We pass the equation, and
        the symbols required.
        ```python
        f, params = sympy2jax(cosx, [x, y])
        ```
        The order you supply the symbols is the same order
        you should supply the features when calling
        the function `f` (shape `[nrows, nfeatures]`).
        In this case, features=2 for x and y.
        The `params` in this case will be
        `jnp.array([1.0, 3.2])`. You pass these parameters
        when calling the function, which will let you change them
        and take gradients.

        Let's generate some JAX data to pass:
        ```python
        key = random.PRNGKey(0)
        X = random.normal(key, (10, 2))
        ```

        We can call the function with:
        ```python
        f(X, params)

        #> DeviceArray([-2.6080756 ,  0.72633684, -6.7557726 , -0.2963162 ,
        #                6.6014843 ,  5.032483  , -0.810931  ,  4.2520013 ,
        #                3.5427954 , -2.7479894 ], dtype=float32)
        ```

        We can take gradients with respect
        to the parameters for each row with JAX
        gradient parameters now:
        ```python
        jac_f = jax.jacobian(f, argnums=1)
        jac_f(X, params)

        #> DeviceArray([[ 0.49364874, -0.9692889 ],
        #               [ 0.8283714 , -0.0318858 ],
        #               [-0.7447336 , -1.8784496 ],
        #               [ 0.70755106, -0.3137085 ],
        #               [ 0.944834  ,  1.767703  ],
        #               [ 0.51673377,  1.4111717 ],
        #               [ 0.87347716, -0.52637756],
        #               [ 0.8760679 ,  1.0549792 ],
        #               [ 0.9961824 ,  0.79581654],
        #               [-0.88465923, -0.5822907 ]], dtype=float32)
        ```

        We can also JIT-compile our function:
        ```python
        compiled_f = jax.jit(f)
        compiled_f(X, params)

        #> DeviceArray([-2.6080756 ,  0.72633684, -6.7557726 , -0.2963162 ,
        #                6.6014843 ,  5.032483  , -0.810931  ,  4.2520013 ,
        #                3.5427954 , -2.7479894 ], dtype=float32)
        ```
    """
    _initialize_jax()
    global jax_initialized
    global jax
    global jnp
    global jsp

    parameters = []
    functional_form_text = sympy2jaxtext(
        expression, parameters, symbols_in, extra_jax_mappings
    )
    hash_string = "A_" + str(abs(hash(str(expression) + str(symbols_in))))
    text = f"def {hash_string}(X, parameters):\n"
    if selection is not None:
        # Impose the feature selection:
        text += f"    X = X[:, {list(selection)}]\n"
    text += "    return "
    text += functional_form_text
    ldict = {}
    exec(text, globals(), ldict)
    return ldict[hash_string], jnp.array(parameters)
