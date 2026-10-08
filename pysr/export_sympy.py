"""Define utilities to export to sympy"""

from __future__ import annotations

from collections.abc import Callable

import sympy  # type: ignore
from sympy import sympify
from sympy.codegen.cfunctions import log1p, log2, log10  # type: ignore

from .utils import ArrayLike

# Julia's `round` sends halves to the even neighbour, as do NumPy, PyTorch, and
# JAX. Keeping it as an opaque function preserves that; `lambdify` maps it to
# NumPy's `round`.
round_half_even = sympy.Function("round")

# SymPy's parser passes `evaluate=False` to the functions it knows, such as
# `sqrt` and `log`, so mappings with those names must accept the keyword.
# Otherwise `pysr2sympy` falls back to parsing with simplification enabled.
sympy_mappings = {
    "div": lambda x, y: x / y,
    "inv": lambda x: 1 / x,
    "mult": lambda x, y: x * y,
    "sqrt": lambda x, **kwargs: sympy.sqrt(x, **kwargs),
    "sqrt_abs": lambda x: sympy.sqrt(abs(x)),
    "cbrt": lambda x, **kwargs: sympy.sign(x) * sympy.cbrt(abs(x), **kwargs),
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "plus": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "pow": lambda x, y: x**y,
    "pow_abs": lambda x, y: abs(x) ** y,
    "cos": sympy.cos,
    "sin": sympy.sin,
    "tan": sympy.tan,
    "cosh": sympy.cosh,
    "sinh": sympy.sinh,
    "tanh": sympy.tanh,
    "exp": sympy.exp,
    "acos": sympy.acos,
    "asin": sympy.asin,
    "atan": sympy.atan,
    "acosh": lambda x, **kwargs: sympy.acosh(x, **kwargs),
    "acosh_abs": lambda x: sympy.acosh(abs(x) + 1),
    "asinh": sympy.asinh,
    # Like SymbolicRegression's `safe_atanh`: NaN outside [-1, 1]
    "atanh": sympy.atanh,
    "atanh_clip": lambda x: sympy.atanh(sympy.Mod(x + 1, 2) - sympy.S(1)),
    "abs": abs,
    "mod": sympy.Mod,
    "erf": sympy.erf,
    "erfc": sympy.erfc,
    "log": lambda x, **kwargs: sympy.log(x, **kwargs),
    "log10": lambda x: log10(x),
    "log2": lambda x: log2(x),
    # Accurate for small `x`, unlike `log(x + 1)`
    "log1p": log1p,
    "log_abs": lambda x: sympy.log(abs(x)),
    "log10_abs": lambda x: sympy.log(abs(x), 10),
    "log2_abs": lambda x: sympy.log(abs(x), 2),
    "log1p_abs": lambda x: sympy.log(abs(x) + 1),
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "sign": sympy.sign,
    "gamma": sympy.gamma,
    "round": round_half_even,
    "max": lambda *args: sympy.Max(*args),
    "min": lambda *args: sympy.Min(*args),
    "greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True)),
    "less": lambda x, y: sympy.Piecewise((1.0, x < y), (0.0, True)),
    "greater_equal": lambda x, y: sympy.Piecewise((1.0, x >= y), (0.0, True)),
    "less_equal": lambda x, y: sympy.Piecewise((1.0, x <= y), (0.0, True)),
    "cond": lambda x, y: sympy.Piecewise((y, x > 0), (0.0, True)),
    "logical_or": lambda x, y: sympy.Piecewise((1.0, (x > 0) | (y > 0)), (0.0, True)),
    "logical_and": lambda x, y: sympy.Piecewise((1.0, (x > 0) & (y > 0)), (0.0, True)),
    "relu": lambda x: sympy.Piecewise((0.0, x < 0), (x, True)),
    "fma": lambda x, y, z: x * y + z,
    "muladd": lambda x, y, z: x * y + z,
    "clamp": lambda x, min_val, max_val: sympy.Piecewise(
        (min_val, x < min_val), (max_val, x > max_val), (x, True)
    ),
}


def create_sympy_symbols_map(
    feature_names_in: ArrayLike[str],
) -> dict[str, sympy.Symbol]:
    return {variable: sympy.Symbol(variable) for variable in feature_names_in}


def create_sympy_symbols(
    feature_names_in: ArrayLike[str],
) -> list[sympy.Symbol]:
    return [sympy.Symbol(variable) for variable in feature_names_in]


def pysr2sympy(
    equation: str | float | int,
    *,
    feature_names_in: ArrayLike[str] | None = None,
    extra_sympy_mappings: dict[str, Callable] | None = None,
):
    if feature_names_in is None:
        feature_names_in = []
    local_sympy_mappings = {
        **create_sympy_symbols_map(feature_names_in),
        **sympy_mappings,
        **(extra_sympy_mappings if extra_sympy_mappings is not None else {}),
    }

    try:
        return sympify(equation, locals=local_sympy_mappings, evaluate=False)
    except TypeError as e:
        if "got an unexpected keyword argument 'evaluate'" in str(e):
            return sympify(equation, locals=local_sympy_mappings)
        raise TypeError(f"Error processing equation '{equation}'") from e


def assert_valid_sympy_symbol(var_name: str) -> None:
    if var_name in sympy_mappings or var_name in sympy.__dict__.keys():
        raise ValueError(f"Variable name {var_name} is already a function name.")
