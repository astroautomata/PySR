from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .export_jax import sympy2jax
from .export_numpy import sympy2numpy
from .export_sympy import create_sympy_symbols, pysr2sympy
from .export_torch import sympy2torch
from .utils import ArrayLike


def add_export_formats(
    output: pd.DataFrame,
    *,
    feature_names_in: ArrayLike[str],
    selection_mask: NDArray[np.bool_] | None = None,
    extra_sympy_mappings: dict[str, Callable] | None = None,
    extra_torch_mappings: dict[Callable, Callable] | None = None,
    output_torch_format: bool = False,
    extra_jax_mappings: dict[Callable, str] | None = None,
    output_jax_format: bool = False,
) -> pd.DataFrame:
    """Create export formats for an equations dataframe.

    Returns a new dataframe containing only the exported formats.
    """
    output = copy.deepcopy(output)

    sympy_format = []
    lambda_format = []
    jax_format = []
    torch_format = []

    for _, eqn_row in output.iterrows():
        eqn = pysr2sympy(
            eqn_row["equation"],
            feature_names_in=feature_names_in,
            extra_sympy_mappings=extra_sympy_mappings,
        )
        sympy_format.append(eqn)

        # NumPy:
        sympy_symbols = create_sympy_symbols(feature_names_in)
        lambda_format.append(
            sympy2numpy(
                eqn,
                sympy_symbols,
                selection=selection_mask,
            )
        )

        # JAX:
        if output_jax_format:
            func, params = sympy2jax(
                eqn,
                sympy_symbols,
                selection=selection_mask,
                extra_jax_mappings=extra_jax_mappings,
            )
            jax_format.append({"callable": func, "parameters": params})

        # Torch:
        if output_torch_format:
            module = sympy2torch(
                eqn,
                sympy_symbols,
                selection=selection_mask,
                extra_torch_mappings=extra_torch_mappings,
            )
            torch_format.append(module)

    exports = pd.DataFrame(
        {
            "sympy_format": sympy_format,
            "lambda_format": lambda_format,
        },
        index=output.index,
    )

    if output_jax_format:
        exports["jax_format"] = pd.Series(jax_format, index=output.index)
    if output_torch_format:
        exports["torch_format"] = pd.Series(torch_format, index=output.index)

    return exports
