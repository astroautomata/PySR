from __future__ import annotations

import copy
import textwrap
import warnings
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NewType, overload

import numpy as np
import pandas as pd

from .export import add_export_formats
from .julia_helpers import jl_array
from .julia_import import AnyValue, SymbolicRegression, jl
from .utils import ArrayLike

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

# For type checking purposes
if TYPE_CHECKING:
    from .sr import PySRRegressor  # pragma: no cover

    PySRRegressor: TypeAlias = PySRRegressor  # pragma: no cover
else:
    PySRRegressor = NewType("PySRRegressor", Any)


class AbstractExpressionSpec(ABC):
    """Abstract base class describing expression types.

    This basically just holds the options for the expression type,
    as well as explains how to parse and evaluate them.

    All expression types must implement:

    1. julia_expression_spec(): The actual expression specification, returned as a Julia object.
        This will get passed as `expression_spec` in `SymbolicRegression.Options`.
    2. create_exports(), which will be used to create the exports of the equations, such as
        the executable format, the SymPy format, etc.

    It may also optionally implement:

    - supports_sympy, supports_torch, supports_jax, supports_latex: Whether this expression type supports the corresponding export format.
    """

    @abstractmethod
    def julia_expression_spec(self) -> AnyValue:
        """The expression specification"""
        pass  # pragma: no cover

    @abstractmethod
    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
        i: int | None = None,
    ) -> pd.DataFrame:
        """Create additional columns in the equations dataframe."""
        pass  # pragma: no cover

    @property
    def evaluates_in_julia(self) -> bool:
        return False

    @property
    def supports_sympy(self) -> bool:
        return False

    @property
    def supports_torch(self) -> bool:
        return False

    @property
    def supports_jax(self) -> bool:
        return False

    @property
    def supports_latex(self) -> bool:
        return False


class ExpressionSpec(AbstractExpressionSpec):
    """The default expression specification, with no special behavior."""

    def julia_expression_spec(self):
        return SymbolicRegression.ExpressionSpec()

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
        i: int | None = None,
    ):
        return add_export_formats(
            equations,
            feature_names_in=model.feature_names_in_,
            selection_mask=model.selection_mask_,
            extra_sympy_mappings=model.extra_sympy_mappings,
            extra_torch_mappings=model.extra_torch_mappings,
            output_jax_format=model.output_jax_format,
            extra_jax_mappings=model.extra_jax_mappings,
            output_torch_format=model.output_torch_format,
        )

    @property
    def supports_sympy(self):
        return True

    @property
    def supports_torch(self):
        return True

    @property
    def supports_jax(self):
        return True

    @property
    def supports_latex(self):
        return True


class TemplateExpressionSpec(AbstractExpressionSpec):
    """Spec for templated expressions.

    This class allows you to specify how multiple sub-expressions should be combined
    in a structured way, with constraints on which variables each sub-expression can use.
    Pass this to PySRRegressor with the `expression_spec` argument.

    Parameters
    ----------
    combine : str
        Julia function string that defines how the sub-expressions are combined.
        For example: "sin(f(x1, x2)) + g(x3)^2" would constrain f to use x1,x2 and g to use x3.
    expressions : list[str]
        List of symbols representing the inner expressions (e.g., ["f", "g"]).
        These will be used as keys in the template structure.
    variable_names : list[str]
        List of variable names that will be used in the combine function.
    parameters : dict[str, int], optional
        Dictionary mapping parameter names to their lengths. For example, {"p1": 2, "p2": 1}
        means p1 is a vector of length 2 and p2 is a vector of length 1. These parameters
        will be optimized during the search.

    Examples
    --------
    ```python
    # Create template that combines f(x1, x2) and g(x3):
    expression_spec = TemplateExpressionSpec(
        expressions=["f", "g"],
        variable_names=["x1", "x2", "x3"],
        combine="sin(f(x1, x2)) + g(x3)^2",
    )

    # With parameters:
    expression_spec = TemplateExpressionSpec(
        expressions=["f", "g"],
        variable_names=["x1", "x2", "x3"],
        parameters={"p1": 2, "p2": 1},
        combine="p1[1] * sin(f(x1, x2)) + p1[2] * g(x3) + p2[1]",
    )

    # Use in PySRRegressor:
    model = PySRRegressor(
        expression_spec=expression_spec
    )
    ```

    Notes
    -----
    You can also use differential operators in the template with `D(f, 1)(x)` to take
    the derivative of f with respect to its first argument, evaluated at x.
    """

    _spec_cache: dict[tuple[str, ...], AnyValue] = {}

    @overload
    def __init__(
        self,
        function_symbols: list[str],
        combine: str,
        num_features: dict[str, int] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        combine: str,
        *,
        expressions: list[str],
        variable_names: list[str],
        parameters: dict[str, int] | None = None,
    ) -> None: ...

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Handle both formats with combine as explicit parameter"""
        self._old_format = len(args) >= 2 or "function_symbols" in kwargs

        if self._old_format:
            self._load_old_format(*args, **kwargs)
        else:
            self._load_new_format(*args, **kwargs)

    def _load_old_format(
        self,
        function_symbols: list[str],
        combine: str,
        num_features: dict[str, int] | None = None,
    ):
        self.function_symbols = function_symbols
        self.combine = combine
        self.num_features = num_features
        # TODO: warn about old format after some versions

    def _load_new_format(
        self,
        combine: str,
        *,
        expressions: list[str],
        variable_names: list[str],
        parameters: dict[str, int] | None = None,
    ):
        self.combine = combine
        self.expressions = expressions
        self.variable_names = variable_names
        self.parameters = parameters

    def _get_cache_key(self):
        if self._old_format:
            return (
                "old",
                str(self.function_symbols),
                self.combine,
                str(self.num_features),
            )
        else:
            return (
                "new",
                self.combine,
                str(self.expressions),
                str(self.variable_names),
                str(self.parameters),
            )

    def julia_expression_spec(self):
        key = self._get_cache_key()
        if key in self._spec_cache:
            return self._spec_cache[key]

        if self._old_format:
            result = SymbolicRegression.TemplateExpressionSpec(
                structure=self.julia_expression_options().structure
            )
        else:
            result = self._call_template_macro()

        self._spec_cache[key] = result
        return result

    def _call_template_macro(self):
        return jl.seval(self._template_macro_str())

    def _template_macro_str(self):
        template_inputs = [f"expressions=({', '.join(self.expressions) + ','})"]
        if self.parameters:
            template_inputs.append(
                f"parameters=({', '.join([f'{p}={self.parameters[p]}' for p in self.parameters]) + ','})"
            )
        return dedent(f"""
        @template_spec({', '.join(template_inputs) + ','}) do {', '.join(self.variable_names)}
            {self.combine}
        end
        """)

    def julia_expression_options(self):
        f_combine = jl.seval(self.combine)
        creator = jl.seval("""
        function _pysr_create_template_structure(
            @nospecialize(function_symbols::AbstractVector),
            @nospecialize(combine::Function),
            @nospecialize(num_features::Union{Nothing,AbstractDict})
        )
            tuple_symbol = (map(Symbol, function_symbols)..., )
            num_features = if num_features === nothing
                nothing
            else
                NamedTuple(Symbol(k) => v for (k, v) in num_features)
            end
            structure = SymbolicRegression.TemplateStructure{tuple_symbol}(combine, num_features)
            return (; structure)
        end
        """)
        return creator(self.function_symbols, f_combine, self.num_features)

    @property
    def evaluates_in_julia(self):
        return True

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
        i: int | None = None,
    ) -> pd.DataFrame:
        # We try to load the raw julia state from a saved binary stream
        # if not provided.
        search_output = search_output or model.julia_state_
        return _search_output_to_callable_expressions(equations, search_output, i)


def parametric_expression_deprecation_warning(
    max_parameters: int, variable_names: ArrayLike[str]
):
    function_name = "f"
    var_names = list(variable_names)
    message = dedent(f"""
        ParametricExpressionSpec is deprecated. you should switch to TemplateExpressionSpec with explicit parameters indexed by category.

        Since you have `max_parameters={max_parameters}` and `variable_names=[{", ".join(f'"{v}"' for v in var_names)}]`, you could migrate like this:

            n_categories = len(np.unique(category))  # count the number of parameters required
            expression_spec = TemplateExpressionSpec(
                expressions=["{function_name}"],
                variable_names=[{", ".join(f'"{v}"' for v in var_names + ["category"])}],
                parameters={{{", ".join(f'"p{i+1}": n_categories' for i in range(max_parameters))}}},
                combine="{function_name}({', '.join(var_names + [f'p{i+1}[category]' for i in range(max_parameters)])})",
            )
            X = np.column_stack([X, category])       # add the category column

        Finally, do not pass `category` when calling .fit().
    """).strip()
    wrapped = "\n".join(textwrap.fill(line, 88) for line in message.splitlines())
    warnings.warn(wrapped, FutureWarning, stacklevel=3)


class ParametricExpressionSpec(AbstractExpressionSpec):
    """Spec for parametric expressions that vary by category.

    **This is deprecated in favor of the `TemplateExpressionSpec` class,
    which now supports parameters indexed by category.**

    This class allows you to specify expressions with parameters that vary across different
    categories in your dataset. The expression structure remains the same, but parameters
    are optimized separately for each category.

    Parameters
    ----------
    max_parameters : int
        Maximum number of parameters that can appear in the expression. Each parameter
        will take on different values for each category in the data.

    Examples
    --------
    For example, if we want to allow for a model with up to 2 parameters (each category
    can have a different value for these parameters), we can use:

    ```python
    model = PySRRegressor(
        expression_spec=ParametricExpressionSpec(max_parameters=2),
        binary_operators=["+", "*"],
        unary_operators=["sin"]
    )
    model.fit(X, y, category=category)
    ```
    """

    def __init__(self, max_parameters: int):
        self.max_parameters = max_parameters

    def julia_expression_spec(self):
        return SymbolicRegression.ParametricExpressionSpec(
            max_parameters=self.max_parameters, warn=False
        )

    @property
    def evaluates_in_julia(self):
        return True

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
        i: int | None = None,
    ):
        search_output = search_output or model.julia_state_
        return _search_output_to_callable_expressions(equations, search_output, i)


class CallableJuliaExpression:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, X: np.ndarray, *args):
        raw_output = self.expression(jl_array(X.T), *args)
        return np.array(raw_output).T


def _search_output_to_callable_expressions(
    equations: pd.DataFrame, search_output, i: int | None
) -> pd.DataFrame:
    equations = copy.deepcopy(equations)
    _, all_out_hof = search_output
    out_hof = all_out_hof[i] if i is not None else all_out_hof
    expressions = []
    callables = []

    for _, row in equations.iterrows():
        curComplexity = row["complexity"]
        expression = out_hof.members[curComplexity - 1].tree
        expressions.append(expression)
        callables.append(CallableJuliaExpression(expression))

    df = pd.DataFrame(
        {"julia_expression": expressions, "lambda_format": callables},
        index=equations.index,
    )
    return df
