from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, ClassVar, NewType

import numpy as np
import pandas as pd

from .export import add_export_formats
from .julia_helpers import jl_array
from .julia_import import AnyValue, SymbolicRegression, jl

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

    Implementations may also declare the export formats they support.
    """

    @abstractmethod
    def julia_expression_spec(self) -> AnyValue:
        """The expression specification"""
        pass  # pragma: no cover

    def _julia_expression_spec_source(self, *, prototype: str | None) -> str | None:
        """Return self-contained Julia source for a TypeSpec-compatible spec.

        ``prototype`` is Julia source for one value of the generated type. PySR
        evaluates the returned source once in the fingerprinted runtime module.
        """
        return None

    def _julia_expression_spec_function_selector(self) -> str | None:
        """Return Julia source selecting the callable nested in the spec.

        PySR evaluates the source to a selector, invokes it with the expression
        spec, and binds the result under a deterministic runtime-module name so
        checkpoints and workers resolve the same callable identity.
        """
        return None

    @property
    def supports_type_spec(self) -> bool:
        return False

    def _validate_type_spec(self) -> None:
        pass

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
    def supports_type_spec(self) -> bool:
        return True

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


@dataclass
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

    combine: str
    expressions: list[str]
    variable_names: list[str]
    parameters: dict[str, int] | None = None

    _spec_cache: ClassVar[dict[tuple[str, ...], AnyValue]] = {}

    def _get_cache_key(self):
        return (
            self.combine,
            str(self.expressions),
            str(self.variable_names),
            str(self.parameters),
        )

    def julia_expression_spec(self):
        key = self._get_cache_key()
        if key not in self._spec_cache:
            self._spec_cache[key] = self._call_template_macro()
        return self._spec_cache[key]

    def _call_template_macro(self):
        return jl.seval(self._template_macro_str())

    def _template_macro_str(self, *, prototype: str | None = None) -> str:
        template_inputs = [f"expressions=({', '.join(self.expressions) + ','})"]
        if self.parameters:
            template_inputs.append(
                f"parameters=({', '.join([f'{p}={self.parameters[p]}' for p in self.parameters]) + ','})"
            )
        if prototype is not None:
            template_inputs.append(f"prototype={prototype}")
        return dedent(f"""
        @template_spec({', '.join(template_inputs) + ','}) do {', '.join(self.variable_names)}
            {self.combine}
        end
        """)

    def _julia_expression_spec_source(self, *, prototype: str | None) -> str:
        return self._template_macro_str(prototype=prototype)

    def _julia_expression_spec_function_selector(self) -> str:
        return "spec -> spec.structure.combine"

    @property
    def supports_type_spec(self) -> bool:
        return True

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
        search_output = search_output or model.julia_state_
        return _search_output_to_callable_expressions(equations, search_output, i)


class CallableJuliaExpression:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, X: np.ndarray, *args):
        raw_output = self.expression(jl_array(X.T), *args)
        return np.array(raw_output).T


def _search_output_to_callable_expressions(
    equations: pd.DataFrame,
    search_output,
    i: int | None,
) -> pd.DataFrame:
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
