"""Mutation configurations for :class:`PySRRegressor`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .julia_helpers import jl_array
from .julia_import import AnyValue, SymbolicRegression, jl

_LEGACY_MUTATION_PARAMETERS = (
    "weight_add_node",
    "weight_insert_node",
    "weight_delete_node",
    "weight_do_nothing",
    "weight_mutate_constant",
    "weight_mutate_operator",
    "weight_mutate_feature",
    "weight_swap_operands",
    "weight_rotate_tree",
    "weight_randomize",
    "weight_simplify",
    "weight_optimize",
    "weight_backsolve",
)


def create_mutations(mutation_pairs: Sequence[tuple[AnyValue, float]]) -> AnyValue:
    """Create a Julia vector of weighted mutations."""
    pair_type = jl.Pair[SymbolicRegression.AbstractMutation, jl.Float64]
    return jl_array(mutation_pairs, dtype=pair_type)


def convert_mutations(
    mutation_weights: Mapping[AbstractMutation, float],
) -> AnyValue:
    """Convert Python mutation weights to their Julia representation."""
    return create_mutations(
        [
            jl.Pair(mutation.julia_mutation(), weight)
            for mutation, weight in mutation_weights.items()
        ]
    )


class AbstractMutation(ABC):
    """Base class for mutation configurations."""

    @abstractmethod
    def julia_mutation(self) -> AnyValue:
        """Create the corresponding SymbolicRegression.jl mutation."""
        pass  # pragma: no cover


@dataclass(frozen=True)
class _ParameterlessMutation(AbstractMutation):
    def julia_mutation(self) -> AnyValue:
        return getattr(SymbolicRegression, type(self).__name__)()


@dataclass(frozen=True)
class ConstantMutation(AbstractMutation):
    """Perturb a constant.

    SymbolicRegression.jl default weight: ``0.0346``.

    Defaults match SymbolicRegression.jl.
    """

    perturbation_factor: float = 0.086
    probability_negate: float = 0.01

    def julia_mutation(self) -> AnyValue:
        return SymbolicRegression.ConstantMutation(
            perturbation_factor=self.perturbation_factor,
            probability_negate=self.probability_negate,
        )


@dataclass(frozen=True)
class OperatorMutation(_ParameterlessMutation):
    """Replace an operator with another operator of the same arity.

    SymbolicRegression.jl default weight: ``0.293``.
    """


@dataclass(frozen=True)
class FeatureMutation(_ParameterlessMutation):
    """Change the feature referenced by a variable node.

    SymbolicRegression.jl default weight: ``0.1``.
    """


@dataclass(frozen=True)
class SwapOperandsMutation(_ParameterlessMutation):
    """Swap the operands of a binary operator.

    SymbolicRegression.jl default weight: ``0.198``.
    """


@dataclass(frozen=True)
class AddNodeMutation(_ParameterlessMutation):
    """Append a node to the expression.

    SymbolicRegression.jl default weight: ``2.47``.
    """


@dataclass(frozen=True)
class InsertNodeMutation(_ParameterlessMutation):
    """Insert a node above an existing node.

    SymbolicRegression.jl default weight: ``0.0112``.
    """


@dataclass(frozen=True)
class DeleteNodeMutation(_ParameterlessMutation):
    """Delete a node from the expression.

    SymbolicRegression.jl default weight: ``0.870``.
    """


@dataclass(frozen=True)
class RotateTreeMutation(_ParameterlessMutation):
    """Rotate a subtree.

    SymbolicRegression.jl default weight: ``4.26``.
    """


@dataclass(frozen=True)
class BacksolveMutation(AbstractMutation):
    """Fit a replacement expression by backsolving through the expression.

    SymbolicRegression.jl default weight: ``0.0``.
    """

    max_library_size: int = 500
    max_terms: int = 8
    min_improvement: float = 1e-3
    node_attempts: int = 8

    def julia_mutation(self) -> AnyValue:
        return SymbolicRegression.BacksolveMutation(
            max_library_size=self.max_library_size,
            max_terms=self.max_terms,
            min_improvement=self.min_improvement,
            node_attempts=self.node_attempts,
        )


@dataclass(frozen=True)
class SimplifyMutation(_ParameterlessMutation):
    """Simplify constant parts of the expression.

    SymbolicRegression.jl default weight: ``0.00209``.
    """


@dataclass(frozen=True)
class RandomizeMutation(_ParameterlessMutation):
    """Replace the expression with a random expression.

    SymbolicRegression.jl default weight: ``0.000502``.
    """


@dataclass(frozen=True)
class OptimizeMutation(_ParameterlessMutation):
    """Optimize constants as a mutation.

    SymbolicRegression.jl default weight: ``0.0``.
    """


@dataclass(frozen=True)
class DoNothingMutation(_ParameterlessMutation):
    """Leave the expression unchanged.

    SymbolicRegression.jl default weight: ``0.273``.
    """
