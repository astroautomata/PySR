"""Watch the search re-allocate its own mutation budget.

`AdaptiveMutationWeightsPlugin` is on by default: per mutation kind it counts attempts
and strictly improving children, and scales that kind's weight by the learned ratio.
SymbolicRegression.jl logs none of those counters, so this example puts the shipped
plugin inside a plugin of its own that forwards every hook to it and reads its state
on the way out, then prints where the budget ended up.
"""

from dataclasses import dataclass, field

import numpy as np

from pysr import PySRRegressor
from pysr.julia_import import AnyValue, jl
from pysr.plugins import (
    AbstractPlugin,
    AdaptiveMutationWeightsPlugin,
    AdaptiveParsimonyPlugin,
    SimulatedAnnealingPlugin,
)

_rng = np.random.default_rng(20260817)
X = _rng.uniform(-3.0, 3.0, size=(200, 5))
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
VARIABLE_NAMES = None

# `DocsTracedAdaptive` holds a real `AdaptiveMutationWeightsPlugin` and forwards
# `init_plugin_state`, `on_mutation_end!` and `condition_mutation_weights!` to it
# unchanged, so the adaptation arithmetic during the run is the library's own. The
# wrapper only counts draws per mutation kind and snapshots the plugin's own
# `multipliers` vector as it moves.
_TRACER_JL = """
if !isdefined(Main, :DocsTracedAdaptive)

const DocsAMW = SymbolicRegression.AdaptiveMutationWeightsModule

struct DocsTracedAdaptive <: SymbolicRegression.AbstractPlugin
    inner::DocsAMW.AdaptiveMutationWeightsPlugin
    stride::Int
end

DocsTracedAdaptive(; stride::Int=1000, kws...) =
    DocsTracedAdaptive(DocsAMW.AdaptiveMutationWeightsPlugin(; kws...), stride)

mutable struct DocsTracedState
    inner::DocsAMW.AdaptiveMutationWeightsState
    draws::Vector{Float64}
    seen::Int
    snapshots::Vector{Tuple{Int,Vector{Float64}}}
    registered::Bool
end

const DOCS_TRACED_STATES = DocsTracedState[]

function SymbolicRegression.init_plugin_state(p::DocsTracedAdaptive, options, dataset)
    inner = SymbolicRegression.init_plugin_state(p.inner, options, dataset)
    n = length(options.mutations)
    return DocsTracedState(inner, zeros(n), 0, Tuple{Int,Vector{Float64}}[], false)
end

function SymbolicRegression.on_mutation_end!(
    s::DocsTracedState,
    p::DocsTracedAdaptive,
    mutation::SymbolicRegression.AbstractMutation,
    event::SymbolicRegression.MutationEvent,
    dataset,
    options::SymbolicRegression.AbstractOptions,
)
    if !s.registered
        push!(DOCS_TRACED_STATES, s)
        s.registered = true
    end
    s.draws[event.mutation_idx] += 1.0
    s.seen += 1
    SymbolicRegression.on_mutation_end!(s.inner, p.inner, mutation, event, dataset, options)
    if s.seen % p.stride == 0
        push!(s.snapshots, (s.seen, copy(s.inner.multipliers)))
    end
    return nothing
end

function SymbolicRegression.condition_mutation_weights!(
    weights::AbstractVector,
    s::DocsTracedState,
    p::DocsTracedAdaptive,
    member,
    options::SymbolicRegression.AbstractOptions,
    curmaxsize,
    nfeatures,
)
    return SymbolicRegression.condition_mutation_weights!(
        weights, s.inner, p.inner, member, options, curmaxsize, nfeatures
    )
end

end
"""


@dataclass(frozen=True)
class TracedAdaptiveMutationWeights(AbstractPlugin):
    """`AdaptiveMutationWeightsPlugin`, wrapped so its learned state can be read."""

    stride: int = 1000
    inner: AdaptiveMutationWeightsPlugin = field(
        default_factory=AdaptiveMutationWeightsPlugin
    )

    def julia_plugin(self) -> AnyValue:
        jl.seval(_TRACER_JL)
        # PySR calls this once per `fit`, while building that fit's options.
        jl.seval("empty!(DOCS_TRACED_STATES)")
        return jl.DocsTracedAdaptive(
            stride=self.stride,
            smoothing=self.inner.smoothing,
            floor=self.inner.floor,
            reward=jl.Symbol(self.inner.reward),
        )


TRACER = TracedAdaptiveMutationWeights()

MODEL_KWARGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    maxsize=14,
    niterations=300,
    # The plugin keeps one set of counters per population, so a single population is
    # what makes the printed numbers a run rather than an average over runs.
    populations=1,
    # The adaptive plugin is a *default* plugin, so it has to be replaced rather than
    # added: an extra plugin of a new type leaves the shipped one in place and the
    # search adapts twice. `alpha=3.17` is PySR's own default, asserted in `main`.
    plugins=[],
    default_plugins=[
        SimulatedAnnealingPlugin(alpha=3.17),
        AdaptiveParsimonyPlugin(),
        TRACER,
    ],
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


def check(model):
    """The target recovered to numerical precision."""
    return bool(model.equations_["loss"].min() < 1e-12)


def _states():
    return int(jl.seval("length(DOCS_TRACED_STATES)"))


def _pooled(expr, n):
    """Sum a per-state vector over every population's state."""
    total = [0.0] * n
    for s in range(1, _states() + 1):
        for i, v in enumerate(jl.seval(f"DOCS_TRACED_STATES[{s}].{expr}")):
            total[i] += float(v)
    return total


def mutation_statistics(model):
    """Per-kind counters read out of the plugin's own state after a fit."""
    jl.DOCS_TRACED_OPTIONS = model.julia_options_
    names = [
        str(v)
        for v in jl.seval(
            "[string(typeof(first(p))) for p in DOCS_TRACED_OPTIONS.mutations]"
        )
    ]
    base = [
        float(v)
        for v in jl.seval("[Float64(last(p)) for p in DOCS_TRACED_OPTIONS.mutations]")
    ]
    active = [bool(v) for v in jl.seval("DOCS_TRACED_STATES[1].inner.active")]
    n = len(names)
    n_states = _states()

    draws = _pooled("draws", n)
    attempts = _pooled("inner.attempts", n)
    successes = _pooled("inner.successes", n)
    multipliers = [m / n_states for m in _pooled("inner.multipliers", n)]
    strength = float(
        jl.seval("DocsAMW.AdaptiveMutationWeightsPlugin().adaptation_strength")
    )

    # The kinds the run actually drew from: nonzero base weight, and legal for a plain
    # expression. Effective weight is base * multiplier ** adaptation_strength.
    drawn = [i for i in range(n) if draws[i] > 0]
    eff = {i: base[i] * multipliers[i] ** strength for i in drawn}
    base_total = sum(base[i] for i in drawn)
    eff_total = sum(eff.values())
    draw_total = sum(draws[i] for i in drawn)

    rows = [
        {
            "mutation": names[i],
            "index": i + 1,
            "adapts": active[i],
            "base_weight": base[i],
            "draws": int(draws[i]),
            "attempts": int(attempts[i]),
            "successes": int(successes[i]),
            # The quantity the plugin scores with: a Laplace-smoothed improvement rate.
            "rate": (successes[i] + 1.0) / (attempts[i] + 2.0) if active[i] else None,
            "multiplier": multipliers[i],
            "share_shipped": base[i] / base_total,
            "share_learned": eff[i] / eff_total,
            "share_drawn": draws[i] / draw_total,
        }
        for i in drawn
    ]
    rows.sort(key=lambda r: -r["multiplier"])
    meta = {
        "n_states": n_states,
        "n_mutations": int(sum(draws)),
        "adaptation_strength": strength,
        "plugins": [
            str(v)
            for v in jl.seval(
                "[string(typeof(p)) for p in DOCS_TRACED_OPTIONS.plugins]"
            )
        ],
    }
    return rows, meta


def multiplier_trajectory(rows, n_points=6):
    """The learned multipliers at geometrically spaced points through the run.

    Geometric rather than even: most of the re-allocation happens in the first few
    thousand mutations, which even spacing would step straight over.
    """
    n_states = _states()
    n_snaps = min(
        int(jl.seval(f"length(DOCS_TRACED_STATES[{s}].snapshots)"))
        for s in range(1, n_states + 1)
    )
    if n_snaps == 0:
        return []
    picks = sorted(
        {1} | {max(1, round(n_snaps ** (k / n_points))) for k in range(1, n_points + 1)}
    )
    out = []
    for p in picks:
        seen = 0
        totals = None
        for s in range(1, n_states + 1):
            snap = f"DOCS_TRACED_STATES[{s}].snapshots[{p}]"
            seen += int(jl.seval(f"{snap}[1]"))
            values = [float(v) for v in jl.seval(f"{snap}[2]")]
            totals = (
                values if totals is None else [a + b for a, b in zip(totals, values)]
            )
        out.append((seen, [totals[r["index"] - 1] / n_states for r in rows]))
    return out


def report(rows, meta):
    print(
        f"\n{meta['n_mutations']} mutations, {meta['n_states']} population state(s), adaptation_strength={meta['adaptation_strength']}, plugins={meta['plugins']}"
    )

    print(
        f"\n{'mutation':<23}{'draws':>8}{'improves':>10}{'multiplier':>12}{'share: shipped':>16}{'learned':>10}{'drawn':>9}"
    )
    for r in rows:
        rate = "         -" if r["rate"] is None else f"{r['rate']:10.4f}"
        print(
            f"{r['mutation']:<23}{r['draws']:>8}{rate}{r['multiplier']:>12.3f}{r['share_shipped']:>16.3%}{r['share_learned']:>10.3%}{r['share_drawn']:>9.3%}"
        )
    print("\nshipped and learned are weight shares; drawn is what the search sampled,")
    print("which also reflects the engine zeroing kinds illegal for a given member.")

    trajectory = multiplier_trajectory(rows)
    if trajectory:
        print("\nlearned multiplier through the run, by mutations elapsed:")
        print(f"{'mutation':<23}" + "".join(f"{seen:>10}" for seen, _ in trajectory))
        for j, r in enumerate(rows):
            print(
                f"{r['mutation']:<23}"
                + "".join(f"{values[j]:>10.3f}" for _, values in trajectory)
            )


def main():
    assert MODEL_KWARGS["default_plugins"][0].alpha == PySRRegressor().alpha

    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))

    rows, meta = mutation_statistics(model)
    assert meta["plugins"] == [
        "SimulatedAnnealingPlugin",
        "AdaptiveParsimonyPlugin",
        "DocsTracedAdaptive",
    ], meta["plugins"]
    report(rows, meta)
    print(f"\nrecovered exactly: {check(model)}")


if __name__ == "__main__":
    main()
