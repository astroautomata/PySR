"""Recover 2.5*cos(3x) + 0.5x^2 - 1 with the backsolve mutation switched on.

Backsolve inverts the operators sitting above a chosen subtree, which turns
"improve this whole expression" into "solve for the piece that belongs in this
hole", then fits that piece as a sparse weighted sum of parts already present in
the population. It is off by default (`weight_backsolve` maps to weight 0.0);
this target is out of reach for the ordinary mutations alone.
"""

import numpy as np

from pysr import PySRRegressor

x = np.linspace(-3.0, 3.0, 200)
X = x.reshape(-1, 1)
y = 2.5 * np.cos(3.0 * x) + 0.5 * x * x - 1.0
VARIABLE_NAMES = ["x"]

MODEL_KWARGS = dict(
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "exp"],
    weight_backsolve=1.0,
    maxsize=30,
    precision=64,
    niterations=10,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

# One backsolve event, applied outside the search loop so its two losses can be
# read directly. `mutate!` returns a member only when its monotone gate has
# already confirmed the child beats the parent on the data.
_BACKSOLVE_EVENT = """
function backsolve_event(dataset, population, options, mutation)
    events = NamedTuple[]
    for member in population.members
        tree = copy(member.tree)
        result = SymbolicRegression.mutate!(
            tree, member, mutation, options;
            trace=nothing, dataset, curmaxsize=options.maxsize,
            nfeatures=size(dataset.X, 1), population_for_backsolve=population,
        )
        result.member === nothing && continue
        push!(events, (
            before=Float64(member.loss),
            after=Float64(result.member.loss),
            parent=SymbolicRegression.string_tree(member.tree, options),
            child=SymbolicRegression.string_tree(result.member.tree, options),
        ))
    end
    return events
end
"""


def check(model):
    """The target recovered to double-precision round-off."""
    prediction = np.asarray(model.predict(X), dtype=float)
    return bool(np.max(np.abs(prediction - y)) < 1e-10)


def backsolve_events(model, seed=0):
    """Apply one backsolve mutation to every member of the search's populations.

    Returns the events the mutation's own acceptance gate let through, each with
    the parent's loss before and the child's loss after.
    """
    from pysr.julia_import import SymbolicRegression as SR
    from pysr.julia_import import jl

    jl.seval("using Random: seed!")
    jl.seval(f"seed!({seed})")
    run_event = jl.seval(_BACKSOLVE_EVENT)

    dtype = np.float64 if MODEL_KWARGS["precision"] == 64 else np.float32
    dataset = SR.Dataset(
        jl.Matrix(np.asfortranarray(X.T.astype(dtype))), jl.Vector(y.astype(dtype))
    )
    mutation = SR.BacksolveMutation()
    options = model.julia_options_
    populations = model.julia_state_[0][0]

    events, swept = [], 0
    for i in range(int(jl.length(populations))):
        population = populations[i]
        swept += int(population.n)
        found = run_event(dataset, population, options, mutation)
        for k in range(int(jl.length(found))):
            e = found[k]
            events.append(
                dict(
                    before=float(e.before),
                    after=float(e.after),
                    parent=str(e.parent),
                    child=str(e.child),
                )
            )
    return events, swept


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))

    prediction = np.asarray(model.predict(X), dtype=float)
    print(f"\nmax abs error: {np.max(np.abs(prediction - y)):.3g}")
    print(f"recovered: {check(model)}")

    events, swept = backsolve_events(model)
    print(f"\nbacksolve events accepted: {len(events)} of {swept} members swept")
    if events:
        ratios = sorted(e["after"] / e["before"] for e in events if e["before"] > 0)
        print(f"median child/parent loss ratio: {ratios[len(ratios) // 2]:.3g}")
        best = min(
            events, key=lambda e: e["after"] / e["before"] if e["before"] > 0 else 1.0
        )
        print(f"largest drop: loss {best['before']:.6g} -> {best['after']:.6g}")
        print(f"  parent: {best['parent']}")
        print(f"  child:  {best['child']}")


if __name__ == "__main__":
    main()
