"""Record the genealogy of a search: every expression evaluated, and its parents.

`use_tracing` makes SymbolicRegression.jl write one JSONL record per (iteration, island)
holding the live population and every mutation, crossover, tuning and death event. Joining
those events on member refs gives the full parent-to-child graph, from which the winner's
ancestry is a walk back up the parent edges.

The trace writer lives in a JSON3 weak-dependency extension, which PySR installs into its
Julia environment the first time a search asks for it.

Trace records name features `x1`, `x2`, ... in column order, so `x1` below is `X[:, 0]`.
"""

import collections
import json
import pathlib
import sys

import numpy as np

from pysr import PySRRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(200, 3))
y = 2.5382 * np.cos(X[:, 2]) + X[:, 0] ** 2 - 1.5
y = y + rng.normal(0, 0.05 * y.std(), size=200)

TRACE_PATH = "search_trace.jsonl"
NITERATIONS = 25

SETTINGS = dict(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    population_size=27,
    populations=8,
    ncycles_per_iteration=90,
    maxsize=20,
    parallelism="serial",
    deterministic=True,
    random_state=0,
)


def run_search(trace_path, niterations=NITERATIONS):
    """Run the search with tracing on, leaving the JSONL at `trace_path`."""
    model = PySRRegressor(
        niterations=niterations,
        use_tracing=True,
        tracing_file=trace_path,
        **SETTINGS,
    )
    model.fit(X, y)
    return model


def load_trace(trace_path):
    """Read the JSONL into `nodes` (ref -> expression record) and `edges` (parent to child).

    A crossover event is pushed onto both parents' event lists, so it is read from
    `parent1` only. Each child records which of the two it descends from, so the edge is
    oriented onto that parent and the other is carried alongside it as `mate`.
    """
    nodes, edges = {}, []
    for line in pathlib.Path(trace_path).read_text().splitlines():
        record = json.loads(line)
        if record["record_type"] != "iteration":
            continue
        for member in record["members"]:
            nodes.setdefault(member["ref"], {}).update(member)
        for ref, entry in record["mutations"].items():
            parent = int(ref)
            nodes.setdefault(parent, {}).update(
                {k: entry[k] for k in ("tree", "loss", "cost", "parent") if k in entry}
            )
            for event in entry["events"]:
                kind = event["type"]
                detail = event.get("mutation") or event.get("details") or {}
                if kind == "crossover" and event["parent1"] != parent:
                    continue
                children = (
                    (event["child1"], event["child2"])
                    if kind == "crossover"
                    else (event["child"],) if "child" in event else ()
                )
                for child in children:
                    edges.append(
                        dict(
                            parent=parent,
                            child=child,
                            kind=kind,
                            mutation=detail.get("type"),
                            result=detail.get("result", "accept"),
                            mate=event.get("parent2"),
                        )
                    )
    for edge in edges:
        if (
            edge["mate"] is not None
            and nodes.get(edge["child"], {}).get("parent") == edge["mate"]
        ):
            edge["parent"], edge["mate"] = edge["mate"], edge["parent"]
    return nodes, edges


def ancestry(nodes, edges):
    """The lit path: the lowest-loss expression walked back up its parent edges."""
    first_edge = {}
    for edge in edges:
        first_edge.setdefault(edge["child"], edge)
    winner = min(
        (ref for ref, n in nodes.items() if n.get("loss") is not None),
        key=lambda ref: nodes[ref]["loss"],
    )
    chain, current, seen = [], winner, set()
    while current in first_edge and current not in seen:
        seen.add(current)
        chain.append(first_edge[current])
        current = first_edge[current]["parent"]
    chain.reverse()
    return winner, chain


def check(nodes, edges):
    """A usable genealogy: every edge resolves, the winner chains back to a seed, and
    crossovers carry a distinct second parent."""
    if not edges or not any(e["kind"] == "crossover" for e in edges):
        return False
    if not any(e["mate"] is not None and e["mate"] != e["parent"] for e in edges):
        return False
    if any(e["parent"] not in nodes for e in edges):
        return False
    winner, chain = ancestry(nodes, edges)
    return bool(chain) and nodes[chain[0]["parent"]].get("parent") == -1


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else TRACE_PATH
    run_search(trace_path)
    nodes, edges = load_trace(trace_path)

    kinds = collections.Counter(e["kind"] for e in edges)
    print(f"trace file: {trace_path} ({pathlib.Path(trace_path).stat().st_size} bytes)")
    print(f"expressions recorded: {len(nodes)}")
    print(f"parent-to-child edges: {len(edges)}  {dict(kinds)}")
    print(f"rejected: {sum(1 for e in edges if e['result'] == 'reject')}")
    print(
        f"edges with a second parent: {sum(1 for e in edges if e['mate'] is not None and e['mate'] != e['parent'])}"
    )

    winner, chain = ancestry(nodes, edges)
    print(f"\nwinner loss {nodes[winner]['loss']:.6g}, ancestry {len(chain)} steps")
    for edge in chain[-8:]:
        print(
            f"  {edge['kind']:10s} {str(edge['mutation']):32s} -> {nodes[edge['child']].get('tree', '?')[:70]}"
        )

    print(f"\nusable genealogy: {check(nodes, edges)}")


if __name__ == "__main__":
    main()
