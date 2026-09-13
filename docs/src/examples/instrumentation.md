# Instrumentation and workflow

## Using TensorBoard for logging

Pass `TensorBoardLoggerSpec` through `logger_spec` to write search progress to TensorBoard. `TensorBoardLoggerSpec` records the model's search hyperparameters and writes scalar summaries for the run. The example uses `min_loss`, which tracks the current best loss, and `pareto_volume`, which summarizes the Pareto front.

```python
import numpy as np

from pysr import PySRRegressor, TensorBoardLoggerSpec

rstate = np.random.RandomState(42)

# Uniform dist between -3 and 3:
X = rstate.uniform(-3, 3, (1000, 2))
y = np.exp(X[:, 0]) + X[:, 1]

# Create a logger that writes to "logs/run*":
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)

model = PySRRegressor(
    binary_operators=["+", "*", "-", "/"],
    logger_spec=logger_spec,
)
model.fit(X, y)
```

Start TensorBoard with the directory that contains the run logs:

```bash
tensorboard --logdir logs/
```

## Recording the genealogy of a search

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D8.mp4"></video>

The complete recorder and post-processing workflow is in `examples/search_trace.py`. `load_trace` and `ancestry` are example-local helpers, not PySR API methods.

Tracing records how a symbolic-regression population changes. Setting `use_tracing=True` activates the optional recorder. PySR loads the required JSON dependency lazily on first use. It then writes JSONL records to `tracing_file`, one for each population at each iteration, with the live members and their mutation, crossover, tuning, and death events. These recorded fields supply the members and parent relationships that the helpers need to reconstruct a graph and one selected path. The helpers work from recorded events and do not expose every historical branch.

With tracing disabled, PySR does not request any trace output or load the tracing dependency.

The target combines one quadratic term, one cosine term, and an offset. The added Gaussian noise has standard deviation equal to $5\%$ of the target's spread:

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(200, 3))
y = 2.5382 * np.cos(X[:, 2]) + X[:, 0] ** 2 - 1.5
y = y + rng.normal(0, 0.05 * y.std(), size=200)
```

Serial execution prevents concurrent populations from interleaving their trace records.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp"],
    population_size=27,
    populations=8,
    ncycles_per_iteration=90,
    maxsize=20,
    niterations=25,
    parallelism="serial",
    deterministic=True,
    random_state=0,
    use_tracing=True,
    tracing_file="search_trace.jsonl",
)
model.fit(X, y)
```

Keep tracing runs short: the files grow with the search, and this `load_trace` helper reads the entire file into memory.

<details>
<summary>Trace-reading code</summary>

```python
import json
import pathlib

def load_trace(trace_path):
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
```

</details>

`ancestry` selects the lowest-loss recorded member and follows the first edge for each child. It returns one path, which ends when no edge remains or a reference repeats.

```python
def ancestry(nodes, edges):
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
```

Trace records label features `x1`, `x2`, and so on by column order, independently of any `variable_names` passed to the search. Thus `x1` refers to `X[:, 0]`, and `x3` refers to `X[:, 2]` in this example.

Use `examples/search_trace.py` for the complete tracing workflow.

## Closing an agent loop with `guesses=`

The `guesses=` keyword accepts a list of expression strings to seed a fit. An external workflow can inspect `model.equations_`, propose an expression, and pass it into a later search. PySR can then optimize its numeric constants.

This example replays a recorded coding-agent proposal from `PROPOSALS`. The custom `agent(front, tried)` helper returns stored expressions; it does not call a live agent or model service. The complete implementation is in `examples/agent_loop_guesses.py`.

The target is relativistic kinetic energy in natural units, with $c=1$ and speeds up to $0.88c$. Gaussian noise has standard deviation $0.15\%$ of the median exact energy. `NOISE_FLOOR` stores its variance for the loop's training-loss stopping rule.

<details>
<summary>Data generation code</summary>

```python
import numpy as np

rng = np.random.default_rng(42)
mass = rng.uniform(0.5, 3.0, 160)
velocity = rng.uniform(0.08, 0.88, 160)
kinetic_energy_exact = mass * (1 / np.sqrt(1 - velocity**2) - 1)
NOISE_SIGMA = 0.0015 * float(np.median(kinetic_energy_exact))
kinetic_energy = kinetic_energy_exact + rng.normal(0.0, NOISE_SIGMA, mass.size)
NOISE_FLOOR = NOISE_SIGMA**2

X = np.stack([mass, velocity], axis=1)
y = kinetic_energy
VARIABLE_NAMES = ["mass", "velocity"]
```

</details>

```python
from pysr import PySRRegressor

model_kwargs = dict(
    operators={1: ["sqrt"], 2: ["+", "-", "*", "/"]},
    niterations=10,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)
```

The proposal uses the known relativistic structure:

```python
PROPOSALS = ["(mass / sqrt(1 - (velocity * velocity))) - mass"]

def agent(front, tried):
    print("front handed to the agent:")
    print(
        front[["complexity", "loss", "equation"]].to_string(index=False),
        "\n",
        flush=True,
    )
    return PROPOSALS[len(tried)] if len(tried) < len(PROPOSALS) else None
```

The loop begins with `guesses=None` and passes each proposal as a one-element list to the next fit. It stops when the minimum training loss reaches `2 * NOISE_FLOOR` or the helper has no proposal left. This threshold is an example-specific stopping rule, not a lower bound on achievable loss.

```python
from pysr import PySRRegressor

guesses, tried = None, []
while True:
    model = PySRRegressor(**model_kwargs, guesses=guesses, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    front = model.equations_
    if front["loss"].min() <= 2 * NOISE_FLOOR:
        break
    proposal = agent(front, tried)
    if proposal is None:
        break
    tried.append(proposal)
    guesses = [proposal]
```

Use `examples/agent_loop_guesses.py` for the complete handoff workflow.
