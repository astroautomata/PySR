# Instrumentation and workflow

## Preamble

```python
import numpy as np

from pysr import *
```

## Using TensorBoard for logging

You can use TensorBoard to visualize the search progress, as well as
record hyperparameters and final metrics (like `min_loss` and `pareto_volume` - the latter of which
is a performance measure of the entire Pareto front).

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

You can then view the logs with:

```bash
tensorboard --logdir logs/
```

## Recording the genealogy of a search

The animation of an evolving population, with dots for evaluated expressions and edges
for the mutations and crossovers between them, is built from a trace file that
SymbolicRegression.jl can write during a search. Turning on tracing gives you one JSONL
record per (iteration, island), holding the live population plus every mutation,
crossover, tuning and death event. Joining those events on member references gives the
full parent-to-child graph, and the ancestry of the winning expression is then a walk
back up the parent edges.

One setup detail is worth stating up front: the trace writer lives in a JSON3
weak-dependency extension, and `use_tracing` loads it on demand, installing it into the
PySR Julia environment the first time a search asks for it. There is nothing to add by
hand, and a search that leaves tracing off never pays for it.

Now the data. Two hundred rows, three features, one quadratic term and one cosine term,
with Gaussian noise at five percent of the target's spread:

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(200, 3))
y = 2.5382 * np.cos(X[:, 2]) + X[:, 0] ** 2 - 1.5
y = y + rng.normal(0, 0.05 * y.std(), size=200)
```

The noise puts a floor under the loss. The realized mean squared error of the noise
itself is 0.0266, so nothing the search can find will score below roughly that.

Eight islands of twenty-seven members, ninety cycles per iteration, twenty-five
iterations. The `deterministic=True`, `random_state=0` and `parallelism="serial"` settings
make the trace reproducible, since tracing across threads would interleave records from
concurrent islands.

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

Be aware of the volume before you point this at a long run. Twenty-five iterations across
eight islands take about half a minute and leave two hundred iteration records and close
to 20 MB of JSONL, holding roughly 48,000 expressions. Tracing is a debugging and
visualization tool, so keep the searches short and delete the files when you are done.

Reading the trace back means walking the `iteration` records: their `members` give you
each expression by reference, and their `mutations` map gives the event list for each
parent. A crossover event is pushed onto both parents' event lists, so we read it from
`parent1` only. Each child records which of its two parents it descends from, so the edge
is oriented onto that parent and the other is carried alongside it as `mate`:

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

With the graph in hand, the lit path in the animation is the lowest-loss expression
followed back up its parent edges:

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

Across five seeds the shape of the trace is stable. Each run recorded between 48,573 and
48,741 distinct expressions and between 51,578 and 51,865 parent-to-child edges, split
into 28,707 to 28,917 mutations, 17,288 to 17,758 crossovers, and exactly 5,400 tuning
events, one per member per island per iteration. Between 9,311 and 10,148 of those edges
were rejected proposals, and between 10,546 and 11,078 carried a distinct second parent.
The winner's ancestry ran 70 to 186 steps, of which 22 to 59 were crossovers and 18 to 38
of those crossovers brought in a second parent. So all four things the animation draws,
the evaluated expressions, the mutation and crossover edges, the path back to the winner,
and the second parents feeding into it, come out of the trace directly.

Each search takes about 32 seconds on one core.

Recovery is another matter, and twenty-five iterations is not many. Two of the five seeds
reached the noise floor, at losses 0.0263 and 0.0263. The other three finished at 0.031,
0.275 and 0.439. Raise `niterations` if you want the answer; the trace is the same shape
either way, just larger.

One naming detail: trace records label features `x1`, `x2`, ... in column order and ignore
any `variable_names` you pass to the search, so `x1` in a trace record is `X[:, 0]` and
`x3` is `X[:, 2]`.

The full runnable script is `examples/search_trace.py`.

## Closing an agent loop with `guesses=`

`guesses=` lets you hand PySR a starting expression, which means one search can be
driven by whatever looked at the results of the previous one. In the release film
that driver is a coding agent working in a scratch directory: it is given a CSV,
runs a cold search, reads the whole Pareto front, writes one candidate equation,
and runs the search again with that candidate in `guesses=`. The example on this
page is that session with one part replaced. A docs page cannot call a live model
and still reproduce, so the model call returns the expression the agent actually
replied with, and everything else is the filmed search: same data, same operators,
same two rounds, same `guesses=` handoff.

The measurements are 160 rows of kinetic energy against mass and velocity in
natural units, with velocities up to 0.88 c, so the Newtonian form is only the
low-velocity limit. Noise is 0.15 percent of the median energy, which puts the
best achievable loss at 5.127e-08:

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

The target is $E = m\left(\left(1 - v^2\right)^{-1/2} - 1\right)$, and the search
is never told that. Its vocabulary is the four arithmetic operators plus `sqrt`,
which is exactly enough to write the Lorentz factor:

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

Those are the same settings the film pinned, and nothing beyond the problem is
configured. `deterministic=True` with `parallelism="serial"` and a fixed
`random_state` is what lets a drop in loss be attributed to the guess that was
fed in rather than to a different random draw.

A cold search at these settings gets close and stops. On seed 3 it ends at loss
5.664e-05, three orders of magnitude above the noise floor, and the front it
hands over reads as a ladder of polynomials in velocity:

```
 complexity     loss                                                            equation
          1 0.161396                                                            velocity
          3 0.157968                                                 velocity * 1.112165
          4 0.111584                                               sqrt(mass) * velocity
          5 0.021796                                        mass * (velocity * velocity)
          7 0.018648                       ((velocity * velocity) * mass) + -0.056107998
          9 0.004324             (velocity * 1.3305349) * ((velocity * velocity) * mass)
         11 0.003208 ((velocity * ((velocity * 1.6953516) * velocity)) * mass) * velocity
```

The ladder continues in the same style up to complexity 30 at loss 5.664e-05. The
complexity-5 row is $mv^2$, the Newtonian energy up to a factor, and every row
above it is a correction series bolted onto that term. This is the reading the
filmed agent made, and its reply was one expression:

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

That function is the stand-in. A real loop puts the printed front and the list of
expressions already tried into a prompt and parses one expression out of the
reply, which is what the film recorded; here the reply is a constant so the page
reproduces. Everything the stand-in stands for is the model call itself.

The loop around it is a few lines. `guesses` takes a plain list of expression
strings for a single-output problem, and the loop stops when the best loss is at
the measurement noise floor or the driver has nothing left to propose:

```python
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

Round one starts from that guess and reaches the floor on all five seeds tested.
The cold round lands between 5.664e-05 and 4.972e-04 depending on the seed; the
seeded round lands between 5.148e-08 and 5.211e-08, against a floor of 5.127e-08.
On seed 3 the front reduces to the law in closed form:

```
 complexity         loss                                                equation
          8 1.112200e-02              (mass / sqrt(1.1663519 - velocity)) - mass
          9 4.324070e-03 velocity * ((velocity * (velocity * 1.3304913)) * mass)
         10 5.210982e-08       (mass / sqrt(1.0 - (velocity * velocity))) - mass
```

That is the filmed session's own result, 5.210982e-08 at complexity 10,
reproduced on PySR 2.1.0. Other seeds reach the same loss carrying redundant
structure around it, complexity 14 to 28 at losses indistinguishable from the
floor, since once the fit is at the noise level extra terms cost nothing. Both
searches together take 5.2 to 5.6 seconds per seed, after about 21 seconds of
one-time Julia compilation in the first process.

Note that the guess only has to be right about structure. PySR optimizes
constants itself, and the guess above writes `1` where the search later prefers
`1.0` to eight digits.

The full runnable script is `examples/agent_loop_guesses.py`.
