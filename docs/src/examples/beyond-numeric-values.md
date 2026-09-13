# Beyond numeric values

These examples carry values that have their own structure: letters, cell states, machine words, and pen programs. A `TypeSpec` supplies each value's representation and hooks such as sampling, mutation, and printing; typed operators define how values combine and transform; and the loss determines what counts as agreement on the scored domain. Read [Custom value types](/examples/value-types#custom-value-types) first for the contract that a custom type must satisfy.

## Breaking an affine cipher with a letter type

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D1.mp4"></video>

The data comes from the affine cipher

$$ E(p) = (5p + 8) \bmod 26, $$

where each letter is represented by its position from `A` through `Z`. The search receives only pairs of ciphertext and plaintext codes. It must infer the inverse relation from those examples instead of being handed the displayed formula.

<details>
<summary>Data generation code</summary>

```python
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLAIN = "THELAWSOFPHYSICSAREWRITTENINTHELANGUAGEOFMATHEMATICS"
CIPHER = "".join(ALPHABET[(5 * ALPHABET.index(c) + 8) % 26] for c in PLAIN)

X = np.array([[ALPHABET.index(c)] for c in CIPHER], dtype=object)
y = np.array([ALPHABET.index(c) for c in PLAIN], dtype=object)
```

</details>

`Letter` stores a code in `0:25`. Its `string` hook displays constants as letters, such as `V` for code `21`.

```python
from pysr import PySRRegressor, TypeSpec

spec = TypeSpec(
    "Letter",
    fields={"code": "Int"},
    sample="rng -> Letter(rand(rng, 0:25))",
    mutate="(rng, value, temperature) -> Letter(mod(value.code + rand(rng, (-1, 1)), 26))",
    string="value -> string(Char('A' + value.code))",
)
```


The three operators implement addition, subtraction, and multiplication modulo $26$. Every result remains a `Letter`, and the modulus appears directly in each operator. That built-in alphabet size makes this the easier cipher search.

Use absolute code distance as `elementwise_loss` to reward partial improvements. Zero loss means every ciphertext/plaintext pair matches:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=spec,
    operators={
        2: [
            "shift(a::Letter, b::Letter) = Letter(mod(a.code + b.code, 26))",
            "unshift(a::Letter, b::Letter) = Letter(mod(a.code - b.code, 26))",
            "mix(a::Letter, b::Letter) = Letter(mod(a.code * b.code, 26))",
        ]
    },
    elementwise_loss="letter_loss(prediction::Letter, target::Letter)::Float64 = abs(prediction.code - target.code)",
    niterations=20,
    deterministic=True,
    parallelism="serial",
    random_state=0,
)
model.fit(X, y)
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```


A `loss_function` is useful when scoring needs the dataset as a whole, as in the rollout in [Swinging up a cart-pole](/examples/objectives#swinging-up-a-cart-pole-with-a-rollout-objective) or the period certificate in [Inventing a pseudorandom generator](/examples/objectives#inventing-a-pseudorandom-generator-with-no-target).

One recovered expression is `mix(V, shift(S, x0))`: `V` has code $21$ and `S` has code $18$, giving $21(x_0 + 18) \bmod 26$. Since $21$ is the multiplicative inverse of $5$ modulo $26$, this expression decodes the ciphertext.

### Learning the modulus too

The harder variant removes the alphabet size from the operators. It uses values in `0:51` with integer addition, multiplication, and modulo, so the search must discover the constant $26$ along with the decryption law. This removes a strong structural hint and enlarges the constant search.

One recovered decryption law is:

```
modulo(mul(add(18, x0), 47), 26)
```

Run `examples/affine_cipher.py` for the first search or `examples/affine_freemod.py` for the full type, operator, and loss setup of this variant. Allow several minutes for the latter.

## Rediscovering Conway's Game of Life

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D3.mp4"></video>

Conway's Game of Life can be written as a rule about a cell and its live-neighbour count. This example withholds that rule and supplies only its transition table, asking PySR to reconstruct the logical expression.

The table has $18$ distinct states: two current cell states, each paired with every neighbour count from $0$ through $8$. Repeating the table $8$ times produces $144$ rows for scoring without introducing new cases. A zero-one loss therefore charges $1/18$ for a rule that misses one distinct state. The arrays use `dtype=object` so each value can be wrapped in the custom Julia type.

<details>
<summary>Data generation code</summary>

```python
import numpy as np

TABLE = [(a, n, int(n == 3 or (a == 1 and n == 2))) for a in (0, 1) for n in range(9)]
REPLICAS = 8
X = np.array([[a, n] for a, n, _ in TABLE] * REPLICAS, dtype=object)
y = np.array([t for _, _, t in TABLE] * REPLICAS, dtype=object)
```

</details>

The `Cell` type wraps a single `Int`, which is enough for the current state, the neighbour count, and the output code. Its hooks propose small counts, move existing constants locally modulo $9$, and print the integer payload.

```python
from pysr import PySRRegressor, TypeSpec

spec = TypeSpec(
    "Cell",
    fields={"v": "Int"},
    sample="rng -> Cell(rand(rng, 0:8))",
    mutate="(rng, value, temperature) -> Cell(mod(value.v + rand(rng, (-1, 1)), 9))",
    string="value -> string(value.v)",
)
```


The operator vocabulary contains `not`, `and`, `or`, and the equality test `eq`. Arithmetic and threshold comparisons are absent, so a neighbour threshold can enter an expression only as a constant in a call such as `eq(n, 3)`. The loss is zero-one equality on each table row. It records exact agreement, which suits a discrete truth table where the numerical distance between cell codes carries no meaning.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=spec,
    operators={
        1: ["not(a::Cell) = Cell(a.v == 0 ? 1 : 0)"],
        2: [
            "and(a::Cell, b::Cell) = Cell((a.v != 0 && b.v != 0) ? 1 : 0)",
            "or(a::Cell, b::Cell) = Cell((a.v != 0 || b.v != 0) ? 1 : 0)",
            "eq(a::Cell, b::Cell) = Cell(a.v == b.v ? 1 : 0)",
        ],
    },
    elementwise_loss="cell_loss(prediction::Cell, target::Cell)::Float64 = prediction.v == target.v ? 0.0 : 1.0",
    niterations=160,
    deterministic=True,
    parallelism="serial",
    random_state=0,
)
model.fit(X, y, variable_names=["alive", "n"])
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```


A recovered rule is:

```
or(eq(n, 3), and(eq(n, 2), alive))
```

The first branch allows birth or survival at three neighbours. The second adds survival of a live cell at two neighbours. It agrees with all 18 possible input states, so the transition table fully checks this local rule.

Run `examples/game_of_life.py` to reproduce the Game of Life search.

## Searching over machine words

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/S3.mp4"></video>

This search targets the bit interleave used by Morton, or Z-order, indexing. It gives every expression a `Word` value and restricts the vocabulary to machine instructions: xor, and, or, not, left shift, and right shift. Shift distances are ordinary `Word` constants, so the search must find them as part of the expression.

The reference interleave is evaluated on all $256$ pairs of $4$-bit inputs.

<details><summary>Data generation code</summary>

```python
import numpy as np

from pysr import PySRRegressor, TypeSpec

WIDTH = 4
WIDER = 5


def morton(x, y, width=WIDTH):
    z = 0
    for i in range(width):
        z |= ((x >> i) & 1) << (2 * i)
        z |= ((y >> i) & 1) << (2 * i + 1)
    return z


PAIRS = [(a, b) for b in range(1 << WIDTH) for a in range(1 << WIDTH)]
X = np.array(PAIRS, dtype=object)
y = np.array([morton(a, b) for a, b in PAIRS], dtype=object)
```

</details>

The `Word` type wraps a `UInt32`. Its `sample` hook favors small values because useful shift distances are small while mask constants still need to reach across the full word. `mutate` either flips a single bit or takes a random unsigned step. This lets a mask change one bit at a time. The `string` hook renders small values as decimals and larger values as zero-padded hexadecimal, keeping shift constants and masks easy to distinguish. The `shl` and `shr` definitions below pass `b.bits` directly; this example does not mask the shift count into `0:31`, so arbitrary sampled `UInt32` counts follow the underlying Julia shift semantics.

```python
from pysr import TypeSpec

spec = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
)
```


Each operator lifts one machine instruction to `Word`. The loss counts the set bits in the xor of prediction and target and divides by $32$. A zero loss therefore means that every output bit agrees for every scored pair.

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=spec,
    operators={
        1: ["bnot(a::Word) = Word(~a.bits)"],
        2: [
            "bxor(a::Word, b::Word) = Word(xor(a.bits, b.bits))",
            "band(a::Word, b::Word) = Word(a.bits & b.bits)",
            "bor(a::Word, b::Word) = Word(a.bits | b.bits)",
            "shl(a::Word, b::Word) = Word(a.bits << b.bits)",
            "shr(a::Word, b::Word) = Word(a.bits >> b.bits)",
        ],
    },
    elementwise_loss="bit_loss(prediction::Word, target::Word)::Float64 = count_ones(xor(prediction.bits, target.bits)) / 32",
    maxsize=45,
    niterations=800,
    deterministic=True,
    parallelism="serial",
    random_state=0,
    verbosity=0,
)

model.fit(X, y, variable_names=["x", "y"])
```


Set `maxsize` large enough for the intended computation. A four-bit reference interleaver uses 41 tree nodes because repeated subexpressions each count toward its size; `maxsize=45` leaves room for that construction.

An expression that matches all four-bit pairs need not work at larger widths: the training domain leaves parts of the masks unconstrained. Use the script's `exact_front_rows` helper to check four-bit agreement and `wider_agreement(model, index)` to test a selected expression on five-bit inputs.

Run `examples/morton_interleave.py` to reproduce this machine-word search.

## Turtle graphics: searching over drawings

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/D4.mp4"></video>

The excerpts below come from `examples/turtle_graphics.py`. Run that script for the complete example; it supplies the Python `path` converter and the Julia `PREAMBLE` containing the underscore-prefixed drawing, mutation, and scoring helpers and constants.

Here the value is a drawing program. A `Path` is a list of pen commands, `forward` and `turn`; operators repeat a path, reflect it, and concatenate command lists. Each expression therefore evaluates to a program, and the value of that program is the picture it draws.

The drawing uses two parallel vectors. `kind=0` marks forward motion. `kind=1` marks a turn. `val` stores the forward length or the turn angle in radians. The `ARM` list below writes turn angles in degrees, and the Python `path` helper converts those entries to radians before storing `y`; the custom `string` hook prints stored turns back in degrees. The target is a plus-sign outline made from four copies of an arm with three edges. The dataset has one row and one feature, `f`, containing a unit forward step. Turns and the path operators must supply the rest of the program.

```python
import numpy as np
from pysr import PySRRegressor, TypeSpec

ARM = [("F", 1.0), ("T", 90.0), ("F", 1.0), ("T", -90.0), ("F", 1.0), ("T", 90.0)]
TARGET = ARM * 4

X = np.empty((1, 1), dtype=object)
X[0, 0] = ([0], [1.0])
y = np.empty(1, dtype=object)
y[0] = path(TARGET)
VARIABLE_NAMES = ["f"]
```


### Scoring a picture

Two programs can trace the same outline in different orders, so the loss compares rasterized pictures instead of command lists. Each drawing is placed on a $96\times96$ grid after its bounding box is centered and its larger extent is scaled to the grid. The comparison therefore responds to shape while ignoring translation and overall size.

The score is a symmetric chamfer distance between the two sets of lit pixels, measured in pixels and divided by the grid width. The score averages the distance from candidate pixels to the target and from target pixels to the candidate, so zero means the raster masks coincide.

<details><summary>The chamfer loss and the raster</summary>

```julia
# Symmetric chamfer distance in pixels, divided by the grid width. Zero means
# the two drawings lit exactly the same pixels.
function _chamfer(pk::Vector{Int8}, pv::Vector{Float64},
                  tk::Vector{Int8}, tv::Vector{Float64})
    a = _raster(pk, pv)
    a === nothing && return _FAIL
    b, db = _target_grids(tk, tv)
    any(b) || return _FAIL
    da = _edt(a)
    ab = _mean_to(a, db); ba = _mean_to(b, da)
    (ab === nothing || ba === nothing) && return _FAIL
    v = 0.5 * (ab + ba) / _GRID
    return isfinite(v) ? v : _FAIL
end
```

</details>

### The pen program type

The `TypeSpec` for `Path` stores the two vectors and supplies hooks for continuous turns. `sample` creates one turn at an angle in the interval from $-180$ to $180$ degrees. `mutate` selects a turn and either perturbs it by a Gaussian step with scale $30$ degrees times the temperature or resamples it. The scalar-constant hooks expose every turn to BFGS, which can polish an angle after evolution has placed it near a good value.

Constant folding can collapse an all-constant subtree into a multi-command path. `mutate` and `string` therefore handle values command by command, and the printer represents a folded multi-command constant as `p[...]`.

```python
from pysr import TypeSpec

PATH = TypeSpec(
    "Path",
    fields={"kind": "Vector{Int8}", "val": "Vector{Float64}"},
    preamble=PREAMBLE,
    sample="rng -> Path(Int8[1], [_real(rng)])",
    mutate="""(rng, value, temperature) -> begin
        turns = findall(isone, value.kind)
        isempty(turns) && return Path(Int8[1], [_real(rng)])
        i = rand(rng, turns)
        val = copy(value.val)
        val[i] = rand(rng, Bool) ? _perturb(rng, val[i], temperature) : _real(rng)
        Path(copy(value.kind), val)
    end""",
    scalar_constants="value -> _turns(value.kind, value.val)",
    with_scalar_constants="(value, c) -> Path(copy(value.kind), _put_turns(value.kind, value.val, c))",
    is_valid="value -> !isempty(value.kind) && all(isfinite, value.val)",
    string="value -> _literal(value.kind, value.val)",
)
```


The five operators form a small program algebra. `seq` runs one path after another. `dup` and `tri` repeat a path twice and three times. `mir` negates every turn, reflecting the path. `nest` performs an L-system substitution: it replaces each forward command with a copy of the full path. Each copy is scaled so its net displacement matches the forward command it replaces. Repetition and reflection let a short expression draw many strokes.

```python
OPERATORS = {
    1: [
        "dup(a::Path) = (t = _seq(a.kind, a.val, a.kind, a.val); Path(t[1], t[2]))",
        "tri(a::Path) = (t = _seq(a.kind, a.val, a.kind, a.val); u = _seq(t[1], t[2], a.kind, a.val); Path(u[1], u[2]))",
        "mir(a::Path) = (t = _mir(a.kind, a.val); Path(t[1], t[2]))",
        "nest(a::Path) = (t = _nest(a.kind, a.val); Path(t[1], t[2]))",
    ],
    2: [
        "seq(a::Path, b::Path) = (t = _seq(a.kind, a.val, b.kind, b.val); Path(t[1], t[2]))"
    ],
}

CHAMFER_LOSS = "chamfer(prediction::Path, target::Path)::Float64 = _chamfer(prediction.kind, prediction.val, target.kind, target.val)"
```


### The search

Pass the path type, operators, and picture loss to the estimator:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    type_spec=PATH,
    operators=OPERATORS,
    elementwise_loss=CHAMFER_LOSS,
    niterations=20,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    random_state=0,
)

model.fit(X, y, variable_names=["f"])
print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
```


Scoring each candidate requires rasterization and a distance transform, so even this 20-iteration search can take tens of minutes to hours. Zero loss means the raster masks match; it does not require identical command order or exact turn angles.

One recovered drawing program is:

```
dup(dup(mir(seq(t-179.6493032061754, tri(seq(p[t-270.04570985370526,t-270.04570985370526,t-270.04570985370526], f))))))
```

The complete drawing search is in `examples/turtle_graphics.py`.

## Evolving a hopping controller

<video controls muted playsinline preload="metadata" src="https://raw.githubusercontent.com/MilesCranmer/PySR_Docs/38b98e49200ee5e1629a62fb7e0b811d64286154/clips/Hopper.mp4"></video>

Each value in this search is a motor command with three numbers, one for each joint of MuJoCo's hopper. The objective measures performance in simulation. Each candidate expression controls the robot for a fixed-horizon rollout, and its loss is the negated reward from that rollout. The search uses no expert policy, demonstration data, or distilled network; it optimises locomotion directly.

The snippets below describe the recorded search. The runnable script, `examples/hopper_controller.py`, replays its saved controller.

The controller is a `Vec3`, three `Float64` components carried through the tree together:

```python
from pysr import TypeSpec

TypeSpec(
    "Vec3",
    fields={"data": "Vector{Float64}"},
    sample="rng -> Vec3(randn(rng, 3))",
    scalar_constants="value -> Float64[value.data[1], value.data[2], value.data[3]]",
    with_scalar_constants="(value, constants) -> Vec3(collect(constants))",
    is_valid="value -> length(value.data) == 3 && all(isfinite, value.data)",
    string='value -> "Vec3([" * join(repr.(Float64.(value.data)), ", ") * "])"',
    loss_type="Float64",
)
```

Each sampled constant contains three independent Gaussian draws, giving each
motor its own bias. The scalar-constant hooks expose all three components to the
optimiser so it can tune each motor separately.

The operators are componentwise addition, subtraction and multiplication, and nothing else. No control-specific primitive was supplied:

```python
{
    2: [
        'vadd(x, y) = Vec3(x.data .+ y.data)',
        'vsub(x, y) = Vec3(x.data .- y.data)',
        'vmul(x, y) = Vec3(x.data .* y.data)',
    ]
}
```

### Scoring a controller by simulating it

The eleven features encode the hopper state as vectors. They include the three
joint angles, angular velocities clipped to $[-10, 10]$ and divided by $10$, and two cyclic permutations
of each. The torso height, pitch, forward velocity, vertical velocity and
angular velocity are each broadcast across all three components. The three torso velocities are divided by $5$. A single tree
computes all three motor commands, so the permuted copies allow each joint to use
other joints' angles. Before reaching the motors, the final value passes through
a componentwise `tanh` that limits each command to the actuator range.

Each rollout accumulates the standard Hopper-v5 reward, a survival term plus forward velocity minus a small control cost, and divides by the full horizon, so a fall contributes zero for every step it did not survive. That single number, averaged over eight frozen training starts and negated, is the loss. The Julia loss function is a thin call into the rollout evaluator:

```python
LOSS_SOURCE = "hopper_rollout_loss(tree, dataset, options) = Main.HopperObjective.loss(tree, dataset, options)"
```

Because the objective never reads the dataset, `X` and `y` are one row of zero vectors and batching is off. All the information comes from the simulator.

### The recovered controller

Here `A` denotes joint angles, `V` scaled joint angular velocities, and `P` the cyclic permutation `(x0, x1, x2) -> (x1, x2, x0)`. The scalars `p`, `f`, `u`, and `w` denote torso pitch and scaled forward, vertical, and angular velocities:

```
D = A - 2 P(A) + P(V) - 2 (f + w) + c0
L = -A - (p + w (P(V) - w)) D - c1
R = -P(A) - u + w + c2
action = tanh(P(A) - w + L R)
```

All products are componentwise, with scalars broadcast. The constant vectors are `c0 = (-0.2477, -4.2268, 8.8397)`, `c1 = (0.0645, 0.6181, 0.3969)`, and `c2 = (0.4974, -1.1921, 1.1808)`, rounded here; the replay script retains their full precision.

### Reproducing the result

Replay the controller on twenty held-out starts with twice the training reset noise.

The replay checks whether the hopper stays upright for eight seconds, travels five metres, and completes at least four contact/liftoff cycles. These criteria check sustained hopping beyond the reward used during training.

The result covers small perturbations of one starting pose in simulation. The evaluation does not test obstacles, hardware, or indefinite hopping.

The saved controller was evaluated with Gymnasium 1.2.2 and MuJoCo 3.3.7. Install `gymnasium[mujoco]==1.2.2` and `mujoco==3.3.7`, then run `python examples/hopper_controller.py`.
