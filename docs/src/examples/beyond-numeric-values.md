# Beyond numeric values

Every example on this page searches over a value that is not a number: letters,
cell states, machine words, and drawings. They all rest on the same mechanism,
so read [Custom value types](/examples/value-types#custom-value-types) first if
you want the rules a `TypeSpec` has to satisfy.

## Breaking an affine cipher with a letter type

Breaking a cipher is a good place to start with custom value types, because the
only thing this example customizes is the value type itself: `type_spec`, the
`operators` that act on that type, the loss, and `niterations`. Everything else
is a default.

The cipher is the classic affine map on the 26-letter alphabet,

$$ E(p) = (5p + 8) \bmod 26, $$

where each letter is identified with its position in `A` to `Z`. The search
never sees that formula. It sees only pairs of (ciphertext letter, plaintext
letter) and has to find the decryption law itself.

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

A letter is an integer code in `0:25`. Mutation walks one letter up or down so
that evolution explores the alphabet locally, and the `string` hook is what
makes the Pareto front readable: a constant prints as `V` rather than `21`.
Without that hook you would recover the same law and then have to translate it
back into letters by hand.

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

The operators are modular letter arithmetic, so every intermediate value stays
inside the alphabet and no expression the search writes can leave the type. Note
that the modulus 26 lives in the operator definitions here, which is what makes
this the easy version of the problem.

The loss is the distance in letter codes between prediction and target, which PySR
averages over the rows. It is zero exactly when all 52 letters are right, and a
hit-or-miss count would score a candidate one letter off the same as one twenty
off. Because it looks at one row at a time it goes in `elementwise_loss`, and the
backend handles evaluation and invalid candidates:

```python
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

A full `loss_function` is for objectives that need the whole dataset at once, like
the rollout in [Swinging up a cart-pole](/examples/objectives#swinging-up-a-cart-pole-with-a-rollout-objective)
or the period certificate in
[Inventing a pseudorandom generator](/examples/objectives#inventing-a-pseudorandom-generator-with-no-target).
Row-at-a-time scoring does not need it.

Every example in this group pins `deterministic=True`, `parallelism="serial"`,
and `random_state`, so a rerun reproduces the same front.

The search reaches loss 0 at complexity 5 on all 8 seeds we ran, in 4.0 to 4.6
seconds each once Julia has compiled, which the first fit of a session pays at
about 37 seconds. The recovered expression is `mix(V, shift(S, x0))`. Reading the
letters as codes, `V` is 21 and `S` is 18, so this is $21(x_0 + 18) \bmod 26$,
and 21 is the multiplicative inverse of 5 modulo 26. Decoding the ciphertext
through the champion returns all 52 letters of the plaintext.

### Learning the modulus too

The variant in the film withholds the modulus. Instead of modular letter
operators it gives the search plain integer add, multiply, and modulo over a
value type spanning `0:51`, so 26 has to be discovered as a constant rather than
declared in the operators. That is a harder search over a larger space, and it
is the version to reach for if you want the law and its modulus recovered
together.

Withholding the modulus makes the search much harder, and the outcome is
all-or-nothing: either all 52 letters come back or the message is garbage. At
`niterations=1000`, about three minutes, the law comes back on roughly three
runs in five, usually at complexity 7, as

```
modulo(mul(add(18, x0), 47), 26)
```

which is $47(x_0 + 18) \bmod 26$, and since $47 \cdot 18 = 846 \equiv 14$, the
same as $47x_0 + 14 \bmod 26$. Runs that miss settle around loss 0.25 to 1.77
and stay there rather than closing in.

The full runnable script for the section above is `examples/affine_cipher.py`.
The withheld-modulus variant is `examples/affine_freemod.py`, the harder search
described here, where a single run may or may not land the law.

## Rediscovering Conway's Game of Life

Conway's Game of Life is usually stated as a rule: a cell is born when it has
exactly three live neighbours, and a live cell survives with two or three. Let's
throw that statement away and hand PySR only the transition table, then see
whether it can write the rule back down.

There are 18 distinct transition states, one for every combination of a cell state
and a neighbour count from 0 to 8. Eight copies of that table give the search 144
rows to score against, so a rule that misses one transition state pays a loss of
1/18. The values are stored as `object` arrays because each entry will be wrapped
in a custom Julia type:

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

Both the cell state and the neighbour count are small integers, and so is the
output, so a single type covers every value flowing through the expression. We
define it with a `TypeSpec` whose payload is one `Int`. The `sample` hook draws a
count in `0:8`, and `mutate` nudges an existing constant up or down by one, wrapping
modulo 9. That keeps the search moving through neighbouring integers instead of
resampling blindly, which is what makes finding the thresholds cheap:

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

The vocabulary is deliberately logical: `not`, `and`, `or`, plus an integer
equality test `eq`. There is no arithmetic and no comparison against a threshold,
so the only way to talk about a neighbour count is to compare it to a constant
the search has to discover. The loss is zero-one on exact agreement, since a
truth table has no notion of being close:

```python
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

Nothing else is customized. `deterministic=True` with `parallelism="serial"` and a
fixed `random_state` makes the run reproducible; the other examples on this page
use the same three settings for the same reason.

Across five seeds, all five recover the rule exactly at complexity 9, taking
between 37 and 73 seconds each. Up to the argument order of the commutative
operators, every seed lands on the same expression:

```
or(eq(n, 3), and(eq(n, 2), alive))
```

It reads as the textbook statement: born when the neighbour count is 3, survives
when the cell is alive and the count is 2. The `eq` calls carry the integer
constants 3 and 2 that the search had to locate on its own, and it agrees with the
target on all 18 transition states, which is all 144 rows.

The front shows the rule arriving in two pieces. At complexity 3 the best
expression is `eq(n, 3)`, the birth condition alone, already correct on 17 of the
18 states at a loss of 1/18. The state it misses is the live cell with two
neighbours, and the survival clause that appears at complexity 9 is what closes
that gap.

One note on `niterations`. The count is not comparable across population layouts,
since one iteration is a fixed number of cycles over every population. With PySR's
population and cycle defaults left alone each iteration does a lot of work, and 160
of them close the search on every seed.

The full runnable script is `examples/game_of_life.py`.

## Searching over machine words

This example searches for a bit-exact interleave of two 4-bit integers, the operation at
the heart of Morton (Z-order) indexing. The value flowing through every expression is a
machine word rather than a float, and the vocabulary is the integer instruction set: xor,
and, or, not, shift left, shift right. Shift distances are ordinary `Word` constants that
the search has to find for itself.

The target comes from a reference interleave, evaluated on all 256 pairs of 4-bit inputs.

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

The `TypeSpec` declares a `Word` wrapping a `UInt32`. The `sample` hook is biased toward
small values, because most useful constants here are shift distances below 32, while the
rest of the range still has to be reachable for masks. The `mutate` hook either flips a
single bit or steps the word by a random amount, which is how a mask gets refined one bit
at a time. The `string` hook prints small words as decimals and everything else as
zero-padded hex, so shift distances and masks read differently:

```python
spec = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
)
```

Each operator is one machine instruction lifted to `Word`. The loss is the fraction of the
32 bits that differ between prediction and target, so a loss of exactly zero means the
expression reproduces every scored output bit for bit:

```python
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

`maxsize=45` is chosen to fit the answer rather than to be generous. Written as a tree in
this vocabulary with no shared subexpressions, the textbook interleave kernel (two
shift-mask spread stages per input, then one shift and an or) is 41 nodes at width 4, so
the budget has to clear that. Every script in this section pins `deterministic=True`,
`parallelism="serial"`, and `random_state`, which makes a run reproducible at the cost of
using one thread.

On PySR 2.1.0 this recovers an exact interleaver on 5 of 5 seeds, taking 189 to 222 s per
seed, at complexity 37 to 41. Complexity 37 is smaller than the 41-node reference form, so
the size of the answer is not bloat. Neither are the hex masks a defect: the hand-written
kernel carries `0x33` and `0x55`, and here the high bits of a 32-bit mask are simply
unconstrained, since no scored pair depends on them.

The important caveat is what "exact" covers. The score only ever sees the 256 four-bit
pairs, and on that domain the recovered expression agrees with Morton everywhere. It is a
width-specific interleaver that coincides with Z-order on the scored domain, and not the
width-independent Z-order algorithm. Replaying the winner on all 1024 pairs of 5-bit
inputs makes the difference measurable: a width-4 winner scores 256 of 256 at width 4 and
256 of 1024 at width 5. That number is the one to quote, since it supports the claim that
the search recovered a 4-bit interleaver and refutes the stronger reading that it found the
general algorithm.

Width 5 is out of reach here for a structural reason. The kernel needs three spread stages
at that width, 89 nodes, and `maxsize=45` excludes it, so no exact solution exists anywhere
in the search space. Getting there requires `maxsize >= 89`.

The full runnable script is `examples/morton_interleave.py`.

## Turtle graphics: searching over drawings

Usually the value flowing through the expression tree is a number, a vector, or a
string. Here it is a drawing. A value is a list of pen commands, `forward` and
`turn`, and the operators move the pen, reflect it, and glue command lists
together. That makes every expression a program, and its value is the picture that
program draws, as the film clip shows.

A drawing is carried as two parallel vectors: `kind`, where `0` means forward and
`1` means turn, and `val`, holding either a length or an angle in radians. The
target is the outline of a plus sign, four copies of a three-edge arm. The data is
one row whose only feature is `f`, one unit step forward, so everything else a
program needs comes from the operators and from turns the search invents:

```python
ARM = [("F", 1.0), ("T", 90.0), ("F", 1.0), ("T", -90.0), ("F", 1.0), ("T", 90.0)]
TARGET = ARM * 4

X = np.empty((1, 1), dtype=object)
X[0, 0] = ([0], [1.0])
y = np.empty(1, dtype=object)
y[0] = path(TARGET)
VARIABLE_NAMES = ["f"]
```

### Scoring a picture

Two programs that draw the same figure need not agree command by command, so the
loss compares pictures rather than command lists. Both drawings are rasterised
onto a 96 by 96 grid after their bounding boxes are centred and their larger
extent is scaled to the grid, which leaves the comparison sensitive to shape and
blind to position and size. The score is the symmetric chamfer distance between
the two lit pixel sets, in pixels, divided by the grid width, so a loss of
exactly zero means the two drawings lit the same pixels. The distances come from
an exact squared Euclidean distance transform, and the target's raster and
transform are computed once and cached. The grid is offset by `0.1373` of a
pixel, because an axis-aligned figure otherwise lands exactly on rounding ties,
where a one-ulp change in an angle flips a pixel and the loss stops being
reproducible.

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

The `TypeSpec` wraps the two vectors in a `Path`. A sampled constant is one turn
at any real angle in the interval from -180 to 180 degrees, and `mutate` either
perturbs one of a value's turns by a Gaussian step of 30 degrees scaled by the
temperature or resamples that turn outright. Every turn is also exposed to the
constant optimiser through the scalar-constant hook pair, which lets BFGS polish
an angle that is nearly right. Constant folding can collapse an all-constant
subtree into a value of several commands, so `mutate` and `string` both work
command by command, and a folded constant prints as `p[...]`:

```python
PATH = TypeSpec(
    "Path",
    fields={"kind": "Vector{Int8}", "val": "Vector{Float64}"},
    preamble=PREAMBLE,
    sample="rng -> Path(Int8[1], [_real(rng)])",
    mutate="""(rng, value, temperature) -> begin
        turns = findall(==(Int8(1)), value.kind)
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

The vocabulary is five operators, all of them program combinators. `seq` runs one
sub-program after another. `dup` and `tri` run a sub-program twice and three
times. `mir` negates every turn, reflecting a sub-program. `nest` is an L-system
substitution: it replaces every forward by a copy of the whole path, scaled so
the copy's net displacement equals that forward. Repetition and reflection let a
short program draw a figure with many strokes:

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

`niterations` is the only budget knob set; `maxsize` is left at PySR's default.
The reproducibility settings are the same three the other examples on this page
use:

```python
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

Twenty iterations is a small count next to the other examples here, and it is
enough because each candidate is expensive: scoring one program means rasterising
it and distance-transforming 9216 pixels.

On PySR 2.1.0, four of the five seeds found a program that draws the target with
a loss of exactly `0.0`, at complexity 11, 9, 8 and 9. The fifth stopped at a
best loss of 0.0029226382607225 at complexity 16, a drawing that is close to the
cross without matching its pixels. Wall time was 2413.45, 1749.41, 2253.99,
6630.48 and 1699.49 seconds for seeds 0 to 4, so the seed that ran longest is
almost four times the seed that ran shortest. Complexity 9 is the middle of the
exact range and the value two of the four exact seeds reach: because the loss
only counts lit pixels, several tracing orders of the same outline score zero,
and which one a seed reaches decides whether the winner comes out at 8, 9 or 11
nodes.

The spread is the search being stochastic: angles are continuous and the score is
a pixel comparison, so a seed refining a near-miss arm can finish without an
exact program, as seed 4 did.

The complexity-9 winner is

```
dup(dup(mir(seq(t-179.6493032061754, tri(seq(p[t-270.04570985370526,t-270.04570985370526,t-270.04570985370526], f))))))
```

Read from the inside out. The folded constant is three equal turns that compose
to a single quarter turn, and `tri` repeats that turn-then-forward pair three
times, giving three unit edges at right angles. The turn of nearly 180 degrees in
front of it sets up the join, `mir` reflects the arm, and the two `dup`s give four
copies of it, one per arm of the cross. Angles print unrounded because turns here
are continuous values BFGS has polished, and -270.0457 degrees is the same heading
change as 89.9543 degrees.

The full runnable script is `examples/turtle_graphics.py`.
