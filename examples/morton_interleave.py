"""Rediscover a bit-exact Morton (Z-order) interleave of two 4-bit integers.

The search works on a machine word: a Word value backed by UInt32 whose vocabulary is
the integer instruction set (xor, and, or, not, shift left, shift right), with shift
distances as ordinary Word constants it has to find. It scores all 256 four-bit input
pairs and has to recover the map that interleaves their bits.

The score only ever sees those 256 pairs, so what comes back is an expression that
agrees with the interleave everywhere it was asked. That is a weaker claim than the
general Z-order map, and the difference is measurable: the recovered forms are exact on
every 4-bit pair and correct on only 256 of the 1024 5-bit pairs, which are exactly the
4-bit ones. The run prints both numbers.

Two things about the shape of the answer are worth knowing before reading it. Written in
this vocabulary as a tree, with no shared subexpressions, the textbook interleave kernel
(two shift-mask spread stages per input, then one shift and or) is 41 nodes, so a
recovered form near 40 is the same size as the hand-written one rather than bloat, and
maxsize has to leave room for it. And the masks print as arbitrary hex because only
their low bits can affect an 8-bit result: in 0x00080022 the search pinned bits 1 and 5
and bit 19 is free, since no scored pair can tell what it holds.
"""

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
VARIABLE_NAMES = ["x", "y"]

SPEC = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
)

MODEL_KWARGS = dict(
    type_spec=SPEC,
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
    verbosity=0,
)


def exact_front_rows(model):
    """Indices of front members that reproduce every one of the 256 interleaves."""
    target = np.array([int(v) for v in y])
    rows = []
    for i in range(len(model.equations_)):
        prediction = np.array([int(v) for v in model.predict(X, index=i)])
        if (prediction == target).all():
            rows.append(i)
    return rows


def check(model):
    """Success is an exact interleaver anywhere on the Pareto front."""
    return bool(exact_front_rows(model))


def wider_agreement(model, index):
    """How many 5-bit pairs the recovered expression interleaves correctly."""
    pairs = [(a, b) for b in range(1 << WIDER) for a in range(1 << WIDER)]
    inputs = np.array(pairs, dtype=object)
    expected = np.array([morton(a, b, WIDER) for a, b in pairs])
    got = np.array([int(v) for v in model.predict(inputs, index=index)])
    return int((got == expected).sum()), len(pairs)


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    rows = exact_front_rows(model)
    if rows:
        best = model.equations_.iloc[rows[0]]
        print(f"\nexact interleaver at complexity {int(best.complexity)}:")
        print(best.equation)
        agree, total = wider_agreement(model, rows[0])
        print(
            f"\non {WIDER}-bit pairs it interleaves {agree}/{total} correctly, so it is a {WIDTH}-bit interleaver and not the Z-order map"
        )
    else:
        print("\nno exact interleaver on the front")


if __name__ == "__main__":
    main()
