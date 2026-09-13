"""Affine cipher with the modulus withheld: 26 has to be learned, not declared.

The data is 52 (ciphertext, plaintext) letter-code pairs from E(p) = (5p + 8) mod 26.
The search sees only the codes. The value type is a plain integer in 0:51 and the three
operators are integer `+`, `*` and `mod` with nothing folded in, so the decryption law
`mod(21*x0 + 14, 26)` has to be written out of three learned constants, the 26 among
them.
"""

import numpy as np

from pysr import PySRRegressor, TypeSpec

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A, B = 5, 8
PLAIN = "THELAWSOFPHYSICSAREWRITTENINTHELANGUAGEOFMATHEMATICS"
CIPHER = "".join(ALPHABET[(A * ALPHABET.index(c) + B) % 26] for c in PLAIN)

X = np.array([[ALPHABET.index(c)] for c in CIPHER], dtype=object)
y = np.array([ALPHABET.index(c) for c in PLAIN], dtype=object)

# The sampled constant range. 26 must not be the only value that fits in it, so the
# range runs to twice the alphabet and every modulus up to 51 is reachable.
CMAX = 51

# Constant-proposal jump rate: the share of proposals that resample the whole range
# instead of stepping by one. One of the three constants is a modulus, and 25 or 27 in
# that slot scrambles the output as badly as any other wrong value, so a walk alone
# cannot find it.
JUMP = 0.35

SPEC = TypeSpec(
    "Num",
    fields={"n": "Int"},
    sample=f"rng -> Num(rand(rng, 0:{CMAX}))",
    mutate=f"(rng, value, temperature) -> Num(rand(rng) < {JUMP} ? rand(rng, 0:{CMAX}) : clamp(value.n + rand(rng, (-1, 1)), 0, {CMAX}))",
    string="value -> string(value.n)",
)

# Integer arithmetic, no alphabet size anywhere. `mod` needs a total extension because
# a subtree can evaluate to zero; a zero modulus returns the dividend, which keeps the
# operator total without making any candidate look better than it is.
OPERATORS = {
    2: [
        "add(a::Num, b::Num) = Num(a.n + b.n)",
        "mul(a::Num, b::Num) = Num(a.n * b.n)",
        "modulo(a::Num, b::Num) = Num(b.n == 0 ? a.n : mod(a.n, b.n))",
    ]
}

# Absolute distance between the candidate's number and the target letter code, which
# PySR averages over the rows. It is zero exactly when every one of the 52 letters is
# right, and a hit-or-miss score would give a candidate that is one off the same loss as
# one that is twenty off, leaving the constant search nothing to climb.
NUM_LOSS = (
    "num_loss(prediction::Num, target::Num)::Float64 = abs(prediction.n - target.n)"
)

MODEL_KWARGS = dict(
    type_spec=SPEC,
    operators=OPERATORS,
    elementwise_loss=NUM_LOSS,
    niterations=1000,
    maxsize=12,
    should_optimize_constants=False,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


def decode(model):
    return "".join(ALPHABET[int(v) % 26] for v in model.predict(X))


def check(model):
    """The whole message comes back, letter for letter."""
    return decode(model) == PLAIN


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y)

    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    print(f"\ndecoded:   {decode(model)}")
    print(f"plaintext: {PLAIN}")
    print(f"exact: {check(model)}")


if __name__ == "__main__":
    main()
