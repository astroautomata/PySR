"""Break an affine cipher with symbolic regression over a custom letter type.

The cipher is E(p) = (5p + 8) mod 26 on the 26-letter alphabet. The search sees
only pairs of (ciphertext letter, plaintext letter) and has to discover the
decryption law. Letters are a custom type rather than floats, so the operators are
modular letter arithmetic and the constants in the printed expressions are letters.
"""

import numpy as np

from pysr import PySRRegressor, TypeSpec

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLAIN = "THELAWSOFPHYSICSAREWRITTENINTHELANGUAGEOFMATHEMATICS"
CIPHER = "".join(ALPHABET[(5 * ALPHABET.index(c) + 8) % 26] for c in PLAIN)

X = np.array([[ALPHABET.index(c)] for c in CIPHER], dtype=object)
y = np.array([ALPHABET.index(c) for c in PLAIN], dtype=object)

# A letter is an integer code in 0:25. Mutation walks one letter up or down, and
# the string form prints the letter itself, so a constant reads as `V`, not 21.
SPEC = TypeSpec(
    "Letter",
    fields={"code": "Int"},
    sample="rng -> Letter(rand(rng, 0:25))",
    mutate="(rng, value, temperature) -> Letter(mod(value.code + rand(rng, (-1, 1)), 26))",
    string="value -> string(Char('A' + value.code))",
)

# Distance in letter codes between the prediction and the target, which PySR averages
# over the rows. It is zero exactly when all 52 letters are right, and a hit-or-miss
# count would score a candidate one letter off the same as one twenty off.
LETTER_LOSS = "letter_loss(prediction::Letter, target::Letter)::Float64 = abs(prediction.code - target.code)"

MODEL_KWARGS = dict(
    type_spec=SPEC,
    # Modular arithmetic on letters: every operator stays inside the alphabet.
    operators={
        2: [
            "shift(a::Letter, b::Letter) = Letter(mod(a.code + b.code, 26))",
            "unshift(a::Letter, b::Letter) = Letter(mod(a.code - b.code, 26))",
            "mix(a::Letter, b::Letter) = Letter(mod(a.code * b.code, 26))",
        ]
    },
    elementwise_loss=LETTER_LOSS,
    niterations=20,
    deterministic=True,
    parallelism="serial",
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
    print(f"exact:     {check(model)}")


if __name__ == "__main__":
    main()
