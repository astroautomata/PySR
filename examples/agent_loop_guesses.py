"""Close an agent loop around PySR: read the front, write the next equation, search again.

The data is 160 measurements of kinetic energy against mass and velocity in natural units,
at velocities high enough that the Newtonian form is only an approximation. A cold search
climbs a ladder of polynomials in velocity and stops three to four orders of magnitude
above the measurement noise. Reading the front says what is missing: the relativistic
law, passed back in through `guesses=`.
"""

import numpy as np

from pysr import PySRRegressor

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

MODEL_KWARGS = dict(
    operators={1: ["sqrt"], 2: ["+", "-", "*", "/"]},
    niterations=10,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

# The filmed session handed the front to a coding agent, which replied with this. Each
# entry is one round's proposal, so the loop stops when the list runs out.
PROPOSALS = ["(mass / sqrt(1 - (velocity * velocity))) - mass"]


def agent(front, tried):
    """Stand in for the model call: show the front, return the next expression, or None.

    A real loop prompts a model right here with the front printed below and the list of
    expressions already tried, and parses one expression out of the reply.
    """
    print("front handed to the agent:")
    print(
        front[["complexity", "loss", "equation"]].to_string(index=False),
        "\n",
        flush=True,
    )
    return PROPOSALS[len(tried)] if len(tried) < len(PROPOSALS) else None


def check(model):
    """The search reached the measurement noise floor."""
    return bool(model.equations_["loss"].min() <= 2 * NOISE_FLOOR)


def main(seed=0):
    guesses, tried, rounds = None, [], []
    while True:
        model = PySRRegressor(**MODEL_KWARGS, guesses=guesses, random_state=seed)
        model.fit(X, y, variable_names=VARIABLE_NAMES)
        front = model.equations_
        best = front.loc[front["loss"].idxmin()]
        print(
            f"round {len(rounds)}  guess: {guesses[0] if guesses else 'none (cold start)'}"
        )
        print(f"          loss {best['loss']:.4g} at complexity {best['complexity']}")
        print(f"          {best['equation']}\n", flush=True)
        rounds.append(
            dict(
                guess=guesses[0] if guesses else None,
                loss=float(best["loss"]),
                complexity=int(best["complexity"]),
                equation=str(best["equation"]),
            )
        )
        if check(model):
            break
        proposal = agent(front, tried)
        if proposal is None:
            break
        tried.append(proposal)
        guesses = [proposal]  # single output: a plain list of expression strings

    print(front[["complexity", "loss", "equation"]].to_string(index=False))
    print(f"\nnoise floor {NOISE_FLOOR:.4g}; reached: {check(model)}")
    return rounds


if __name__ == "__main__":
    main()
