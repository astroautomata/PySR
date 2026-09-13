"""Search for a cart-pole swing-up policy through a custom expression-level objective.

There is no target column. The loss takes a whole candidate expression, drives a cart-pole
plant with it for ten simulated seconds from each of sixteen starts, and returns minus the
mean per-step reward. The reward pays for an upright pole and charges for cart offset,
motor effort and jerk. A rollout that leaves the numerical-divergence box scores a
sentinel.

Most of the starts hang the pole straight down, so a policy has to pump energy in, catch
the pole at the top and hold it there, all from one closed-form expression of the five
observations. The expression returns a force in units of the ten-newton actuator cap.

Reward-shaped losses go negative, which the default logarithmic complexity scaling cannot
represent, so the search uses linear loss scaling.

Success is the champion earning positive mean reward on all sixty-four held-out starts,
which are drawn wider than the training starts in every coordinate.
"""

import numpy as np
import sympy

from pysr import PySRRegressor

M_C, M_P, LENGTH, GRAVITY = 1.0, 0.1, 0.5, 9.8
FORCE_CAP, DT = 10.0, 0.02
RAIL, V_SCALE, OMEGA_SCALE = 2.4, 4.0, 8.0
HORIZON = 500  # 10 s of control at dt = 0.02


def _starts(seed, rows, groups):
    """Starts drawn one coordinate at a time, in the order the film registered them."""
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    out = [(x, 0.0, np.pi, 0.0) for x in rows]
    for count, x_limit, centre, theta_limit, omega_limit in groups:
        for _ in range(count):
            out.append(
                (
                    rng.uniform(-x_limit, x_limit),
                    rng.uniform(-x_limit, x_limit),
                    centre + rng.uniform(-theta_limit, theta_limit),
                    rng.uniform(-omega_limit, omega_limit),
                )
            )
    return np.asarray(out, dtype=float)


TRAIN_STARTS = _starts(
    20260819,
    (0.0, 0.25),
    (
        (6, 0.35, np.pi, 0.35, 0.5),
        (2, 0.35, np.pi / 2, 0.20, 0.5),
        (2, 0.35, -np.pi / 2, 0.20, 0.5),
        (4, 0.20, 0.0, 0.18, 0.5),
    ),
)
HELD_OUT_STARTS = _starts(
    20260820,
    (-0.40, -0.15, 0.15, 0.40),
    (
        (28, 0.55, np.pi, 0.70, 1.0),
        (8, 0.55, np.pi / 2, 0.35, 1.0),
        (8, 0.55, -np.pi / 2, 0.35, 1.0),
        (16, 0.30, 0.0, 0.25, 1.0),
    ),
)


def step(state, force):
    """One semi-implicit Euler action interval for a batch of cart-pole states."""
    x, v, theta, omega = state.T
    total = M_C + M_P
    sin, cos = np.sin(theta), np.cos(theta)
    q = (force + M_P * LENGTH * omega**2 * sin) / total
    theta_dd = (GRAVITY * sin - cos * q) / (LENGTH * (4 / 3 - M_P * cos**2 / total))
    x_dd = q - M_P * LENGTH * theta_dd * cos / total
    v, omega = v + DT * x_dd, omega + DT * theta_dd
    return np.stack([x + DT * v, v, theta + DT * omega, omega], axis=-1)


def observe(state):
    """The five policy inputs: scaled cart offset and speed, the pole as a unit vector,
    and scaled pole rate. They encode the four physical coordinates without loss."""
    x, v, theta, omega = state.T
    return np.stack(
        [x / RAIL, v / V_SCALE, np.sin(theta), np.cos(theta), omega / OMEGA_SCALE],
        axis=-1,
    )


X = observe(TRAIN_STARTS)
y = np.zeros(len(X))
VARIABLE_NAMES = ["x_n", "v_n", "s", "c", "omega_n"]

CARTPOLE_REWARD = """
function cartpole_reward(ex, dataset::Dataset{T,L}, options)::L where {T,L}
    n = size(dataset.X, 2)
    x     = %(rail)r .* Float64.(dataset.X[1, :])
    v     = %(v_scale)r .* Float64.(dataset.X[2, :])
    theta = atan.(Float64.(dataset.X[3, :]), Float64.(dataset.X[4, :]))
    omega = %(omega_scale)r .* Float64.(dataset.X[5, :])

    obs = Matrix{Float64}(undef, 5, n)
    previous = zeros(Float64, n)
    total = %(m_c)r + %(m_p)r
    earned = 0.0
    for _ in 1:%(horizon)d
        sin_t, cos_t = sin.(theta), cos.(theta)
        obs[1, :] .= x ./ %(rail)r
        obs[2, :] .= v ./ %(v_scale)r
        obs[3, :] .= sin_t
        obs[4, :] .= cos_t
        obs[5, :] .= omega ./ %(omega_scale)r
        raw, ok = eval_tree_array(ex, obs, options)
        # A batched evaluation is only usable if it returned one force per start.
        (ok && length(raw) == n && all(isfinite, raw)) || return L(Inf)

        force = %(cap)r .* clamp.(raw, -1.0, 1.0)
        earned += sum(
            2.0 .* cos_t .- 0.05 .* x .^ 2 .-
            0.01 .* (force ./ %(cap)r) .^ 2 .-
            0.005 .* ((force .- previous) ./ %(cap)r) .^ 2
        )
        previous = force

        q = (force .+ %(m_p)r * %(length)r .* omega .^ 2 .* sin_t) ./ total
        theta_dd = (%(gravity)r .* sin_t .- cos_t .* q) ./
                   (%(length)r .* (4 / 3 .- %(m_p)r .* cos_t .^ 2 ./ total))
        x_dd = q .- %(m_p)r * %(length)r .* theta_dd .* cos_t ./ total
        v .+= %(dt)r .* x_dd
        omega .+= %(dt)r .* theta_dd
        x .+= %(dt)r .* v
        theta .+= %(dt)r .* omega

        all(isfinite, x) && all(isfinite, v) && all(isfinite, omega) &&
            maximum(abs, x) <= 20.0 && maximum(abs, v) <= 100.0 &&
            maximum(abs, omega) <= 100.0 || return L(1.0e12)
    end
    return L(-earned / (%(horizon)d * n))
end
""" % dict(
    m_c=M_C,
    m_p=M_P,
    length=LENGTH,
    gravity=GRAVITY,
    cap=FORCE_CAP,
    dt=DT,
    rail=RAIL,
    v_scale=V_SCALE,
    omega_scale=OMEGA_SCALE,
    horizon=HORIZON,
)

MODEL_KWARGS = dict(
    operators={
        1: ["square", "abs", "tanh"],
        2: ["+", "-", "*", "/", "max", "min"],
        3: ["ifelse(t, a, b) = t > 0 ? a : b"],
    },
    extra_sympy_mappings={
        "ifelse": lambda t, a, b: sympy.Piecewise((a, t > 0), (b, True))
    },
    loss_function_expression=CARTPOLE_REWARD,
    loss_scale="linear",  # rewards make the loss negative, which log scaling forbids
    maxsize=28,
    maxdepth=10,
    parsimony=0.001,
    niterations=120,
    populations=32,
    population_size=64,
    parallelism="multithreading",
    verbosity=0,
)


def mean_rewards(model, starts, index=None):
    """Per-start mean reward over the full horizon, with the training divergence rule."""
    state = np.array(starts, dtype=float)
    previous = np.zeros(len(state))
    earned = np.zeros(len(state))
    alive = np.ones(len(state), dtype=bool)
    for _ in range(HORIZON):
        raw = np.asarray(model.predict(observe(state), index=index), dtype=float)
        force = FORCE_CAP * np.clip(np.nan_to_num(raw, nan=0.0), -1.0, 1.0)
        earned += np.where(
            alive,
            2.0 * np.cos(state[:, 2])
            - 0.05 * state[:, 0] ** 2
            - 0.01 * (force / FORCE_CAP) ** 2
            - 0.005 * ((force - previous) / FORCE_CAP) ** 2,
            0.0,
        )
        previous = force
        state = step(state, force)
        alive &= np.isfinite(state).all(axis=-1)
        alive &= np.abs(state[:, 0]) <= 20.0
        alive &= np.abs(state[:, 1]) <= 100.0
        alive &= np.abs(state[:, 3]) <= 100.0
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(alive, earned / HORIZON, -1.0e12)


def check(model):
    """The champion earns positive mean reward on every one of the 64 held-out starts."""
    index = int(model.equations_["loss"].idxmin())
    return bool((mean_rewards(model, HELD_OUT_STARTS, index=index) > 0.0).all())


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))

    index = int(model.equations_["loss"].idxmin())
    row = model.equations_.loc[index]
    print(f"\nchampion (complexity {row.complexity}, loss {row.loss:+.6f}):")
    print(f"  {row.equation}")
    returns = mean_rewards(model, HELD_OUT_STARTS, index=index)
    print(f"held-out positive: {int((returns > 0).sum())}/64 starts")


if __name__ == "__main__":
    main()
