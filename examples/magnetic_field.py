"""Recover a magnetic field from force measurements alone.

A charged particle obeys F = -eta(T) v + v x B(t), and 1000 experiments record only
the total force and the inputs (t, v, T). A TemplateExpressionSpec supplies that
structure and a Force type spec carries the three-component data, so the search has
only to fill in B_x(t), B_y(t), B_z(t) and the drag scale eta(T).
"""

import numpy as np

from pysr import PySRRegressor, TemplateExpressionSpec, TypeSpec

OMEGA = 2 * np.pi


def experiments(n, seed):
    rng = np.random.default_rng(seed)
    T = 298.15 + 0.5 * rng.random(n)
    t = 10.0 * rng.random(n)
    v = 2.0 * rng.random((n, 3)) - 1.0
    B = np.stack([np.sin(OMEGA * t), np.cos(OMEGA * t), np.exp(-t / 10.0)], axis=1)
    F = -1e-5 * np.sqrt(T)[:, None] * v + np.cross(v, B)
    return np.column_stack([t, v[:, 0], v[:, 1], v[:, 2], T]), F


def as_forces(rows):
    """An (n, 3) float array as a column of n Force values."""
    out = np.empty(len(rows), dtype=object)
    out[:] = [tuple(map(float, row)) for row in rows]
    return out


def on_diagonal(columns):
    """Every value in the problem is a Force, so scalar inputs ride the diagonal."""
    out = np.empty(columns.shape, dtype=object)
    for j in range(columns.shape[1]):
        out[:, j] = as_forces(np.repeat(columns[:, j, None], 3, axis=1))
    return out


INPUTS, FORCES = experiments(1000, seed=0)
X = on_diagonal(INPUTS)
y = as_forces(FORCES)
VARIABLE_NAMES = ["t", "v_x", "v_y", "v_z", "T"]

HELD_OUT_INPUTS, HELD_OUT_FORCES = experiments(500, seed=12345)
HELD_OUT_X = on_diagonal(HELD_OUT_INPUTS)

# The one-argument constructor is the diagonal embedding of a scalar. It also gives
# the evaluator the Force(Inf) sentinel it substitutes for an invalid value. A constant
# is a whole vector: all three slots are sampled and all three are handed to BFGS, so
# the spec stays usable when an expression itself has to return a direction.
FORCE = TypeSpec(
    "Force",
    fields={"x": "Float64", "y": "Float64", "z": "Float64"},
    sample="""begin
        Force(u::Real) = Force(u, u, u)
        rng -> Force(randn(rng, 3)...)
    end""",
    scalar_constants="value -> [value.x, value.y, value.z]",
    with_scalar_constants="(value, c) -> Force(c[1], c[2], c[3])",
    string="value -> sprint(show, (value.x, value.y, value.z); context = :compact => true)",
)


def _method(body, name):
    return body + "\n" + name


OPERATORS = {
    1: [
        _method(
            "Base.sin(a::Force)::Force = Force(sin(a.x), sin(a.y), sin(a.z))",
            "Base.sin",
        ),
        _method(
            "Base.cos(a::Force)::Force = Force(cos(a.x), cos(a.y), cos(a.z))",
            "Base.cos",
        ),
        _method(
            """function Base.sqrt(a::Force)::Force
            f(v) = v < 0 ? NaN : sqrt(v)
            return Force(f(a.x), f(a.y), f(a.z))
            end""",
            "Base.sqrt",
        ),
        _method(
            "Base.exp(a::Force)::Force = Force(exp(a.x), exp(a.y), exp(a.z))",
            "Base.exp",
        ),
    ],
    2: [
        _method(
            "Base.:+(a::Force, b::Force)::Force = Force(a.x + b.x, a.y + b.y, a.z + b.z)",
            "Base.:+",
        ),
        _method(
            "Base.:-(a::Force, b::Force)::Force = Force(a.x - b.x, a.y - b.y, a.z - b.z)",
            "Base.:-",
        ),
        _method(
            "Base.:*(a::Force, b::Force)::Force = Force(a.x * b.x, a.y * b.y, a.z * b.z)",
            "Base.:*",
        ),
        _method(
            "Base.:/(a::Force, b::Force)::Force = Force(a.x / b.x, a.y / b.y, a.z / b.z)",
            "Base.:/",
        ),
    ],
}

FORCE_LOSS = "force_loss(a::Force, b::Force)::Float64 = (a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2"

PHYSICS = r"""
begin
    _B_x = B_x(t)
    _B_y = B_y(t)
    _B_z = B_z(t)
    _F_d_scale = F_d_scale(T)
    if !(_B_x.valid && _B_y.valid && _B_z.valid && _F_d_scale.valid)
        return ValidVector(_B_x.x, false)
    end
    F = map(_B_x.x, _B_y.x, _B_z.x, _F_d_scale.x,
            v_x.x, v_y.x, v_z.x) do bx, by, bz, fd, ux, uy, uz
        b1, b2, b3 = bx.x, by.x, bz.x
        u1, u2, u3 = ux.x, uy.x, uz.x
        s = fd.x
        Force(u2 * b3 - u3 * b2 + s * u1,
              u3 * b1 - u1 * b3 + s * u2,
              u1 * b2 - u2 * b1 + s * u3)
    end
    ValidVector(F, true)
end
"""

STRUCTURE = TemplateExpressionSpec(
    combine=PHYSICS,
    expressions=["B_x", "B_y", "B_z", "F_d_scale"],
    variable_names=VARIABLE_NAMES,
)

MODEL_KWARGS = dict(
    type_spec=FORCE,
    expression_spec=STRUCTURE,
    operators=OPERATORS,
    elementwise_loss=FORCE_LOSS,
    niterations=100,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


def held_out_error(model):
    predicted = np.array([list(f) for f in model.predict(HELD_OUT_X)], dtype=float)
    return float(np.abs(predicted - HELD_OUT_FORCES).max())


def check(model):
    """Forces on unseen experiments reproduced to machine precision."""
    return held_out_error(model) < 1e-6


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    print(f"\nheld-out max abs error: {held_out_error(model):.3g}")


if __name__ == "__main__":
    main()
