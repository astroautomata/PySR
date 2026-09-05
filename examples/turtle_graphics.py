"""Search for a turtle program that draws a plus-sign outline.

The value the search carries is a drawing: a list of pen commands, `forward` and
`turn`. The operators move and turn the pen, so every expression is a program and
its value is the picture that program draws. The loss rasterizes both drawings
and compares them with a chamfer distance.
"""

import math

import numpy as np

from pysr import PySRRegressor, TypeSpec

# --------------------------------------------------------------- the turtle
# A Path is a flat command list carried as two parallel vectors: `kind` (0
# forward, 1 turn) and `val` (a length, or an angle in radians).
PREAMBLE = r"""
import Random

const _GRID = 96
const _MARGIN = 3
# Off-centre by an arbitrary fraction of a pixel: axis-aligned figures otherwise
# land exactly on rounding ties, where a one-ulp change of an angle flips a pixel
# and the loss stops being reproducible across implementations.
const _OFFSET = 0.1373
const _MAXCMD = 8192
const _FAIL = 4.0
const _DTCACHE = Dict{UInt64,Tuple{BitMatrix,Matrix{Float64}}}()
const _DTLOCK = ReentrantLock()

_empty() = (Int8[], Float64[])

# ----------------------------------------------------- constants and printing
# The width of the mutation step at temperature 1, in degrees.
const _SIGMA_DEG = 30.0

# One turn, any real angle in (-180, 180].
_real(rng) = deg2rad(180.0 - 360.0 * rand(rng))

function _wrap(deg)
    d = mod(deg + 180.0, 360.0) - 180.0
    return d <= -180.0 ? 180.0 : d
end

_perturb(rng, angle, temperature) =
    deg2rad(_wrap(rad2deg(angle) + _SIGMA_DEG * temperature * Random.randn(rng)))

# The optimizer sees exactly the turns: one Float64 per turn, forwards untouched.
_turns(kind::Vector{Int8}, val::Vector{Float64}) =
    [val[i] for i in eachindex(kind) if kind[i] == Int8(1)]

function _put_turns(kind::Vector{Int8}, val::Vector{Float64}, c)
    out = copy(val)
    j = 0
    for i in eachindex(kind)
        if kind[i] == Int8(1)
            j += 1
            out[i] = c[j]
        end
    end
    return out
end

function _degstr(angle)
    d = rad2deg(angle)
    r = round(d)
    return abs(d - r) < 1e-9 ? string(Int(r)) : string(d)
end

_cmd(k::Int8, v::Float64) =
    k == Int8(1) ? "t" * _degstr(v) : (v == 1.0 ? "f" : "f" * string(v))

# Constant folding turns an all-constant subtree into one multi-command Path, so
# a constant has to print as something that can be read back command by command.
function _literal(kind::Vector{Int8}, val::Vector{Float64})
    length(kind) == 1 && return _cmd(kind[1], val[1])
    return "p[" * join([_cmd(kind[i], val[i]) for i in eachindex(kind)], ",") * "]"
end

# --------------------------------------------------------- the path algebra
function _endpoint(kind::Vector{Int8}, val::Vector{Float64})
    x = 0.0; y = 0.0; c = 1.0; s = 0.0
    @inbounds for i in eachindex(kind)
        if kind[i] == Int8(0)
            x += c * val[i]; y += s * val[i]
        else
            ca = cos(val[i]); sa = sin(val[i])
            c, s = c * ca - s * sa, s * ca + c * sa
        end
    end
    return x, y
end

function _seq(ak::Vector{Int8}, av::Vector{Float64},
              bk::Vector{Int8}, bv::Vector{Float64})
    (isempty(ak) && isempty(bk)) && return _empty()
    length(ak) + length(bk) > _MAXCMD && return _empty()
    return (vcat(ak, bk), vcat(av, bv))
end

function _mir(kind::Vector{Int8}, val::Vector{Float64})
    isempty(kind) && return _empty()
    out = copy(val)
    @inbounds for i in eachindex(kind)
        kind[i] == Int8(1) && (out[i] = -out[i])
    end
    return (copy(kind), out)
end

# L-system self-application: replace every forward by a copy of the whole path,
# scaled so the copy's net displacement equals that forward.
function _nest(kind::Vector{Int8}, val::Vector{Float64})
    isempty(kind) && return _empty()
    nf = count(==(Int8(0)), kind)
    nf == 0 && return _empty()
    n = length(kind)
    n + nf * (n - 1) > _MAXCMD && return _empty()
    x, y = _endpoint(kind, val)
    d = hypot(x, y)
    d < 1e-9 && return _empty()
    ok = Vector{Int8}(); ov = Vector{Float64}()
    sizehint!(ok, n + nf * (n - 1)); sizehint!(ov, n + nf * (n - 1))
    @inbounds for i in eachindex(kind)
        if kind[i] == Int8(0)
            r = val[i] / d
            for j in eachindex(kind)
                push!(ok, kind[j])
                push!(ov, kind[j] == Int8(0) ? val[j] * r : val[j])
            end
        else
            push!(ok, kind[i]); push!(ov, val[i])
        end
    end
    return (ok, ov)
end

# ------------------------------------------------------------------ the loss
# Walk the commands and stamp every drawn segment onto a 96x96 grid, after
# centring the bounding box and scaling its larger extent to the grid, so the
# comparison sees shape and not place or size.
function _raster(kind::Vector{Int8}, val::Vector{Float64})
    isempty(kind) && return nothing
    nf = count(==(Int8(0)), kind)
    nf == 0 && return nothing
    xs = Vector{Float64}(undef, 4 * nf)
    x = 0.0; y = 0.0; c = 1.0; s = 0.0
    j = 0
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    @inbounds for i in eachindex(kind)
        if kind[i] == Int8(0)
            x2 = x + c * val[i]; y2 = y + s * val[i]
            xs[4j + 1] = x; xs[4j + 2] = y; xs[4j + 3] = x2; xs[4j + 4] = y2
            j += 1
            xmin = min(xmin, x, x2); xmax = max(xmax, x, x2)
            ymin = min(ymin, y, y2); ymax = max(ymax, y, y2)
            x = x2; y = y2
        else
            ca = cos(val[i]); sa = sin(val[i])
            c, s = c * ca - s * sa, s * ca + c * sa
        end
    end
    (isfinite(xmin) && isfinite(xmax) && isfinite(ymin) && isfinite(ymax)) ||
        return nothing
    scale = max(xmax - xmin, ymax - ymin)
    (scale < 1e-9 || !isfinite(scale)) && return nothing
    span = _GRID - 1 - 2 * _MARGIN
    cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
    mask = falses(_GRID, _GRID)
    @inbounds for k in 0:(nf - 1)
        u0 = _MARGIN + _OFFSET + ((xs[4k + 1] - cx) / scale + 0.5) * span
        v0 = _MARGIN + _OFFSET + ((xs[4k + 2] - cy) / scale + 0.5) * span
        u1 = _MARGIN + _OFFSET + ((xs[4k + 3] - cx) / scale + 0.5) * span
        v1 = _MARGIN + _OFFSET + ((xs[4k + 4] - cy) / scale + 0.5) * span
        steps = max(1, ceil(Int, 2 * hypot(u1 - u0, v1 - v0)))
        for t in 0:steps
            a = t / steps
            i = clamp(round(Int, u0 + a * (u1 - u0)) + 1, 1, _GRID)
            jj = clamp(round(Int, v0 + a * (v1 - v0)) + 1, 1, _GRID)
            mask[i, jj] = true
        end
    end
    return mask
end

# Exact squared Euclidean distance transform, one dimension at a time.
function _dt1d!(d::Vector{Float64}, f::Vector{Float64}, n::Int,
                v::Vector{Int}, z::Vector{Float64})
    k = 1; v[1] = 1; z[1] = -1e20; z[2] = 1e20
    @inbounds for q in 2:n
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2q - 2v[k])
        while s <= z[k]
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2q - 2v[k])
        end
        k += 1; v[k] = q; z[k] = s; z[k + 1] = 1e20
    end
    k = 1
    @inbounds for q in 1:n
        while z[k + 1] < q
            k += 1
        end
        d[q] = (q - v[k])^2 + f[v[k]]
    end
    return d
end

function _edt(mask::BitMatrix)
    n1, n2 = size(mask)
    big = 1e12
    d2 = Matrix{Float64}(undef, n1, n2)
    @inbounds for j in 1:n2, i in 1:n1
        d2[i, j] = mask[i, j] ? 0.0 : big
    end
    m = max(n1, n2)
    f = Vector{Float64}(undef, m); d = Vector{Float64}(undef, m)
    v = Vector{Int}(undef, m + 1); z = Vector{Float64}(undef, m + 2)
    @inbounds for j in 1:n2
        for i in 1:n1; f[i] = d2[i, j]; end
        _dt1d!(d, f, n1, v, z)
        for i in 1:n1; d2[i, j] = d[i]; end
    end
    @inbounds for i in 1:n1
        for j in 1:n2; f[j] = d2[i, j]; end
        _dt1d!(d, f, n2, v, z)
        for j in 1:n2; d2[i, j] = d[j]; end
    end
    return d2
end

# The target never changes, so its raster and its transform are computed once.
function _target_grids(kind::Vector{Int8}, val::Vector{Float64})
    key = hash(kind, hash(val))
    lock(_DTLOCK) do
        get!(_DTCACHE, key) do
            mask = _raster(kind, val)
            mask === nothing ? (falses(_GRID, _GRID), fill(1e12, _GRID, _GRID)) :
                (mask, _edt(mask))
        end
    end
end

function _mean_to(mask::BitMatrix, d2::Matrix{Float64})
    total = 0.0; n = 0
    @inbounds for i in eachindex(mask)
        if mask[i]
            total += sqrt(d2[i]); n += 1
        end
    end
    return n == 0 ? nothing : total / n
end

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
"""

PATH = TypeSpec(
    "Path",
    fields={"kind": "Vector{Int8}", "val": "Vector{Float64}"},
    preamble=PREAMBLE,
    # A constant is one turn, any real angle in (-180, 180].
    sample="rng -> Path(Int8[1], [_real(rng)])",
    # Perturb one of the value's angles by a temperature-scaled step, or
    # resample it outright. Constant folding can hand this hook a value of
    # several commands, so it moves a command in place.
    mutate="""(rng, value, temperature) -> begin
        turns = findall(==(Int8(1)), value.kind)
        isempty(turns) && return Path(Int8[1], [_real(rng)])
        i = rand(rng, turns)
        val = copy(value.val)
        val[i] = rand(rng, Bool) ? _perturb(rng, val[i], temperature) : _real(rng)
        Path(copy(value.kind), val)
    end""",
    # Every turn in the value is a scalar the constant optimizer may fit.
    scalar_constants="value -> _turns(value.kind, value.val)",
    with_scalar_constants="(value, c) -> Path(copy(value.kind), _put_turns(value.kind, value.val, c))",
    is_valid="value -> !isempty(value.kind) && all(isfinite, value.val)",
    string="value -> _literal(value.kind, value.val)",
)

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


def path(commands):
    """A ('F', length) / ('T', degrees) command list as the two Path vectors."""
    kind = [0 if op == "F" else 1 for op, _ in commands]
    val = [a if op == "F" else math.radians(a) for op, a in commands]
    return kind, val


# The target: a plus-sign outline, four copies of a three-edge arm.
ARM = [("F", 1.0), ("T", 90.0), ("F", 1.0), ("T", -90.0), ("F", 1.0), ("T", 90.0)]
TARGET = ARM * 4

# One row. The only feature is `f`, a single unit step forward.
X = np.empty((1, 1), dtype=object)
X[0, 0] = ([0], [1.0])
y = np.empty(1, dtype=object)
y[0] = path(TARGET)
VARIABLE_NAMES = ["f"]

MODEL_KWARGS = dict(
    type_spec=PATH,
    operators=OPERATORS,
    elementwise_loss=CHAMFER_LOSS,
    niterations=20,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

EXACT = 1e-9


def check(model):
    """Some program on the front draws the target, to floating-point precision."""
    return float(model.equations_["loss"].min()) < EXACT


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    print(model.equations_[["complexity", "loss", "equation"]].to_string(index=False))
    exact = model.equations_[model.equations_["loss"] < EXACT]
    if len(exact):
        best = exact.loc[exact["complexity"].idxmin()]
        print(f"\ndraws the target: {best.equation}")
    else:
        print("\nno program drew the target")


if __name__ == "__main__":
    main()
