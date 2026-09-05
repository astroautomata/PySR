"""Invent a 32-bit pseudorandom generator with no target array.

`y` is a column of zeros and the objective never reads it. The search sees 32 seed
states, and the loss runs each candidate as an iterated 32-bit map and scores the
generator that map is: how long its orbit is, how balanced its bit columns are, how
uncorrelated successive states are, how far one flipped input bit spreads, and whether
any bit row of the output is a shifted copy of a bit row of the input. Every term is 0
exactly when its desideratum is met.

The period term is the interesting one, because the whole state space is 4294967295
nonzero words and nothing may walk it. The objective probes the candidate on 0, on the
32 basis states and on 32 fixed states; if it is GF(2)-linear on that evidence, it
builds the minimal polynomial of the first seed's Krylov sequence, and if that
polynomial is irreducible the orbit's length is exactly the multiplicative order of x
modulo it, obtained by dividing the prime factors out of 2^deg - 1. A map whose period
cannot be certified is credited only with the orbit actually walked, so a candidate is
paid for the period it can prove.

`check` re-derives the claim in Python from the recovered expression, by a different
route than the loss uses. It reads the map's GF(2) matrix out of `predict`, confirms
the map really is linear on random states, and computes the multiplicative order of
that matrix by repeated squaring. An order of 4294967295 forces the period: the order
of a matrix over GF(2) divides the unit-group order of the algebra it generates, which
is a power of two times a product of 2^d - 1 over the degrees of the minimal
polynomial's irreducible factors, and 65537 divides 2^d - 1 only when 32 divides d. So
the minimal polynomial is irreducible of degree 32, the state space is one copy of
GF(2^32) under the map, and every nonzero word lies on one orbit of length 4294967295.
"""

import numpy as np

from pysr import PySRRegressor, TypeSpec

WIDTH = 32
MASK = (1 << WIDTH) - 1
FULL_PERIOD = MASK
ORDER_PRIMES = (3, 5, 17, 257, 65537)

SEEDS = [int(v) for v in np.random.default_rng(0).integers(1, 1 << WIDTH, size=32)]
X = np.empty((len(SEEDS), 1), dtype=object)
for i, seed in enumerate(SEEDS):
    X[i, 0] = seed
y = np.zeros(len(SEEDS), dtype=object)
VARIABLE_NAMES = ["x"]

SPEC = TypeSpec(
    "Word",
    fields={"bits": "UInt32"},
    sample="rng -> Word(rand(rng, Bool) ? rand(rng, UInt32(0):UInt32(31)) : rand(rng, UInt32))",
    mutate="(rng, value, temperature) -> Word(rand(rng, Bool) ? xor(value.bits, UInt32(1) << rand(rng, 0:31)) : value.bits + rand(rng, (UInt32(1), typemax(UInt32))))",
    string='value -> value.bits < UInt32(32) ? string(value.bits) : "0x" * string(value.bits, base = 16, pad = 8)',
    loss_type="Float64",
)

# The certificate: linear algebra over GF(2), so a period of four billion states costs
# a few dozen 32-bit operations instead of four billion evaluations.
PERIOD_CERTIFICATE = r"""
const PRNG_PROBES = UInt32[UInt32(0x9e3779b9) * UInt32(k) for k in 1:32]

function _prng_matvec(cols::Vector{UInt32}, v::UInt32)
    acc = UInt32(0)
    vv = v
    while vv != 0
        acc ⊻= cols[trailing_zeros(vv) + 1]
        vv &= vv - UInt32(1)
    end
    return acc
end

function _prng_mulmod(a::UInt64, b::UInt64, f::UInt64, deg::Int)
    out = UInt64(0)
    aa, bb = a, b
    while bb != 0
        if bb & 1 == 1
            out ⊻= aa
        end
        bb >>= 1
        aa <<= 1
        if (aa >> deg) & 1 == 1
            aa ⊻= f
        end
    end
    return out
end

function _prng_powmod(base::UInt64, e::UInt64, f::UInt64, deg::Int)
    r = UInt64(1)
    b, ee = base, e
    while ee != 0
        if ee & 1 == 1
            r = _prng_mulmod(r, b, f, deg)
        end
        b = _prng_mulmod(b, b, f, deg)
        ee >>= 1
    end
    return r
end

_prng_deg(p::UInt64) = 63 - leading_zeros(p)

function _prng_gcd(a::UInt64, b::UInt64)
    x, y = a, b
    while y != 0
        dy = _prng_deg(y)
        while x != 0 && _prng_deg(x) >= dy
            x ⊻= y << (_prng_deg(x) - dy)
        end
        x, y = y, x
    end
    return x
end

function _prng_minpoly(cols::Vector{UInt32}, v0::UInt32)
    leads = Int[]
    vecs = UInt32[]
    combs = UInt64[]
    cur = v0
    comb = UInt64(1)
    for _ in 0:32
        red, rc = cur, comb
        for i in eachindex(leads)
            if (red >> leads[i]) & UInt32(1) == 1
                red ⊻= vecs[i]
                rc ⊻= combs[i]
            end
        end
        if red == 0
            return rc
        end
        push!(leads, 31 - leading_zeros(red))
        push!(vecs, red)
        push!(combs, rc)
        cur = _prng_matvec(cols, cur)
        comb <<= 1
    end
    return UInt64(0)
end

function _prng_irreducible(f::UInt64, deg::Int)
    deg < 1 && return false
    _prng_powmod(UInt64(2), UInt64(1) << deg, f, deg) == UInt64(2) || return false
    d = deg
    p = 2
    while p * p <= d
        if d % p == 0
            g = _prng_gcd(
                _prng_powmod(UInt64(2), UInt64(1) << (deg ÷ p), f, deg) ⊻ UInt64(2), f)
            g == UInt64(1) || return false
            while d % p == 0
                d ÷= p
            end
        end
        p += 1
    end
    if d > 1
        g = _prng_gcd(
            _prng_powmod(UInt64(2), UInt64(1) << (deg ÷ d), f, deg) ⊻ UInt64(2), f)
        g == UInt64(1) || return false
    end
    return true
end

function _prng_order(f::UInt64, deg::Int)
    order = (UInt64(1) << deg) - UInt64(1)
    rest = order
    p = UInt64(2)
    while p * p <= rest
        if rest % p == 0
            while order % p == 0 && _prng_powmod(UInt64(2), order ÷ p, f, deg) == UInt64(1)
                order ÷= p
            end
            while rest % p == 0
                rest ÷= p
            end
        end
        p += UInt64(1)
    end
    if rest > 1
        while order % rest == 0 &&
              _prng_powmod(UInt64(2), order ÷ rest, f, deg) == UInt64(1)
            order ÷= rest
        end
    end
    return order
end

function _prng_certified(tree, options, seed::UInt32)
    m = 33 + length(PRNG_PROBES)
    grid = Matrix{Word}(undef, 1, m)
    grid[1, 1] = Word(UInt32(0))
    for b in 0:31
        grid[1, 2 + b] = Word(UInt32(1) << b)
    end
    for (i, p) in enumerate(PRNG_PROBES)
        grid[1, 33 + i] = Word(p)
    end
    out, ok = eval_tree_array(tree, grid, options)
    ok || return UInt64(0)
    out[1].bits == 0 || return UInt64(0)

    cols = [out[2 + b].bits for b in 0:31]
    for (i, p) in enumerate(PRNG_PROBES)
        _prng_matvec(cols, p) == out[33 + i].bits || return UInt64(0)
    end

    f = _prng_minpoly(cols, seed)
    f == 0 && return UInt64(0)
    deg = _prng_deg(f)
    _prng_irreducible(f, deg) || return UInt64(0)
    return _prng_order(f, deg)
end
"""

PRNG_LOSS = PERIOD_CERTIFICATE + r"""
function prng_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    steps = 256
    lags = (1, 2, 3, 5)
    n = dataset.n
    walk = Matrix{UInt32}(undef, steps, n)
    x = copy(dataset.X)
    for s in 1:steps
        out, ok = eval_tree_array(tree, x, options)
        ok || return L(Inf)
        length(out) == n || error("prng_loss scored " * string(length(out)) *
                                  " of " * string(n) * " samples; set batching=false")
        for j in 1:n
            walk[s, j] = out[j].bits
        end
        x = reshape(out, 1, n)
    end

    seen = typemax(Int)
    for j in 1:n
        seen = min(seen, length(Set(@view walk[:, j])))
    end
    certified = _prng_certified(tree, options, dataset.X[1, 1].bits)
    proven = certified > 0 ? Float64(certified) : Float64(seen)
    period = 1 - log2(proven) / 32

    ones = zeros(Int, 32)
    for j in 1:n, s in 1:steps
        w = walk[s, j]
        for b in 1:32
            ones[b] += (w >> (b - 1)) & UInt32(1)
        end
    end
    total = steps * n
    balance = 0.0
    for b in 1:32
        p = ones[b] / total
        balance += (2p - 1)^2
    end
    balance /= 32

    autocorr = 0.0
    joint = zeros(Int, 32)
    head = zeros(Int, 32)
    tail = zeros(Int, 32)
    for lag in lags
        fill!(joint, 0)
        fill!(head, 0)
        fill!(tail, 0)
        for j in 1:n, s in 1:(steps - lag)
            u = walk[s, j]
            v = walk[s + lag, j]
            for b in 1:32
                a = (u >> (b - 1)) & UInt32(1)
                c = (v >> (b - 1)) & UInt32(1)
                head[b] += a
                tail[b] += c
                joint[b] += a & c
            end
        end
        pairs = (steps - lag) * n
        for b in 1:32
            ph = head[b] / pairs
            pt = tail[b] / pairs
            cov = joint[b] / pairs - ph * pt
            var = ph * (1 - ph) * pt * (1 - pt)
            autocorr += var > 0 ? cov^2 / var : 0.0
        end
    end
    autocorr /= 32 * length(lags)

    m = length(PRNG_PROBES)
    grid = Matrix{Word}(undef, 1, m)
    for i in 1:m
        grid[1, i] = Word(PRNG_PROBES[i])
    end
    base, bok = eval_tree_array(tree, grid, options)
    bok || return L(Inf)
    moved = 0
    deps = zeros(Int, 32)
    for b in 0:31
        for i in 1:m
            grid[1, i] = Word(PRNG_PROBES[i] ⊻ (UInt32(1) << b))
        end
        flipped, fok = eval_tree_array(tree, grid, options)
        fok || return L(Inf)
        touched = UInt32(0)
        for i in 1:m
            d = base[i].bits ⊻ flipped[i].bits
            moved += count_ones(d)
            touched |= d
        end
        for j in 1:32
            deps[j] += (touched >> (j - 1)) & UInt32(1)
        end
    end
    diffusion = max(0.0, (4.0 - moved / (32 * m)) / 4.0)^2
    edge = 0.0
    for j in 1:32
        edge += max(0.0, (2.0 - deps[j]) / 2.0)^2
    end
    edge /= 32

    shear = 0.0
    npairs = 0
    sjoint = zeros(Int, 32)
    shead = zeros(Int, 32)
    stail = zeros(Int, 32)
    for lag in (1, 2), d in (-3, -2, -1, 1, 2, 3)
        blo = d >= 0 ? 0 : -d
        bhi = d >= 0 ? 31 - d : 31
        fill!(sjoint, 0)
        fill!(shead, 0)
        fill!(stail, 0)
        for j in 1:n, s in 1:(steps - lag)
            u = walk[s, j]
            v = walk[s + lag, j]
            for b in blo:bhi
                a = (u >> b) & UInt32(1)
                c = (v >> (b + d)) & UInt32(1)
                shead[b + 1] += a
                stail[b + 1] += c
                sjoint[b + 1] += a & c
            end
        end
        spairs = (steps - lag) * n
        for b in blo:bhi
            ph = shead[b + 1] / spairs
            pt = stail[b + 1] / spairs
            cov = sjoint[b + 1] / spairs - ph * pt
            var = ph * (1 - ph) * pt * (1 - pt)
            shear += var > 0 ? cov^2 / var : 0.0
            npairs += 1
        end
    end
    shear /= npairs

    return L(period + balance + autocorr + diffusion + shear + edge)
end
"""

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
    loss_function=PRNG_LOSS,
    # The objective walks every seed and indexes the sample axis, so it has to see the
    # whole sample set; a minibatch would score a different problem.
    batching=False,
    maxsize=20,
    # Eight small populations, evolved deeply: each candidate costs a 256-step walk of
    # every seed, so the search is worth more generations rather than more members.
    populations=8,
    population_size=30,
    ncycles_per_iteration=30,
    niterations=800,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)

IDENTITY = [1 << b for b in range(WIDTH)]
LINEARITY_PROBES = [
    int(v) for v in np.random.default_rng(1).integers(0, 1 << WIDTH, size=256)
]


def matvec(cols, v):
    """Image of `v` under the GF(2) matrix whose columns are `cols`."""
    acc = 0
    while v:
        acc ^= cols[(v & -v).bit_length() - 1]
        v &= v - 1
    return acc


def matmul(a, b):
    return [matvec(a, col) for col in b]


def matpow(m, e):
    result = IDENTITY
    while e:
        if e & 1:
            result = matmul(result, m)
        m = matmul(m, m)
        e >>= 1
    return result


def linear_matrix(model, index):
    """The map's GF(2) matrix, or None if the recovered expression is not linear."""
    probe = np.empty((1 + WIDTH + len(LINEARITY_PROBES), 1), dtype=object)
    probe[0, 0] = 0
    for b in range(WIDTH):
        probe[1 + b, 0] = 1 << b
    for i, state in enumerate(LINEARITY_PROBES):
        probe[1 + WIDTH + i, 0] = state
    image = [int(v) & MASK for v in model.predict(probe, index=index)]
    if image[0] != 0:
        return None
    cols = image[1 : 1 + WIDTH]
    for state, got in zip(LINEARITY_PROBES, image[1 + WIDTH :]):
        if matvec(cols, state) != got:
            return None
    return cols


def period(model, index):
    """4294967295 when the map is a full-period generator, 0 when it is not.

    The map is linear, so its order under composition is settled by matrix powers:
    order 4294967295 forces the minimal polynomial to be irreducible of degree 32, and
    then every nonzero word sits on a single orbit of that length.
    """
    cols = linear_matrix(model, index)
    if cols is None or matpow(cols, FULL_PERIOD) != IDENTITY:
        return 0
    if any(matpow(cols, FULL_PERIOD // q) == IDENTITY for q in ORDER_PRIMES):
        return 0
    return FULL_PERIOD


def check(model):
    """Some row of the front runs through all 4294967295 nonzero words."""
    return any(period(model, i) == FULL_PERIOD for i in model.equations_.index)


def main():
    model = PySRRegressor(**MODEL_KWARGS, random_state=0)
    model.fit(X, y, variable_names=VARIABLE_NAMES)
    front = model.equations_[["complexity", "loss", "equation"]].copy()
    front["full_period"] = [
        period(model, i) == FULL_PERIOD for i in model.equations_.index
    ]
    print(front.to_string(index=False))
    full = front[front["full_period"]]
    if len(full):
        row = full.iloc[0]
        print(
            f"\nperiod {FULL_PERIOD} at complexity {int(row.complexity)}: {row.equation}"
        )
    print(f"full period reached: {bool(len(full))}")


if __name__ == "__main__":
    main()
