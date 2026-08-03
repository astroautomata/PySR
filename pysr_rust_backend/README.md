# pysr-rust-backend

Experimental Rust backend package for PySR.

The distribution name is `pysr-rust-backend`; it installs the
`symbolic_regression_rs` import module consumed by PySR's optional Rust backend.
It deliberately exposes a small wire contract first: Python arrays and plain
dictionaries in, plain dictionaries out.

> This package is experimental and should be treated as a PySR backend boundary,
> not a stable standalone Python API, while the Rust engine API is still
> stabilizing.

This draft places the Python/Rust connector package inside the PySR repository
for review. The Rust search engine crates are still consumed from
`astroautomata/symbolic_regression.rs`.

## Install For Development

From the workspace root:

```bash
uv pip install --python /path/to/python -e pysr_rust_backend
```

Or, from this directory:

```bash
python -m pip install -e .
```

## Python API

```python
import numpy as np
import symbolic_regression_rs

X = np.linspace(-1, 1, 64, dtype=np.float32).reshape(-1, 1)
y = X[:, 0]

out = symbolic_regression_rs.search(
    X,
    y,
    options={
        "seed": 0,
        "niterations": 1,
        "populations": 1,
        "population_size": 16,
        "ncycles_per_iteration": 20,
        "deterministic": True,
        "progress": False,
    },
    operators={2: ["+", "sub", "*"]},
    variable_names=["x0"],
)

print(out["hall_of_fame"])
```

The return value is a dictionary:

```python
{
    "backend_version": "0.1.0",
    "hall_of_fame": [
        {"complexity": 1, "loss": 0.0, "equation": "x0"},
    ],
}
```

## Current Limits

- Dense NumPy arrays only.
- `float32` and `float64` only.
- Single-output regression only.
- Builtin Rust operators only, with operator arities validated from the
  `operators={arity: [names]}` mapping.
- Unknown or unsupported option keys raise `ValueError`.
- No sample weights, non-MSE losses, per-variable/per-operator complexities,
  constraints, nested constraints, Python callbacks, custom losses, custom
  operators, GPU, warm start, or richer PySR output yet.

## Test

Install the package first, then run:

```bash
python -m unittest discover pysr_rust_backend/tests
cargo check --manifest-path pysr_rust_backend/Cargo.toml
cargo test --manifest-path pysr_rust_backend/Cargo.toml
```

## Publish

The repository includes a manual `Publish Python wrapper` GitHub Actions
workflow. It builds Linux, Windows, and macOS wheels for Python 3.9 through
3.13 plus a source distribution from a selected ref, uploads the artifacts, and
publishes to TestPyPI or PyPI through trusted publishing.
