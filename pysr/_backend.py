"""Import-time backend selection and Julia availability."""

import os
from typing import Literal

DEFAULT_BACKEND: Literal["julia", "rust"] = (
    "rust" if os.environ.get("PYSR_BACKEND") == "rust" else "julia"
)
JULIA_DISABLED = DEFAULT_BACKEND == "rust"


def require_julia() -> None:
    if JULIA_DISABLED:
        raise RuntimeError(
            "Julia is disabled because PYSR_BACKEND=rust. "
            "To use Julia, unset PYSR_BACKEND before importing pysr in a new process."
        )
