"""Utilities for managing Julia registry preferences during package operations."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

PREFERENCE_KEY = "JULIA_PKG_SERVER_REGISTRY_PREFERENCE"


def try_with_registry_fallback(f: Callable[..., T], *args, **kwargs) -> T:
    """Execute function with modified Julia registry preference.

    First tries with existing registry preference. If that fails with a Julia registry error,
    temporarily modifies the registry preference to 'eager'. Restores original preference after
    execution.
    """
    try:
        return f(*args, **kwargs)
    except Exception as initial_error:
        # Check if this is a Julia registry error by looking at the error message
        if "JuliaError" not in str(
            type(initial_error)
        ) or "Unsatisfiable requirements detected" not in str(initial_error):
            raise initial_error

        old_value = os.environ.get(PREFERENCE_KEY, None)
        if old_value == "eager":
            raise initial_error

        warnings.warn(
            "Initial Julia registry operation failed. Attempting to use the `eager` registry flavor of the Julia "
            + f"General registry from the Julia Pkg server (via the `{PREFERENCE_KEY}` environment variable)."
        )
        os.environ[PREFERENCE_KEY] = "eager"
        try:
            return f(*args, **kwargs)
        finally:
            if old_value is not None:
                os.environ[PREFERENCE_KEY] = old_value
            else:
                del os.environ[PREFERENCE_KEY]
