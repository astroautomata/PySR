"""Shared backend adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BackendSearchResult:
    """Normalized search result returned by non-Julia backends."""

    hall_of_fame: pd.DataFrame
    backend_version: str | None = None
