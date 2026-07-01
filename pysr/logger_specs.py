from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .julia_helpers import jl_array, jl_dict
from .julia_import import AnyValue, jl


class AbstractLoggerSpec(ABC):
    """Abstract base class for logger specifications."""

    @abstractmethod
    def create_logger(self) -> AnyValue:
        """Create a logger instance."""
        pass  # pragma: no cover

    @abstractmethod
    def write_hparams(self, logger: AnyValue, hparams: dict[str, Any]) -> None:
        """Write hyperparameters to the logger."""
        pass  # pragma: no cover

    @abstractmethod
    def close(self, logger: AnyValue) -> None:
        """Close the logger instance."""
        pass  # pragma: no cover


@dataclass
class TensorBoardLoggerSpec(AbstractLoggerSpec):
    """Specification for TensorBoard logger.

    Parameters
    ----------
    log_dir : str
        Directory where TensorBoard logs will be saved. If `overwrite` is `False`,
        new logs will be saved to `{log_dir}_1`, and so on. Default is `"logs/run"`.
    log_interval : int, optional
        Interval (in steps) at which logs are written. Default is 10.
    overwrite : bool, optional
        Whether to overwrite existing logs in the directory. Default is False.
    """

    log_dir: str = "logs/run"
    log_interval: int = 1
    overwrite: bool = False

    def create_logger(self) -> AnyValue:
        # We assume that TensorBoardLogger is already imported via `julia_extensions.py`
        make_logger = jl.seval("""
            function make_logger(log_dir::AbstractString, overwrite::Bool, log_interval::Int)
                base_logger = TensorBoardLogger.TBLogger(
                    log_dir,
                    (overwrite ? (TensorBoardLogger.tb_overwrite,) : ())...
                )
                return SRLogger(; logger=base_logger, log_interval)
            end
        """)
        log_dir = str(self.log_dir)
        return make_logger(log_dir, self.overwrite, self.log_interval)

    def write_hparams(self, logger: AnyValue, hparams: dict[str, Any]) -> None:
        base_logger = jl.SymbolicRegression.get_logger(logger)
        writer = jl.seval("TensorBoardLogger.write_hparams!")
        jl_clean_hparams = jl_dict(
            {
                k: (v if isinstance(v, (bool, int, float)) else str(v))
                for k, v in hparams.items()
            }
        )
        writer(
            base_logger,
            jl_clean_hparams,
            jl_array(
                [
                    "search/data/summaries/pareto_volume",
                    "search/data/summaries/min_loss",
                ],
            ),
        )

    def close(self, logger: AnyValue) -> None:
        base_logger = jl.SymbolicRegression.get_logger(logger)
        jl.close(base_logger)
