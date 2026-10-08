import os
import sys
import warnings
from types import ModuleType
from typing import cast

from .julia_registry_helpers import try_with_registry_fallback

autoload_extensions = os.environ.get("PYSR_AUTOLOAD_EXTENSIONS")
if autoload_extensions is not None:
    warnings.warn(
        "PYSR_AUTOLOAD_EXTENSIONS is deprecated and will be removed in a future release. "
        "Set PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION instead.",
        FutureWarning,
    )
    # Only fill the gap; an explicitly set juliacall variable takes precedence.
    os.environ.setdefault(
        "PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", autoload_extensions
    )


def _requested_julia_threads() -> str | None:
    """The Julia thread count the user asked for, if any.

    This follows juliacall's own precedence: `-X juliacall-threads`, then
    `PYTHON_JULIACALL_THREADS`, then `JULIA_NUM_THREADS`.
    """
    threads = sys._xoptions.get("juliacall-threads")
    if isinstance(threads, str):
        return threads
    for key in ("PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS"):
        if key in os.environ:
            return os.environ[key]
    return None


def _default_gc_threads(threads: str) -> str | None:
    threads = threads.split(",")[0].strip()
    if threads == "auto":
        # Match Julia's --threads=auto, which respects the CPU affinity mask.
        if hasattr(os, "sched_getaffinity"):
            threads = str(len(os.sched_getaffinity(0)))
        else:
            threads = str(os.cpu_count() or 2)
    try:
        return f"{max(1, int(threads) // 2)},1"
    except ValueError:
        # Leave anything we don't understand to Julia:
        return None


pysr_set_gc_threads = False

# Check if JuliaCall is already loaded, and if so, warn the user
# about the relevant environment variables. If not loaded,
# set up sensible defaults.
if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. "
        "Make sure that you have set the environment variable `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults. "
        "Also note that PySR will not be able to configure `PYTHON_JULIACALL_THREADS` for you."
    )
else:
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so PySR was not able to set it. You may wish to set it to `'auto'` for full use "
            "of your CPU."
        )

    # Use every CPU by default, but respect a thread count the user already
    # requested, including through Julia's own `JULIA_NUM_THREADS`:
    requested_threads = _requested_julia_threads()
    if requested_threads is None:
        requested_threads = os.environ["PYTHON_JULIACALL_THREADS"] = "auto"

    if "JULIA_NUM_GC_THREADS" not in os.environ:
        # Concurrent GC sweeping (`N,1`): the search is GC-bound at high
        # thread counts, and this was neutral at 8 threads, 8% faster at 32,
        # and 25% faster at 96.
        gc_threads = _default_gc_threads(requested_threads)
        if gc_threads is not None:
            os.environ["JULIA_NUM_GC_THREADS"] = gc_threads
            pysr_set_gc_threads = True

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        # Don't hijack `%%julia` magics in notebooks unless asked;
        # opt back in with PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=yes
        # or `%load_ext juliacall`.
        ("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no"),
    ):
        os.environ[k] = os.environ.get(k, default)


def _import_juliacall():
    import juliacall  # type: ignore


try_with_registry_fallback(_import_juliacall)

if pysr_set_gc_threads:
    # Julia has read this setting. Let child workers use Julia's own default.
    del os.environ["JULIA_NUM_GC_THREADS"]


from juliacall import AnyValue  # type: ignore
from juliacall import VectorValue  # type: ignore
from juliacall import Main as jl  # type: ignore

jl = cast(ModuleType, jl)


jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using SymbolicRegression")
SymbolicRegression = jl.SymbolicRegression

# Expose `D` operator:
jl.seval("using SymbolicRegression: D")

# Expose other operators:
jl.seval("using SymbolicRegression: less, greater_equal, less_equal")

jl.seval("using Pkg: Pkg")
Pkg = jl.Pkg
