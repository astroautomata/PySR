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


def _default_gc_threads() -> str:
    threads = os.environ.get("PYTHON_JULIACALL_THREADS", "auto").split(",")[0]
    if threads == "auto":
        # Match Julia's --threads=auto, which respects the CPU affinity mask.
        if hasattr(os, "sched_getaffinity"):
            threads = str(len(os.sched_getaffinity(0)))
        else:
            threads = str(os.cpu_count() or 2)
    return f"{max(1, int(threads) // 2)},1"


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

    pysr_set_gc_threads = "JULIA_NUM_GC_THREADS" not in os.environ

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        # Don't hijack `%%julia` magics in notebooks unless asked;
        # opt back in with PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=yes
        # or `%load_ext juliacall`.
        ("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no"),
        # Concurrent GC sweeping (`N,1`): the search is GC-bound at high
        # thread counts, and this was neutral at 8 threads, 8% faster at 32,
        # and 25% faster at 96.
        ("JULIA_NUM_GC_THREADS", _default_gc_threads()),
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
