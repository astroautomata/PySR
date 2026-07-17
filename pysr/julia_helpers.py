"""Functions for initializing the Julia environment and installing deps."""

from typing import Any, Callable, cast, overload

import numpy as np
from juliacall import convert as jl_convert  # type: ignore
from numpy.typing import NDArray

from .deprecated import init_julia, install
from .julia_import import AnyValue, jl

jl_convert = cast(Callable[[Any, Any], Any], jl_convert)

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

Serialization = jl.Serialization
PythonCall = jl.PythonCall

jl.seval("using SymbolicRegression: plus, sub, mult, div, pow")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    if filename is None:
        return None
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


KNOWN_CLUSTERMANAGER_BACKENDS = ["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"]


def load_cluster_manager(cluster_manager: str) -> AnyValue:
    if cluster_manager == "slurm":
        jl.seval("using SlurmClusterManager: SlurmManager")
        jl.seval("using Distributed")
        jl.seval("""
            function addprocs_slurm(numprocs::Integer; exeflags=``, lazy=false, kws...)
                manager = SlurmManager()
                if manager.ntasks != numprocs
                    error(
                        "Requested $numprocs processes, but Slurm allocation has $(manager.ntasks) tasks. " *
                        "Set Slurm `--ntasks`/`--ntasks-per-node` to match, and set `procs` accordingly."
                    )
                end
                procs = Distributed.addprocs(manager; exeflags=exeflags, lazy=lazy, kws...)
                # SymbolicRegression may serialize the addprocs function to workers. Defining a
                # stub on the new workers avoids deserialization failures if it gets captured.
                Distributed.@everywhere procs begin
                    function addprocs_slurm(
                        numprocs::Integer;
                        exeflags=``,
                        lazy=false,
                        kws...,
                    )
                        error("addprocs_slurm should only be called on the master process.")
                    end
                end
                return procs
            end
            """)
        return jl.addprocs_slurm
    elif cluster_manager in KNOWN_CLUSTERMANAGER_BACKENDS:
        jl.seval(f"using ClusterManagers: addprocs_{cluster_manager}")
        return jl.seval(f"addprocs_{cluster_manager}")
    else:
        # Assume it's a function
        return jl.seval(cluster_manager)


def jl_array(x, dtype=None):
    if x is None:
        return None
    elif dtype is None:
        return jl_convert(jl.Array, x)
    else:
        return jl_convert(jl.Array[dtype], x)


def jl_dict(x):
    return jl_convert(jl.Dict, x)


def jl_named_tuple(d):
    return jl.NamedTuple({jl.Symbol(k): v for k, v in d.items()})


def jl_is_function(f) -> bool:
    return cast(bool, jl.seval("op -> op isa Function")(f))


def jl_serialize(obj: Any) -> NDArray[np.uint8]:
    buf = jl.IOBuffer()
    Serialization.serialize(buf, obj)
    return np.array(jl.take_b(buf))


@overload
def jl_deserialize(s: NDArray[np.uint8]) -> AnyValue: ...
@overload
def jl_deserialize(s: None) -> None: ...
def jl_deserialize(s):
    if s is None:
        return s
    buf = jl.IOBuffer()
    jl.write(buf, jl_array(s))
    jl.seekstart(buf)
    return Serialization.deserialize(buf)
