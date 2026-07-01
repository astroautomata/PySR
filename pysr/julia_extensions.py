"""This file installs and loads extensions for SymbolicRegression."""

from __future__ import annotations

from typing import Literal

from .julia_import import Pkg, jl
from .julia_registry_helpers import try_with_registry_fallback
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec


def load_required_packages(
    *,
    turbo: bool = False,
    bumper: bool = False,
    autodiff_backend: Literal["Zygote", "Mooncake", "Enzyme"] | None = None,
    cluster_manager: str | None = None,
    logger_spec: AbstractLoggerSpec | None = None,
):
    if turbo:
        load_package("LoopVectorization", "bdcacae8-1622-11e9-2a5c-532679323890")
    if bumper:
        load_package("Bumper", "8ce10254-0962-460f-a3d8-1f77fea1446e")
    if autodiff_backend == "Zygote":
        load_package("Zygote", "e88e6eb3-aa80-5325-afca-941959d7151f")
    elif autodiff_backend == "Mooncake":
        load_package("Mooncake", "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6")
    elif autodiff_backend == "Enzyme":
        load_package("Enzyme", "7da242da-08ed-463a-9acd-ee780be4f1d9")
    if cluster_manager is not None:
        load_package("ClusterManagers", "34f1f09b-3a8b-5176-ab39-66d58a4d544e")
    if isinstance(logger_spec, TensorBoardLoggerSpec):
        load_package("TensorBoardLogger", "899adc3e-224a-11e9-021f-63837185c80f")


def load_all_packages():
    """Install and load all Julia extensions available to PySR."""
    load_required_packages(
        turbo=True,
        bumper=True,
        autodiff_backend="Zygote",
        cluster_manager="slurm",
        logger_spec=TensorBoardLoggerSpec(log_dir="logs"),
    )


# TODO: Refactor this file so we can install all packages at once using `juliapkg`,
#       ideally parameterizable via the regular Python extras API


def isinstalled(uuid_s: str):
    return jl.haskey(Pkg.dependencies(), jl.Base.UUID(uuid_s))


def load_package(package_name: str, uuid_s: str) -> None:
    if not isinstalled(uuid_s):

        def _add_package():
            Pkg.add(name=package_name, uuid=uuid_s)
            Pkg.resolve()

        try_with_registry_fallback(_add_package)

    # TODO: Protect against loading the same symbol from two packages,
    #       maybe with a @gensym here.
    jl.seval(f"using {package_name}: {package_name}")
    return None
