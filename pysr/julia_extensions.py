"""This file installs and loads extensions for SymbolicRegression."""

from __future__ import annotations

import os
from typing import Literal

import juliapkg  # type: ignore

from .julia_import import Pkg, jl
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec

_PACKAGE_UUIDS: dict[str, str] = {
    "LoopVectorization": "bdcacae8-1622-11e9-2a5c-532679323890",
    "Bumper": "8ce10254-0962-460f-a3d8-1f77fea1446e",
    "Zygote": "e88e6eb3-aa80-5325-afca-941959d7151f",
    "Mooncake": "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6",
    "Enzyme": "7da242da-08ed-463a-9acd-ee780be4f1d9",
    "SlurmClusterManager": "c82cd089-7bf7-41d7-976b-6b5d413cbe0a",
    "ClusterManagers": "34f1f09b-3a8b-5176-ab39-66d58a4d544e",
    "TensorBoardLogger": "899adc3e-224a-11e9-021f-63837185c80f",
    "JSON": "682c06a0-de6a-54ab-a142-c8b1cf79cde6",
}


def load_required_packages(
    *,
    turbo: bool = False,
    bumper: bool = False,
    autodiff_backend: Literal["Zygote", "Mooncake", "Enzyme"] | None = None,
    cluster_manager: str | None = None,
    logger_spec: AbstractLoggerSpec | None = None,
    use_tracing: bool = False,
):
    if turbo:
        load_package("LoopVectorization")
    if bumper:
        load_package("Bumper")
    if autodiff_backend == "Zygote":
        load_package("Zygote")
    elif autodiff_backend == "Mooncake":
        load_package("Mooncake")
    elif autodiff_backend == "Enzyme":
        load_package("Enzyme")
    if cluster_manager == "slurm":
        load_package("SlurmClusterManager")
    elif cluster_manager is not None:
        load_package("ClusterManagers")
    if isinstance(logger_spec, TensorBoardLoggerSpec):
        load_package("TensorBoardLogger")
    if use_tracing:
        load_package("JSON")


def load_all_packages():
    """Install and load all Julia extensions available to PySR."""
    load_required_packages(
        turbo=True,
        bumper=True,
        autodiff_backend="Zygote",
        cluster_manager="slurm",
        logger_spec=TensorBoardLoggerSpec(log_dir="logs"),
        use_tracing=True,
    )
    load_package("ClusterManagers")


def install_all_packages() -> None:
    """Install all Julia extensions without loading them.

    Installing can change versions of packages this session already loaded.
    Run this in its own Python process before the session that uses the packages.
    """
    install_packages(
        [
            "LoopVectorization",
            "Bumper",
            "Zygote",
            "SlurmClusterManager",
            "ClusterManagers",
            "TensorBoardLogger",
            "JSON",
        ]
    )


def isinstalled(uuid_s: str):
    dependencies = Pkg.dependencies()
    uuid = jl.Base.UUID(uuid_s)
    return jl.haskey(dependencies, uuid) and dependencies[uuid].is_direct_dep


def install_packages(package_names) -> None:
    missing = [name for name in package_names if not isinstalled(_PACKAGE_UUIDS[name])]
    if not missing:
        return

    # juliapkg rebuilds the Julia project from its dependency files whenever it
    # re-resolves, for example in a process whose `sys.path` finds a different set of
    # `juliapkg.json` files. Recording extensions there keeps them, and the versions
    # they force, in every resolve.
    for name in missing:
        juliapkg.add(name, _PACKAGE_UUIDS[name])

    # Julia 1.13.2's auto-precompile after `Pkg.add` fails with "Unknown package" when the add
    # changes the version of an indirect dependency (JuliaLang/Pkg.jl#4866, fixed in #4867).
    # juliapkg runs `Pkg.precompile()` afterwards instead.
    old_value = os.environ.get("JULIA_PKG_PRECOMPILE_AUTO")
    os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    try:
        juliapkg.resolve()
    finally:
        if old_value is None:
            del os.environ["JULIA_PKG_PRECOMPILE_AUTO"]
        else:
            os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = old_value


def _loaded_packages_with_changed_versions() -> list[str]:
    changed = jl.seval("""
        let dependencies = Pkg.dependencies(), changed = String[]
            for (id, mod) in Base.loaded_modules
                haskey(dependencies, id.uuid) || continue
                loaded_version = Base.pkgversion(mod)
                environment_version = dependencies[id.uuid].version
                if loaded_version !== nothing && environment_version !== nothing &&
                    loaded_version != environment_version
                    push!(changed, "$(id.name) $(loaded_version) -> $(environment_version)")
                end
            end
            sort!(changed)
        end
    """)
    return [str(entry) for entry in changed]


def load_package(package_name: str) -> None:
    if not isinstalled(_PACKAGE_UUIDS[package_name]):
        install_packages([package_name])
        changed = _loaded_packages_with_changed_versions()
        if changed:
            raise RuntimeError(
                f"Installing {package_name} changed Julia packages this Python session "
                f"already loaded ({', '.join(changed)}). The installation finished. "
                f"Restart Python to use {package_name}; continuing would run Julia "
                "processes on different package versions than this session."
            )

    # TODO: Protect against loading the same symbol from two packages,
    #       maybe with a @gensym here.
    jl.seval(f"using {package_name}: {package_name}")
    return None
