def runtests(*args, **kwargs):
    from .test_main import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_jax(*args, **kwargs):
    from .test_jax import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_torch(*args, **kwargs):
    from .test_torch import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_autodiff(*args, **kwargs):
    from .test_autodiff import runtests as _runtests

    return _runtests(*args, **kwargs)


def get_runtests_cli(*args, **kwargs):
    from .test_cli import get_runtests as _get_runtests

    return _get_runtests(*args, **kwargs)


def runtests_startup(*args, **kwargs):
    from .test_startup import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_dev(*args, **kwargs):
    from .test_dev import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_slurm(*args, **kwargs):
    from .test_slurm import runtests as _runtests

    return _runtests(*args, **kwargs)


def runtests_rust(*args, **kwargs):
    from .test_rust_backend import runtests as _runtests

    return _runtests(*args, **kwargs)


__all__ = [
    "runtests",
    "runtests_jax",
    "runtests_torch",
    "runtests_autodiff",
    "get_runtests_cli",
    "runtests_startup",
    "runtests_dev",
    "runtests_slurm",
    "runtests_rust",
]
