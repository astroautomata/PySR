import fnmatch
import sys
import unittest
import warnings

import click

from ..test import (
    get_runtests_cli,
    runtests,
    runtests_autodiff,
    runtests_dev,
    runtests_jax,
    runtests_slurm,
    runtests_startup,
    runtests_torch,
)


@click.group("pysr")
@click.pass_context
def pysr(context):
    ctx = context


@pysr.command("install", help="DEPRECATED (dependencies are now installed at import).")
@click.option(
    "-p",
    "julia_project",
    "--project",
    default=None,
    type=str,
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Disable logging.")
@click.option(
    "--precompile",
    "precompile",
    flag_value=True,
    default=None,
)
@click.option(
    "--no-precompile",
    "precompile",
    flag_value=False,
    default=None,
)
def _install(julia_project, quiet, precompile):
    warnings.warn(
        "This command is deprecated. Julia dependencies are now installed at first import."
    )


TEST_OPTIONS = {"main", "jax", "torch", "autodiff", "cli", "dev", "startup", "slurm"}


@pysr.command("test")
@click.argument("tests", nargs=1)
@click.option(
    "-k",
    "expressions",
    multiple=True,
    type=str,
    help="Filter expressions to select specific tests.",
)
def _tests(tests, expressions):
    """Run parts of the PySR test suite.

    Choose from main, jax, torch, autodiff, cli, dev, startup, and slurm.
    You can give multiple tests, separated by commas.
    """
    test_cases = []
    for test in tests.split(","):
        if test == "main":
            test_cases.extend(runtests(just_tests=True))
        elif test == "jax":
            test_cases.extend(runtests_jax(just_tests=True))
        elif test == "torch":
            test_cases.extend(runtests_torch(just_tests=True))
        elif test == "autodiff":
            test_cases.extend(runtests_autodiff(just_tests=True))
        elif test == "cli":
            runtests_cli = get_runtests_cli()
            test_cases.extend(runtests_cli(just_tests=True))
        elif test == "dev":
            test_cases.extend(runtests_dev(just_tests=True))
        elif test == "startup":
            test_cases.extend(runtests_startup(just_tests=True))
        elif test == "slurm":
            test_cases.extend(runtests_slurm(just_tests=True))
        else:
            warnings.warn(f"Invalid test {test}. Skipping.")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in test_cases:
        loaded_tests = loader.loadTestsFromTestCase(test_case)
        for test in loaded_tests:
            if len(expressions) == 0 or any(
                fnmatch.fnmatch(test.id(), "*" + expression + "*")
                for expression in expressions
            ):
                suite.addTest(test)

    runner = unittest.TextTestRunner()
    results = runner.run(suite)

    if not results.wasSuccessful():
        sys.exit(1)
