import os
import subprocess
import unittest
from pathlib import Path


class TestDev(unittest.TestCase):
    def test_simple_change_to_backend(self):
        """Test that we can use a development version of SymbolicRegression.jl"""
        PYSR_TEST_JULIA_VERSION = os.environ.get("PYSR_TEST_JULIA_VERSION", "1.11")
        PYSR_TEST_PYTHON_VERSION = os.environ.get("PYSR_TEST_PYTHON_VERSION", "3.12")
        repo_root = Path(__file__).parent.parent.parent

        build_result = subprocess.run(
            [
                "docker",
                "buildx",
                "bake",
                "-f",
                "docker-bake.hcl",
                "pysr-dev",
                "--set",
                f"pysr-dev.args.JLVERSION={PYSR_TEST_JULIA_VERSION}",
                "--set",
                f"pysr-dev.args.PYVERSION={PYSR_TEST_PYTHON_VERSION}",
            ],
            env=os.environ,
            cwd=repo_root,
            universal_newlines=True,
        )
        self.assertEqual(build_result.returncode, 0)
        test_result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "pysr-dev",
                "python3",
                "-c",
                "from pysr import SymbolicRegression as SR; print(SR.__test_function())",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
            cwd=repo_root,
        )
        self.assertEqual(test_result.returncode, 0)
        self.assertEqual(test_result.stdout.decode("utf-8").strip(), "2.3")


def runtests(just_tests=False):
    tests = [TestDev]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
