import os
import unittest
from textwrap import dedent
from unittest.mock import patch

from click import testing as click_testing


def get_runtests():
    # Lazy load to avoid circular imports.

    from .._cli.main import pysr

    class TestCli(unittest.TestCase):
        # TODO: Include test for custom project here.
        def setUp(self):
            self.cli_runner = click_testing.CliRunner()

        def test_help_on_all_commands(self):
            expected = dedent("""
                    Usage: pysr [OPTIONS] COMMAND [ARGS]...

                    Options:
                      --help  Show this message and exit.

                    Commands:
                      install  DEPRECATED (dependencies are now installed at import).
                      test     Run parts of the PySR test suite.
                """)
            result = self.cli_runner.invoke(pysr, ["--help"])
            self.assertEqual(result.output.strip(), expected.strip())
            self.assertEqual(result.exit_code, 0)

        def test_help_on_install(self):
            expected = dedent("""
                Usage: pysr install [OPTIONS]

                  DEPRECATED (dependencies are now installed at import).

                Options:
                  -p, --project TEXT
                  -q, --quiet         Disable logging.
                  --precompile
                  --no-precompile
                  --help              Show this message and exit.
                """)
            result = self.cli_runner.invoke(pysr, ["install", "--help"])
            self.assertEqual(result.output.strip(), expected.strip())
            self.assertEqual(result.exit_code, 0)

        def test_help_on_test(self):
            expected = dedent("""
                Usage: pysr test [OPTIONS] TESTS

                  Run parts of the PySR test suite.

                  Choose from main, jax, torch, autodiff, cli, dev, startup, slurm, and
                  interrupt. You can give multiple tests, separated by commas.

                Options:
                  -k TEXT  Filter expressions to select specific tests.
                  --help   Show this message and exit.
                """)
            result = self.cli_runner.invoke(pysr, ["test", "--help"])
            self.assertEqual(result.output.strip(), expected.strip())
            self.assertEqual(result.exit_code, 0)

        def test_test_shards_partition_filtered_duplicate_groups(self):
            executed = []

            class ShardTests(unittest.TestCase):
                def test_ignored(self):
                    executed.append(self.id())

                test_selected_a = test_ignored
                test_selected_b = test_ignored
                test_selected_c = test_ignored
                test_selected_d = test_ignored
                test_selected_e = test_ignored

            loader = unittest.TestLoader()
            selected = [
                test.id()
                for _ in range(2)
                for test in loader.loadTestsFromTestCase(ShardTests)
                if "selected" in test.id()
            ]
            shard_runs = []
            with patch(
                "pysr._cli.main.get_runtests_cli",
                return_value=lambda **_: [ShardTests],
            ):
                for shard_index in range(3):
                    executed.clear()
                    with patch.dict(
                        os.environ,
                        {
                            "PYSR_TEST_SHARD_COUNT": "3",
                            "PYSR_TEST_SHARD_INDEX": str(shard_index),
                        },
                    ):
                        result = self.cli_runner.invoke(
                            pysr, ["test", "cli,cli", "-k", "selected"]
                        )
                    self.assertEqual(result.exit_code, 0, result.output)
                    shard_runs.append(executed.copy())

            self.assertEqual(
                shard_runs,
                [selected[shard_index::3] for shard_index in range(3)],
            )

    def runtests(just_tests=False):
        """Run all tests in cliTest.py."""
        tests = [TestCli]
        if just_tests:
            return tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test in tests:
            suite.addTests(loader.loadTestsFromTestCase(test))
        runner = unittest.TextTestRunner()
        return runner.run(suite)

    return runtests
