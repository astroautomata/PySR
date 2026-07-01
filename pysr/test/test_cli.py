import unittest
from textwrap import dedent

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

                  Choose from main, jax, torch, autodiff, cli, dev, startup, and slurm. You can
                  give multiple tests, separated by commas.

                Options:
                  -k TEXT  Filter expressions to select specific tests.
                  --help   Show this message and exit.
                """)
            result = self.cli_runner.invoke(pysr, ["test", "--help"])
            self.assertEqual(result.output.strip(), expected.strip())
            self.assertEqual(result.exit_code, 0)

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
