import unittest

import numpy as np
import symbolic_regression_rs


class TestSearch(unittest.TestCase):
    def _search(self, X, y, *, options=None, operators=None, variable_names=None):
        search_options = {
            "seed": 0,
            "niterations": 1,
            "populations": 1,
            "population_size": 16,
            "ncycles_per_iteration": 20,
            "maxsize": 10,
            "maxdepth": 10,
            "deterministic": True,
            "progress": False,
        }
        if options is not None:
            search_options.update(options)
        return symbolic_regression_rs.search(
            X,
            y,
            options=search_options,
            operators=operators or {2: ["+", "sub", "*"]},
            variable_names=variable_names or ["x0"],
        )

    def test_search_float32_returns_hall_of_fame(self):
        X = np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]

        out = self._search(X, y)

        self.assertEqual(out["backend_version"], symbolic_regression_rs.__version__)
        self.assertTrue(out["hall_of_fame"])
        best_loss = min(row["loss"] for row in out["hall_of_fame"])
        self.assertLessEqual(best_loss, 1e-6)
        for row in out["hall_of_fame"]:
            self.assertIsInstance(row["complexity"], int)
            self.assertIsInstance(row["loss"], float)
            self.assertIsInstance(row["equation"], str)

    def test_search_float64_returns_hall_of_fame(self):
        X = np.linspace(-1, 1, 24, dtype=np.float64).reshape(-1, 1)
        y = X[:, 0]

        out = self._search(X, y)

        self.assertTrue(out["hall_of_fame"])

    def test_search_rejects_mismatched_rows(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(3, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "X has 4 rows"):
            self._search(X, y)

    def test_search_rejects_bad_variable_names_length(self):
        X = np.ones((4, 2), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "expected 2 variable names"):
            self._search(X, y, variable_names=["x0"])

    def test_search_rejects_unknown_operator(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "unsupported builtin operator"):
            self._search(X, y, operators={1: ["not_an_operator"]})

    def test_search_rejects_operator_under_wrong_arity(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "has arity 2"):
            self._search(X, y, operators={1: ["+"]})

    def test_search_accepts_pysr_division_token(self):
        X = np.linspace(1, 2, 16, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]

        out = self._search(X, y, operators={2: ["/"]})

        self.assertTrue(out["hall_of_fame"])

    def test_search_accepts_serial_parallelism(self):
        X = np.linspace(-1, 1, 24, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]

        out = self._search(X, y, options={"parallelism": "serial"})

        self.assertTrue(out["hall_of_fame"])

    def test_search_rejects_unknown_parallelism(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "parallelism"):
            self._search(X, y, options={"parallelism": "multiprocessing"})

    def test_search_rejects_unknown_option(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "unsupported or unknown options"):
            self._search(X, y, options={"not_a_real_option": 1})

    def test_search_rejects_unknown_mutation_weight(self):
        X = np.ones((4, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError, "unsupported or unknown mutation_weights"
        ):
            self._search(X, y, options={"mutation_weights": {"not_a_real_weight": 1.0}})

    def test_search_accepts_nondefault_seed(self):
        X = np.linspace(-1, 1, 24, dtype=np.float32).reshape(-1, 1)
        y = X[:, 0]

        out = self._search(X, y, options={"seed": 123})

        self.assertTrue(out["hall_of_fame"])


if __name__ == "__main__":
    unittest.main()
