"""Benchmark PySR Julia and Rust backends on the same synthetic dataset.

Examples
--------
Quick smoke benchmark:

    python benchmarks/compare_backends.py --niterations 1 --repeats 1

Measure each backend in a fresh Python process:

    python benchmarks/compare_backends.py --mode subprocess --repeats 3

Longer run with JSON output:

    python benchmarks/compare_backends.py --niterations 10 --repeats 5 --json-output results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

BACKENDS = ("julia", "rust")


def make_dataset(
    *,
    samples: int,
    features: int,
    seed: int,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    if features < 2:
        raise ValueError("features must be at least 2 for the default benchmark target")

    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(samples, features)).astype(np.float32)
    y = (
        1.5 * np.sin(X[:, 0])
        + 0.75 * X[:, 1] * X[:, 1]
        - 0.25 * X[:, 0] * X[:, 1]
        + 0.5
    ).astype(np.float32)
    if noise > 0:
        y = y + rng.normal(0.0, noise, size=samples).astype(np.float32)
    return X, y


def build_model(backend: str, args: argparse.Namespace, *, seed: int):
    from pysr import PySRRegressor

    return PySRRegressor(
        backend=backend,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp"],
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        ncycles_per_iteration=args.ncycles_per_iteration,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        deterministic=not args.non_deterministic,
        random_state=seed,
        parallelism=args.parallelism,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        model_selection="accuracy",
    )


def run_fit_once(
    backend: str,
    args: argparse.Namespace,
    *,
    repeat_index: int,
    include_import: bool = False,
) -> dict[str, Any]:
    X, y = make_dataset(
        samples=args.samples,
        features=args.features,
        seed=args.seed,
        noise=args.noise,
    )

    if include_import:
        start = time.perf_counter()
    else:
        from pysr import PySRRegressor  # noqa: F401

        start = time.perf_counter()

    model = build_model(backend, args, seed=args.seed + repeat_index)
    model.fit(X, y)
    seconds = time.perf_counter() - start

    prediction = model.predict(X)
    mse = float(np.mean((prediction - y) ** 2))
    best = model.get_best()

    return {
        "backend": backend,
        "mode": "fit",
        "repeat": repeat_index,
        "seconds": seconds,
        "best_loss": float(best["loss"]),
        "mse": mse,
        "equation": str(best["equation"]),
        "n_equations": int(len(model.equations_)),
        "backend_version": (
            getattr(model, "rust_backend_version_", None) if backend == "rust" else None
        ),
    }


def worker_command(
    args: argparse.Namespace, backend: str, output_path: Path, repeat: int
) -> list[str]:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "--_worker-backend",
        backend,
        "--_worker-output",
        str(output_path),
        "--_worker-repeat",
        str(repeat),
        "--samples",
        str(args.samples),
        "--features",
        str(args.features),
        "--seed",
        str(args.seed),
        "--noise",
        str(args.noise),
        "--niterations",
        str(args.niterations),
        "--populations",
        str(args.populations),
        "--population-size",
        str(args.population_size),
        "--ncycles-per-iteration",
        str(args.ncycles_per_iteration),
        "--maxsize",
        str(args.maxsize),
        "--maxdepth",
        str(args.maxdepth),
        "--parallelism",
        args.parallelism,
    ]
    if args.non_deterministic:
        command.append("--non-deterministic")
    return command


def run_subprocess_once(
    backend: str,
    args: argparse.Namespace,
    *,
    repeat_index: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "result.json"
        command = worker_command(args, backend, output_path, repeat_index)
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        seconds = time.perf_counter() - start

        if completed.returncode != 0:
            raise RuntimeError(
                f"{backend!r} subprocess benchmark failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["mode"] = "subprocess"
        result["worker_seconds"] = result["seconds"]
        result["seconds"] = seconds
        return result


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(row["backend"], row["mode"]) for row in results})
    for backend, mode in keys:
        group = [
            row for row in results if row["backend"] == backend and row["mode"] == mode
        ]
        seconds = [row["seconds"] for row in group]
        losses = [row["best_loss"] for row in group]
        mses = [row["mse"] for row in group]
        rows.append(
            {
                "backend": backend,
                "mode": mode,
                "repeats": len(group),
                "mean_seconds": statistics.fmean(seconds),
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "mean_best_loss": statistics.fmean(losses),
                "mean_mse": statistics.fmean(mses),
                "last_equation": group[-1]["equation"],
            }
        )
    return rows


def format_seconds(value: float) -> str:
    return f"{value:9.3f}"


def print_summary(results: list[dict[str, Any]]) -> None:
    rows = summarize(results)
    print()
    print("Backend timing summary")
    print(
        "backend     mode          repeats   mean_s   median_s      min_s   "
        "mean_loss    mean_mse   last_equation"
    )
    print("-" * 108)
    for row in rows:
        print(
            f"{row['backend']:<11} {row['mode']:<13} {row['repeats']:>7} "
            f"{format_seconds(row['mean_seconds'])} "
            f"{format_seconds(row['median_seconds'])} "
            f"{format_seconds(row['min_seconds'])} "
            f"{row['mean_best_loss']:11.4g} "
            f"{row['mean_mse']:11.4g}   "
            f"{row['last_equation']}"
        )


def write_json(path: Path, results: list[dict[str, Any]]) -> None:
    payload = {
        "results": results,
        "summary": summarize(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "mode",
        "repeat",
        "seconds",
        "worker_seconds",
        "best_loss",
        "mse",
        "equation",
        "n_equations",
        "backend_version",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def run_requested_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = ["fit", "subprocess"] if args.mode == "both" else [args.mode]
    results = []

    for mode in modes:
        for backend in args.backends:
            for warmup_index in range(args.warmups):
                if mode == "fit":
                    run_fit_once(
                        backend,
                        args,
                        repeat_index=-(warmup_index + 1),
                        include_import=False,
                    )
                else:
                    run_subprocess_once(
                        backend,
                        args,
                        repeat_index=-(warmup_index + 1),
                    )

            for repeat_index in range(args.repeats):
                if mode == "fit":
                    result = run_fit_once(
                        backend,
                        args,
                        repeat_index=repeat_index,
                        include_import=False,
                    )
                else:
                    result = run_subprocess_once(
                        backend,
                        args,
                        repeat_index=repeat_index,
                    )
                result["mode"] = mode
                results.append(result)
                print(
                    f"{backend:<5} {mode:<10} repeat={repeat_index} "
                    f"seconds={result['seconds']:.3f} "
                    f"loss={result['best_loss']:.4g} "
                    f"equation={result['equation']}"
                )
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--mode", choices=["fit", "subprocess", "both"], default="fit")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--niterations", type=int, default=3)
    parser.add_argument("--populations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--ncycles-per-iteration", type=int, default=100)
    parser.add_argument("--maxsize", type=int, default=20)
    parser.add_argument("--maxdepth", type=int, default=10)
    parser.add_argument(
        "--parallelism",
        choices=["serial", "multithreading"],
        default="serial",
        help="Use serial by default so deterministic=True is accepted by both backends.",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic=True. Useful when benchmarking multithreading.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)

    parser.add_argument("--_worker-backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.features < 2:
        parser.error("--features must be at least 2")
    if args._worker_backend and args._worker_output is None:
        parser.error("--_worker-output is required with --_worker-backend")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if args._worker_backend:
        result = run_fit_once(
            args._worker_backend,
            args,
            repeat_index=args._worker_repeat,
            include_import=True,
        )
        args._worker_output.write_text(json.dumps(result), encoding="utf-8")
        return 0

    print("Dataset target: y = 1.5*sin(x0) + 0.75*x1^2 - 0.25*x0*x1 + 0.5")
    print(
        f"samples={args.samples} features={args.features} "
        f"niterations={args.niterations} repeats={args.repeats} "
        f"mode={args.mode}"
    )

    results = run_requested_benchmarks(args)
    print_summary(results)

    if args.json_output is not None:
        write_json(args.json_output, results)
        print(f"\nWrote JSON results to {args.json_output}")
    if args.csv_output is not None:
        write_csv(args.csv_output, results)
        print(f"Wrote CSV results to {args.csv_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
