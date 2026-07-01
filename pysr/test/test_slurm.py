from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


class TestSlurm(unittest.TestCase):
    def setUp(self):
        if os.environ.get("PYSR_RUN_SLURM_TESTS", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise unittest.SkipTest(
                "Set PYSR_RUN_SLURM_TESTS=1 to enable Slurm Docker tests."
            )
        if shutil.which("docker") is None:
            raise unittest.SkipTest("docker not found on PATH.")

        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.cluster_dir = Path(__file__).resolve().parent / "slurm_docker_cluster"

        self.data_dir_obj = tempfile.TemporaryDirectory(prefix="pysr-slurm-data-")
        self.addCleanup(self.data_dir_obj.cleanup)
        self.data_dir = Path(self.data_dir_obj.name).resolve()

        self.julia_depot_dir_obj = None
        julia_depot_dir = os.environ.get("PYSR_SLURM_TEST_JULIA_DEPOT_DIR")
        if julia_depot_dir is None:
            self.julia_depot_dir_obj = tempfile.TemporaryDirectory(
                prefix="pysr-slurm-julia-depot-"
            )
            self.addCleanup(self.julia_depot_dir_obj.cleanup)
            julia_depot_dir = self.julia_depot_dir_obj.name

        self.compose_env = dict(os.environ)
        # Uniquify per-test so parallel/unexpected leftovers don't collide.
        self.compose_env["COMPOSE_PROJECT_NAME"] = (
            f"pysrslurm{os.getpid()}_{time.time_ns()}"
        )
        self.compose_env["PYSR_SLURM_TEST_DATA_DIR"] = str(self.data_dir)
        self.compose_env["PYSR_SLURM_TEST_JULIA_DEPOT_DIR"] = str(
            Path(julia_depot_dir).resolve()
        )

        info = _run(["docker", "info"], env=self.compose_env)
        if info.returncode != 0:
            raise unittest.SkipTest(f"Docker daemon not reachable:\n{info.stdout}")

        build = _run(
            ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "pysr-slurm"],
            cwd=self.repo_root,
            env=self.compose_env,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Failed to build pysr-slurm image:\n{build.stdout}")

        up = _run(
            ["docker", "compose", "up", "-d"],
            cwd=self.cluster_dir,
            env=self.compose_env,
        )
        if up.returncode != 0:
            raise RuntimeError(f"Failed to start Slurm cluster:\n{up.stdout}")

        self.addCleanup(
            lambda: _run(
                ["docker", "compose", "down", "-v"],
                cwd=self.cluster_dir,
                env=self.compose_env,
            )
        )

        self._wait_for_cluster_ready(timeout_s=300)

    def _wait_for_cluster_ready(self, *, timeout_s: int):
        start = time.time()
        last_out = ""
        while True:
            ping = _run(
                [
                    "docker",
                    "compose",
                    "exec",
                    "-T",
                    "slurmctld",
                    "bash",
                    "-lc",
                    "scontrol ping",
                ],
                cwd=self.cluster_dir,
                env=self.compose_env,
            )
            last_out = ping.stdout
            if ping.returncode == 0 and "Slurmctld" in ping.stdout:
                sinfo = _run(
                    [
                        "docker",
                        "compose",
                        "exec",
                        "-T",
                        "slurmctld",
                        "bash",
                        "-lc",
                        "sinfo -N -h -o '%N %T'",
                    ],
                    cwd=self.cluster_dir,
                    env=self.compose_env,
                )
                if sinfo.returncode == 0:
                    states = {}
                    for line in sinfo.stdout.splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            states[parts[0]] = parts[1].lower()
                    if states.get("c1") == "idle" and states.get("c2") == "idle":
                        return
                last_out = sinfo.stdout
            if time.time() - start > timeout_s:
                logs = _run(
                    ["docker", "compose", "logs", "--no-color", "c1", "c2"],
                    cwd=self.cluster_dir,
                    env=self.compose_env,
                ).stdout
                raise RuntimeError(
                    f"Slurm cluster did not become ready in {timeout_s}s:\n{last_out}\n\n{logs}"
                )
            time.sleep(2)

    def test_pysr_slurm_cluster_manager(self):
        job_script = self.data_dir / "pysr_slurm_job.sh"
        job_script.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "#SBATCH --job-name=pysr-slurm-test",
                    "#SBATCH --partition=normal",
                    "#SBATCH --nodes=2",
                    "#SBATCH --ntasks=4",
                    "#SBATCH --time=40:00",
                    "set -euo pipefail",
                    "python3 - <<'PY'",
                    "import numpy as np",
                    "from pysr import PySRRegressor",
                    "X = np.random.RandomState(0).randn(30, 2)",
                    "y = X[:, 0] + 1.0",
                    "model = PySRRegressor(",
                    "    niterations=3,",
                    "    populations=3,",
                    "    progress=False,",
                    "    temp_equation_file=True,",
                    "    parallelism='multiprocessing',",
                    "    procs=2,",
                    "    cluster_manager='slurm',",
                    "    verbosity=0,",
                    ")",
                    "model.fit(X, y)",
                    "print('PYSR_SLURM_OK')",
                    "PY",
                ]
            )
            + "\n"
        )
        job_script.chmod(0o755)

        submit = _run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "slurmctld",
                "bash",
                "-lc",
                f"cd /data && sbatch {job_script.name}",
            ],
            cwd=self.cluster_dir,
            env=self.compose_env,
        )
        self.assertEqual(submit.returncode, 0, msg=submit.stdout)
        match = re.search(r"Submitted batch job (\d+)", submit.stdout)
        self.assertIsNotNone(match, msg=f"Could not parse job id:\n{submit.stdout}")
        job_id = match.group(1)

        start = time.time()
        while True:
            state = _run(
                [
                    "docker",
                    "compose",
                    "exec",
                    "-T",
                    "slurmctld",
                    "bash",
                    "-lc",
                    f"scontrol show job {job_id} -o",
                ],
                cwd=self.cluster_dir,
                env=self.compose_env,
            )
            self.assertEqual(state.returncode, 0, msg=state.stdout)
            if "JobState=COMPLETED" in state.stdout:
                break
            if (
                "JobState=FAILED" in state.stdout
                or "JobState=CANCELLED" in state.stdout
            ):
                out = _run(
                    [
                        "docker",
                        "compose",
                        "exec",
                        "-T",
                        "slurmctld",
                        "bash",
                        "-lc",
                        f"tail -n 200 /data/slurm-{job_id}.out 2>/dev/null || true",
                    ],
                    cwd=self.cluster_dir,
                    env=self.compose_env,
                ).stdout
                self.fail(
                    f"Slurm job {job_id} did not complete successfully:\n{state.stdout}\n\n{out}"
                )
            if time.time() - start > 2400:
                out = _run(
                    [
                        "docker",
                        "compose",
                        "exec",
                        "-T",
                        "slurmctld",
                        "bash",
                        "-lc",
                        f"tail -n 200 /data/slurm-{job_id}.out 2>/dev/null || true",
                    ],
                    cwd=self.cluster_dir,
                    env=self.compose_env,
                ).stdout
                self.fail(
                    f"Timed out waiting for Slurm job {job_id}:\n{state.stdout}\n\n{out}"
                )
            time.sleep(5)

        output = _run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "slurmctld",
                "bash",
                "-lc",
                f"cat /data/slurm-{job_id}.out",
            ],
            cwd=self.cluster_dir,
            env=self.compose_env,
        )
        self.assertEqual(output.returncode, 0, msg=output.stdout)
        self.assertIn("PYSR_SLURM_OK", output.stdout)


def runtests(just_tests=False):
    tests = [TestSlurm]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
