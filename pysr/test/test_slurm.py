from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import cast


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
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

        self.docker_config_obj = tempfile.TemporaryDirectory(
            prefix="pysr-slurm-docker-config-"
        )
        self.addCleanup(self.docker_config_obj.cleanup)

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
        self.compose_env["DOCKER_CONFIG"] = self.docker_config_obj.name
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

    def _run_sbatch(
        self,
        job_script: Path,
        *,
        timeout_s: int = 2400,
    ) -> str:
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
        if match is None:
            self.fail(f"Could not parse job id:\n{submit.stdout}")
        job_id = cast(str, match.group(1))

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
            if time.time() - start > timeout_s:
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
        return output.stdout

    def _assert_scontrol_step_usage(
        self,
        output: str,
        *,
        expected_tasks: int,
        expected_nodes: int,
        expected_nodelist: set[str] | None = None,
        label: str,
    ) -> None:
        marker = "PYSR_SCONTROL_STEP_SAMPLE"
        self.assertIn(
            marker, output, msg=f"{label}: did not capture any scontrol samples."
        )

        saw_expected = False
        for sample in output.split(marker)[1:]:
            step_lines = [
                line.strip()
                for line in sample.splitlines()
                if line.strip().startswith("StepId=")
            ]
            steps = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in step_lines]
            running_steps: dict[str, dict[str, str]] = {}
            for s in steps:
                step_id = s.get("StepId", "")
                if step_id.endswith(".batch") or s.get("State", "") != "RUNNING":
                    continue
                running_steps[step_id] = s

            total_tasks = 0
            for step in running_steps.values():
                tasks = int(step.get("Tasks", "0"))
                nodes = int(step.get("Nodes", "0"))
                if tasks > expected_tasks:
                    self.fail(
                        f"{label}: too many tasks in Slurm step: expected <= {expected_tasks}, got {tasks}.\n\n{sample}"
                    )
                if tasks == expected_tasks and nodes != expected_nodes:
                    self.fail(
                        f"{label}: expected {expected_tasks} tasks across {expected_nodes} nodes, got Nodes={nodes}.\n\n{sample}"
                    )
                if tasks == expected_tasks and nodes == expected_nodes:
                    if expected_nodelist is not None:
                        node_list = step.get("NodeList")
                        if not node_list:
                            self.fail(
                                f"{label}: expected NodeList to be present in scontrol step output.\n\n{sample}"
                            )
                        if node_list not in expected_nodelist:
                            self.fail(
                                f"{label}: expected NodeList in {sorted(expected_nodelist)}, got {node_list!r}.\n\n{sample}"
                            )
                    saw_expected = True
                total_tasks += tasks

            if total_tasks > expected_tasks:
                self.fail(
                    f"{label}: too many concurrent tasks across steps: expected <= {expected_tasks}, got {total_tasks}.\n\n{sample}"
                )

        tail = "\n".join(output.splitlines()[-200:])
        self.assertTrue(
            saw_expected,
            msg=(
                f"{label}: never observed a RUNNING Slurm step with Tasks={expected_tasks} "
                f"and Nodes={expected_nodes}.\n\nLast 200 lines:\n{tail}"
            ),
        )

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
        def _assert_worker_distribution(
            output: str,
            *,
            expected_procs: int,
            expected_nodes: int,
            expected_per_node: int,
        ) -> None:
            worker_hosts = re.findall(
                r"Worker \d+ ready on host ([^,]+), port \d+",
                output,
            )
            self.assertEqual(
                len(worker_hosts),
                expected_procs,
                msg=f"Expected {expected_procs} workers.\n\n{output}",
            )
            counts: dict[str, int] = {}
            for host in worker_hosts:
                counts[host] = counts.get(host, 0) + 1
            self.assertEqual(
                len(counts),
                expected_nodes,
                msg=f"Expected workers on {expected_nodes} nodes.\n\n{output}",
            )
            self.assertTrue(
                all(v == expected_per_node for v in counts.values()),
                msg=f"Expected exactly {expected_per_node} workers per node.\n\n{output}",
            )

        def _run_case(*, ntasks_per_node: int, procs: int, seed: int) -> str:
            marker = f"PYSR_SLURM_OK:slurm:{procs}"
            mismatch_marker = "PYSR_SLURM_MISMATCH_REJECTED"
            job = self.data_dir / f"pysr_slurm_job_{procs}.sh"
            python_lines = [
                "import os",
                "os.environ['JULIA_DEBUG'] = 'SlurmClusterManager'",
                "import numpy as np",
                "from pysr import PySRRegressor",
                f"X = np.random.RandomState({seed}).randn(30, 2)",
                "y = X[:, 0] + 1.0",
            ]
            if procs == 2:
                python_lines.extend(
                    [
                        "from juliacall import JuliaError",
                        "from pysr.julia_extensions import load_required_packages",
                        "from pysr.julia_helpers import load_cluster_manager",
                        "load_required_packages(cluster_manager='slurm')",
                        "addprocs_slurm = load_cluster_manager('slurm')",
                        "try:",
                        "    addprocs_slurm(1)",
                        "except JuliaError as exc:",
                        "    expected = 'Requested 1 processes, but Slurm allocation has 2 tasks.'",
                        "    assert expected in str(exc), str(exc)",
                        "else:",
                        "    raise AssertionError('A mismatched procs value was accepted.')",
                        f"print('{mismatch_marker}')",
                    ]
                )
            python_lines.extend(
                [
                    "model = PySRRegressor(",
                    "    niterations=2,",
                    "    populations=2,",
                    "    progress=False,",
                    "    temp_equation_file=True,",
                    "    parallelism='multiprocessing',",
                    f"    procs={procs},",
                    "    cluster_manager='slurm',",
                    "    verbosity=0,",
                    ")",
                    "model.fit(X, y)",
                    f"print('{marker}')",
                ]
            )
            job.write_text(
                "\n".join(
                    [
                        "#!/bin/bash",
                        f"#SBATCH --job-name=pysr-slurm-test-{procs}",
                        "#SBATCH --partition=normal",
                        "#SBATCH --nodes=2",
                        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
                        "#SBATCH --time=40:00",
                        "set -euo pipefail",
                        'jobid="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"',
                        'if [ -z "$jobid" ]; then echo "Missing SLURM_JOB_ID/SLURM_JOBID" >&2; exit 1; fi',
                        "monitor_steps() {",
                        "  while true; do",
                        "    echo PYSR_SCONTROL_STEP_SAMPLE",
                        '    scontrol show step "$jobid" -o 2>/dev/null || true',
                        "    sleep 1",
                        "  done",
                        "}",
                        "monitor_steps &",
                        "MONITOR_PID=$!",
                        "trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT",
                        "python3 - <<'PY'",
                        *python_lines,
                        "PY",
                    ]
                )
                + "\n"
            )
            job.chmod(0o755)

            output = self._run_sbatch(job)
            self.assertIn(marker, output)
            self.assertEqual(
                len(re.findall(rf"^{re.escape(marker)}$", output, flags=re.MULTILINE)),
                1,
                msg=f"Expected marker exactly once.\n\n{output}",
            )
            self.assertEqual(
                len(
                    re.findall(r"^\[ Info: Starting SLURM job .*", output, re.MULTILINE)
                ),
                0,
                msg=(
                    "Expected Slurm backend to use SlurmClusterManager (allocation-based), "
                    "not ClusterManagers.\n\n" + output
                ),
            )
            if procs == 2:
                self.assertIn(mismatch_marker, output)
                before_mismatch = output.split(mismatch_marker, maxsplit=1)[0]
                self.assertNotRegex(
                    before_mismatch,
                    r"Worker \d+ ready on host",
                    msg="A mismatched procs value launched workers before failing.",
                )
            return output

        cases = [
            dict(ntasks_per_node=1, procs=2, seed=0),
            dict(ntasks_per_node=3, procs=6, seed=2),
        ]
        for case in cases:
            output = _run_case(**case)
            self._assert_scontrol_step_usage(
                output,
                expected_tasks=case["procs"],
                expected_nodes=2,
                expected_nodelist={"c1,c2", "c[1-2]"},
                label=f"slurm({case['procs']})",
            )
            _assert_worker_distribution(
                output,
                expected_procs=case["procs"],
                expected_nodes=2,
                expected_per_node=case["ntasks_per_node"],
            )


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
