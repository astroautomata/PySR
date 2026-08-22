"""End-to-end tests for graceful SIGINT handling during a search.

Both tests drive a *separate* process (a plain Python subprocess and a real
Jupyter kernel) so that signal delivery is tested exactly as a user would
trigger it. They skip themselves when the installed backend does not provide
the cooperative-stop API (``SymbolicRegression.stop_fd_trigger``).
"""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

# Interrupts are only delivered this way on POSIX; the feature is gated
# identically in `pysr/sr.py`.
POSIX = os.name == "posix"

# The child asserts partial results, a byte-identical native SIGINT
# disposition after the fit, and that a second fit in the same process works.
CHILD_SCRIPT = textwrap.dedent("""
    import ctypes
    import os
    import signal
    import sys

    import numpy as np

    from pysr import PySRRegressor, jl

    supported = os.name == "posix" and jl.seval(
        "isdefined(SymbolicRegression, :stop_fd_trigger)"
    )
    print(f"SUPPORT:{supported}", flush=True)
    if not supported:
        sys.exit(0)

    libc = ctypes.CDLL(None)
    before = (ctypes.c_char * 512)()
    libc.sigaction(int(signal.SIGINT), None, before)

    rstate = np.random.RandomState(0)
    X = rstate.randn(150, 2)
    y = X[:, 0] * X[:, 1]
    model = PySRRegressor(
        niterations=1_000_000,  # only an interrupt can end this fit
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    print("SEARCHING", flush=True)
    model.fit(X, y)
    assert model.equations_ is not None
    print(f"INTERRUPTED_OK:{len(model.equations_)}", flush=True)

    after = (ctypes.c_char * 512)()
    libc.sigaction(int(signal.SIGINT), None, after)
    print(f"HANDLER_RESTORED:{bytes(before) == bytes(after)}", flush=True)

    model2 = PySRRegressor(
        niterations=2,
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model2.fit(X, y)
    assert model2.equations_ is not None
    print("SECOND_FIT_OK", flush=True)
    """)

# Wait this long after the SEARCHING marker before the first SIGINT, so the
# fit has armed the cooperative handler; repeat in case Julia's signal
# listener thread wins the delivery race for one signal.
FIRST_SIGNAL_DELAY = 15.0
SIGNAL_REPEAT_INTERVAL = 15.0
TOTAL_TIMEOUT = 600.0


class TestSubprocessInterrupt(unittest.TestCase):
    def test_sigint_returns_partial_results_and_restores_state(self):
        if not POSIX:
            self.skipTest("SIGINT-based interruption is POSIX-only")

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "child.py"
            script.write_text(CHILD_SCRIPT)
            p = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            watchdog = threading.Timer(TOTAL_TIMEOUT, p.kill)
            watchdog.start()

            def send_signals_after_marker():
                time.sleep(FIRST_SIGNAL_DELAY)
                while p.poll() is None:
                    try:
                        os.kill(p.pid, signal.SIGINT)
                    except ProcessLookupError:
                        return
                    time.sleep(SIGNAL_REPEAT_INTERVAL)

            lines = []
            signaler = None
            try:
                for line in p.stdout:
                    lines.append(line.strip())
                    if line.startswith("SUPPORT:False"):
                        p.wait()
                        self.skipTest("backend lacks the cooperative-stop API")
                    if line.startswith("SEARCHING"):
                        signaler = threading.Thread(
                            target=send_signals_after_marker, daemon=True
                        )
                        signaler.start()
                p.wait()
            finally:
                watchdog.cancel()
                if p.poll() is None:
                    p.kill()

            output = "\n".join(lines)
            self.assertEqual(p.returncode, 0, f"child failed:\n{output}")
            self.assertIn("INTERRUPTED_OK:", output)
            self.assertIn("HANDLER_RESTORED:True", output)
            self.assertIn("SECOND_FIT_OK", output)


JUPYTER_FIT_CELL = textwrap.dedent("""
    import numpy as np
    from pysr import PySRRegressor

    rstate = np.random.RandomState(0)
    X = rstate.randn(150, 2)
    y = X[:, 0] * X[:, 1]
    model = PySRRegressor(
        niterations=1_000_000,
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    print("SEARCHING", flush=True)
    model.fit(X, y)
    print("INTERRUPTED_OK", len(model.equations_), flush=True)
    """)


class TestJupyterInterrupt(unittest.TestCase):
    """Drive a real ipykernel and interrupt it exactly like the Jupyter UI."""

    def test_kernel_interrupt_stops_search_and_kernel_survives(self):
        if not POSIX:
            self.skipTest("SIGINT-based interruption is POSIX-only")
        try:
            from jupyter_client.manager import start_new_kernel
        except ImportError:
            self.skipTest("jupyter_client is not installed")

        km, kc = start_new_kernel()
        try:
            streams = []

            def collect(msg):
                if msg["msg_type"] == "stream":
                    streams.append(msg["content"]["text"])

            reply = kc.execute_interactive(
                "import os; from pysr import jl\n"
                "print('SUPPORT:', os.name == 'posix' and jl.seval("
                "'isdefined(SymbolicRegression, :stop_fd_trigger)'), flush=True)",
                timeout=TOTAL_TIMEOUT,
                output_hook=collect,
            )
            self.assertEqual(reply["content"]["status"], "ok")
            if "SUPPORT: True" not in "".join(streams):
                self.skipTest("backend lacks the cooperative-stop API")

            msg_id = kc.execute(JUPYTER_FIT_CELL)
            self._await_stream(kc, "SEARCHING", timeout=TOTAL_TIMEOUT)

            # Interrupt like the Jupyter UI (interrupt_mode="signal" sends
            # SIGINT to the kernel process); repeat in case one delivery is
            # consumed by Julia's signal listener thread.
            deadline = time.monotonic() + TOTAL_TIMEOUT
            time.sleep(FIRST_SIGNAL_DELAY)
            reply = None
            while reply is None:
                km.interrupt_kernel()
                reply = self._await_reply(kc, msg_id, timeout=SIGNAL_REPEAT_INTERVAL)
                self.assertLess(
                    time.monotonic(), deadline, "kernel never returned from fit"
                )
            # The fit returns normally with partial results: no error status.
            self.assertEqual(reply["content"]["status"], "ok")

            # The kernel must remain fully usable afterwards.
            reply = kc.execute_interactive(
                "print('ALIVE', len(model.equations_), flush=True)",
                timeout=60,
            )
            self.assertEqual(reply["content"]["status"], "ok")
        finally:
            kc.stop_channels()
            km.shutdown_kernel(now=True)

    # -- helpers -----------------------------------------------------------

    def _await_stream(self, kc, text, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = kc.get_iopub_msg(timeout=deadline - time.monotonic())
            if msg["msg_type"] == "stream" and text in msg["content"]["text"]:
                return
        raise AssertionError(f"never saw {text!r} on iopub")

    def _await_reply(self, kc, msg_id, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = kc.get_shell_msg(timeout=max(0.1, deadline - time.monotonic()))
            except Exception:
                return None
            if msg["parent_header"].get("msg_id") == msg_id:
                return msg
        return None


def runtests(just_tests=False):
    tests = [TestSubprocessInterrupt, TestJupyterInterrupt]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
