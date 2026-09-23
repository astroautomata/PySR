"""End-to-end tests for graceful SIGINT handling during a search."""

import ctypes
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pysr.interrupt as interrupt_module
from pysr.sr import _resolve_input_stream as sr_resolve

# Native sigaction checks are only available on POSIX.
POSIX = os.name == "posix"


class TestResolveInputStream(unittest.TestCase):
    def test_default_uses_stdin_on_interactive_terminal(self):
        with mock.patch.object(sys, "stdin") as stdin:
            stdin.isatty.return_value = True
            self.assertEqual(sr_resolve(None), "stdin")

    def test_default_disables_stdin_watching_without_terminal(self):
        with mock.patch.object(sys, "stdin") as stdin:
            stdin.isatty.return_value = False
            self.assertEqual(sr_resolve(None), "devnull")

    def test_default_handles_missing_or_closed_stdin(self):
        with mock.patch.object(sys, "stdin", None):
            self.assertEqual(sr_resolve(None), "devnull")
        with mock.patch.object(sys, "stdin") as stdin:
            stdin.isatty.side_effect = ValueError("I/O operation on closed file")
            self.assertEqual(sr_resolve(None), "devnull")

    def test_explicit_values_pass_through(self):
        for value in ("stdin", "devnull", "Main.my_stream"):
            self.assertEqual(sr_resolve(value), value)


@unittest.skipUnless(POSIX, "SIGINT signal contexts are POSIX-only")
class TestExternalStopSignalContext(unittest.TestCase):
    def setUp(self):
        self.native_handler_before = self._native_handler()
        self.python_before = signal.getsignal(signal.SIGINT)
        self.stop_fds = []
        backend = SimpleNamespace(ExternalStop=self._external_stop)
        self.patches = (
            mock.patch.dict(os.environ, {"PYTHON_JULIACALL_HANDLE_SIGNALS": "yes"}),
            mock.patch.object(interrupt_module, "SymbolicRegression", backend),
        )
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.assertEqual(signal.getsignal(signal.SIGINT), self.python_before)
        self.assertEqual(self._native_handler(), self.native_handler_before)

    def _external_stop(self, read_fd, trigger):
        self.stop_fds.append(read_fd)
        return SimpleNamespace(fd=read_fd, trigger=trigger)

    @staticmethod
    def _native_handler():
        # Compare only `sa_handler`. It is the whole of what this test
        # asserts, and as the first member on both Linux and Darwin it needs
        # no platform-specific struct offsets to read.
        class Action(ctypes.Union):
            _fields_ = [
                ("handler", ctypes.c_void_p),
                ("alignment", ctypes.c_longdouble),
                ("storage", ctypes.c_ubyte * 1024),
            ]

        libc = ctypes.CDLL(None, use_errno=True)
        libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        libc.sigaction.restype = ctypes.c_int
        action = Action()
        result = libc.sigaction(signal.SIGINT, None, ctypes.byref(action))
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return action.handler

    @staticmethod
    def _current_wakeup_fd():
        current = signal.set_wakeup_fd(-1)
        signal.set_wakeup_fd(current)
        return current

    def _assert_stop_fds_closed(self):
        for fd in self.stop_fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_cleanup_after_normal_return_restores_existing_wakeup_fd(self):
        wakeup_read, wakeup_write = os.pipe()
        os.set_blocking(wakeup_write, False)
        previous_wakeup = signal.set_wakeup_fd(wakeup_write)
        try:
            model = SimpleNamespace()
            with interrupt_module._external_stop_signal_context(model) as external_stop:
                self.assertEqual(external_stop.fd, self.stop_fds[0])
                self.assertEqual(external_stop.trigger, signal.SIGINT)
                self.assertFalse(os.get_blocking(external_stop.fd))
                self.assertFalse(os.get_blocking(self._current_wakeup_fd()))
            self.assertEqual(self._current_wakeup_fd(), wakeup_write)
            self.assertFalse(model.interrupted_)
            self._assert_stop_fds_closed()
        finally:
            signal.set_wakeup_fd(previous_wakeup)
            os.close(wakeup_write)
            os.close(wakeup_read)

    def test_cleanup_after_search_error(self):
        model = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "search failed"):
            with interrupt_module._external_stop_signal_context(model):
                raise RuntimeError("search failed")
        self._assert_stop_fds_closed()

    def test_cleanup_continues_after_cleanup_failure(self):
        real_set_wakeup_fd = signal.set_wakeup_fd
        calls = []

        def fail_after_restoring(fd):
            result = real_set_wakeup_fd(fd)
            calls.append(fd)
            if len(calls) == 2:
                raise RuntimeError("injected cleanup failure")
            return result

        model = SimpleNamespace()
        with mock.patch.object(signal, "set_wakeup_fd", fail_after_restoring):
            with self.assertRaisesRegex(RuntimeError, "injected cleanup failure"):
                with interrupt_module._external_stop_signal_context(model):
                    pass
        self._assert_stop_fds_closed()

    def test_signals_disabled_does_not_arm(self):
        model = SimpleNamespace(interrupted_=True)
        with (
            mock.patch.dict(os.environ, {"PYTHON_JULIACALL_HANDLE_SIGNALS": "no"}),
            mock.patch.object(os, "pipe") as pipe,
        ):
            with interrupt_module._external_stop_signal_context(model) as external_stop:
                self.assertIsNone(external_stop)
        pipe.assert_not_called()
        self.assertFalse(model.interrupted_)

    def test_interrupted_return_sets_attribute_and_warns(self):
        model = SimpleNamespace()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with interrupt_module._external_stop_signal_context(model):
                handler = signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT, None)
        self.assertTrue(model.interrupted_)
        self.assertTrue(any("partial" in str(item.message).lower() for item in caught))


# The child asserts partial results, restored native SIGINT disposition on
# POSIX, and that a second fit in the same process works.
CHILD_SCRIPT = textwrap.dedent("""
    import ctypes
    import os
    import signal
    import threading
    import warnings

    import numpy as np

    from pysr import PySRRegressor

    if os.name == "posix":
        libc = ctypes.CDLL(None)

        def native_handler():
            # Compare only `sa_handler`: the first member on both platforms, and
            # all this check needs.
            storage = (ctypes.c_char * 512)()
            libc.sigaction(int(signal.SIGINT), None, storage)
            return ctypes.cast(storage, ctypes.POINTER(ctypes.c_void_p)).contents.value

        before = native_handler()

    def send_interrupt():
        if os.name == "posix":
            os.kill(os.getpid(), signal.SIGINT)
        elif not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0):
            raise ctypes.WinError()

    rstate = np.random.RandomState(0)
    X = rstate.randn(150, 2)
    y = X[:, 0] * X[:, 1]
    warmup = PySRRegressor(
        niterations=1,
        populations=3,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    warmup.fit(X[:20], y[:20])

    model = PySRRegressor(
        niterations=1_000_000,  # only an interrupt can end this fit
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    # Warm-up is complete, so the delayed interrupt reaches a running search.
    threading.Timer(10.0, send_interrupt).start()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    assert model.equations_ is not None
    assert len(model.equations_) > 0, "search returned no partial results"
    assert model.interrupted_ is True
    assert any("partial" in str(item.message).lower() for item in caught)
    print(f"INTERRUPTED_OK:{len(model.equations_)}", flush=True)

    if os.name == "posix":
        print(f"HANDLER_RESTORED:{native_handler() == before}", flush=True)

    model2 = PySRRegressor(
        niterations=2,
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model2.fit(X, y)
    assert model2.equations_ is not None
    assert model2.interrupted_ is False
    print("SECOND_FIT_OK", flush=True)
    """)

# Wait this long after Jupyter's SEARCHING marker so the fit has armed the
# cooperative handler before the single user interrupt.
FIRST_SIGNAL_DELAY = 15.0
TOTAL_TIMEOUT = 600.0


class TestSubprocessInterrupt(unittest.TestCase):
    def test_sigint_returns_partial_results_and_restores_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "child.py"
            script.write_text(CHILD_SCRIPT)
            p = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=0 if POSIX else subprocess.CREATE_NEW_CONSOLE,
            )
            watchdog = threading.Timer(TOTAL_TIMEOUT, p.kill)
            watchdog.start()
            lines = []
            try:
                for line in p.stdout:
                    lines.append(line.strip())
                p.wait()
            finally:
                watchdog.cancel()
                if p.poll() is None:
                    p.kill()

            output = "\n".join(lines)
            self.assertEqual(p.returncode, 0, f"child failed:\n{output}")
            self.assertIn("INTERRUPTED_OK:", output)
            if POSIX:
                self.assertIn("HANDLER_RESTORED:True", output)
            self.assertIn("SECOND_FIT_OK", output)


JUPYTER_FIT_CELL = textwrap.dedent("""
    import numpy as np
    from pysr import PySRRegressor

    rstate = np.random.RandomState(0)
    X = rstate.randn(150, 2)
    y = X[:, 0] * X[:, 1]

    # Compile the search pipeline before the timed window opens, so that the
    # interrupt lands on a running search rather than on Julia's warm-up.
    warmup = PySRRegressor(
        niterations=1,
        populations=3,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    warmup.fit(X[:20], y[:20])

    model = PySRRegressor(
        niterations=1_000_000,
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    print("SEARCHING", flush=True)
    model.fit(X, y)
    assert model.interrupted_ is True
    assert len(model.equations_) > 0, "search returned no partial results"
    print("INTERRUPTED_OK", len(model.equations_), flush=True)
    """)


class TestJupyterInterrupt(unittest.TestCase):
    """Drive a real ipykernel and interrupt it exactly like the Jupyter UI."""

    def test_kernel_interrupt_stops_search_and_kernel_survives(self):
        try:
            from jupyter_client.manager import start_new_kernel
        except ImportError:
            self.skipTest("jupyter_client is not installed")

        km, kc = start_new_kernel()
        try:
            msg_id = kc.execute(JUPYTER_FIT_CELL)
            self._await_stream(kc, "SEARCHING", timeout=TOTAL_TIMEOUT)

            # Interrupt once, exactly like one click on Jupyter's interrupt action.
            time.sleep(FIRST_SIGNAL_DELAY)
            km.interrupt_kernel()
            reply = self._await_reply(kc, msg_id, timeout=TOTAL_TIMEOUT)
            assert reply is not None, "kernel never returned from fit"
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
    tests = [
        TestResolveInputStream,
        TestExternalStopSignalContext,
        TestSubprocessInterrupt,
        TestJupyterInterrupt,
    ]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
