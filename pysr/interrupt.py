"""Cooperative SIGINT handling so searches stop gracefully instead of killing the process."""

from __future__ import annotations

import _thread
import ctypes
import os
import signal
import socket
import threading
import warnings
from contextlib import ExitStack, contextmanager

from .julia_import import SymbolicRegression


class _SigactionStorage(ctypes.Union):
    _fields_ = [
        ("alignment", ctypes.c_longdouble),
        ("storage", ctypes.c_ubyte * 1024),
    ]


def _libc_with_sigaction():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.sigaction.restype = ctypes.c_int
    return libc


def _checked_sigaction(libc, action, old_action):
    result = libc.sigaction(signal.SIGINT, action, old_action)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


@contextmanager
def _console_ctrl_c_to_python():
    """Hand console Ctrl-C to Python before Julia's handler exits the process.

    Windows calls console handlers last-registered first, so this one runs first.
    """
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel32.SetConsoleCtrlHandler.argtypes = [HANDLER, wintypes.BOOL]
    kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL

    def handler(event):
        if event == 0:  # CTRL_C_EVENT
            _thread.interrupt_main()
            return True
        return False

    callback = HANDLER(handler)
    if not kernel32.SetConsoleCtrlHandler(callback, True):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not kernel32.SetConsoleCtrlHandler(callback, False):
            raise ctypes.WinError(ctypes.get_last_error())


def _should_arm_external_stop() -> bool:
    return (
        threading.current_thread() is threading.main_thread()
        and os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") == "yes"
    )


def _stop_channel(cleanup: ExitStack) -> tuple[int, int]:
    """Open a nonblocking pair of descriptors carrying the signal wakeup byte.

    Windows writes that byte with `send` and can only poll sockets, so a socket
    pair is the one transport that works everywhere.
    """
    read_end, write_end = socket.socketpair()
    cleanup.callback(read_end.close)
    cleanup.callback(write_end.close)
    read_end.setblocking(False)
    write_end.setblocking(False)
    return read_end.fileno(), write_end.fileno()


@contextmanager
def _external_stop_signal_context(model):
    interrupted = False
    external_stop = None

    with ExitStack() as cleanup:
        if _should_arm_external_stop():
            stop_read_fd, stop_write_fd = _stop_channel(cleanup)
            external_stop = SymbolicRegression.ExternalStop(stop_read_fd, signal.SIGINT)

            # Julia intercepts Ctrl-C through SIGINT sigaction on POSIX, which
            # Python replaces below, or through a Windows console handler that
            # runs before Python's. Save the former or preempt the latter.
            if os.name == "posix":
                libc = _libc_with_sigaction()
                saved_sigaction = _SigactionStorage()
                _checked_sigaction(libc, None, ctypes.byref(saved_sigaction))
                cleanup.callback(
                    _checked_sigaction, libc, ctypes.byref(saved_sigaction), None
                )
            else:
                cleanup.enter_context(_console_ctrl_c_to_python())

            saved_python_handler = signal.getsignal(signal.SIGINT)

            def record_interrupt(*_):
                nonlocal interrupted
                interrupted = True

            signal.signal(signal.SIGINT, record_interrupt)
            cleanup.callback(signal.signal, signal.SIGINT, saved_python_handler)
            previous_wakeup_fd = signal.set_wakeup_fd(stop_write_fd)
            cleanup.callback(signal.set_wakeup_fd, previous_wakeup_fd)

        yield external_stop

    model.interrupted_ = interrupted
    if interrupted:
        warnings.warn(
            "The search was interrupted. Returning partial results.",
            RuntimeWarning,
            stacklevel=3,
        )
