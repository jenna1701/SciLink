"""Capture stdout/stderr while still printing to the original streams."""

import io
import sys
import threading


class TeeStream:
    """A stream that writes to both the original stream and a StringIO buffer."""

    def __init__(self, original: io.TextIOBase, buffer: io.StringIO):
        self._original = original
        self._buffer = buffer
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self._original.write(data)
            self._buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()

    # Delegate attribute access so libraries checking .isatty() etc. still work.
    def __getattr__(self, name: str):
        return getattr(self._original, name)


class OutputCapture:
    """Context manager that captures stdout/stderr through a TeeStream.

    Usage::

        with OutputCapture() as cap:
            agent.chat("hello")
        print(cap.getvalue())
    """

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self) -> "OutputCapture":
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = TeeStream(self._old_stdout, self._buffer)
        sys.stderr = TeeStream(self._old_stderr, self._buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def getvalue(self) -> str:
        return self._buffer.getvalue()
