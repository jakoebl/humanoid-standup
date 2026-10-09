"""Replace the previous RSL-RL iteration table in interactive terminals."""

import io
import os
import re
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from functools import wraps


@contextmanager
def latest_iteration(runner):
    """Keep persistent metric writers intact and refresh only console tables."""
    stream = sys.stdout
    if not stream.isatty() or os.environ.get("TERM") == "dumb":
        yield
        return

    # RSL-RL 3.x moved log() from the runner to a separate logger.
    owner = runner.logger if hasattr(runner, "logger") else runner
    original_log = owner.log
    previous_rows = 0
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    @wraps(original_log)
    def log(*args, **kwargs):
        nonlocal previous_rows
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                return original_log(*args, **kwargs)
        finally:
            text = output.getvalue()
            if text:
                if previous_rows:
                    stream.write(f"\x1b[{previous_rows}A\r\x1b[J")
                stream.write(text)
                stream.flush()
                columns = shutil.get_terminal_size().columns
                previous_rows = sum(
                    max(1, (len(ansi.sub("", line)) + columns - 1) // columns)
                    for line in text.splitlines()
                )

    owner.log = log
    try:
        yield
    finally:
        owner.log = original_log
