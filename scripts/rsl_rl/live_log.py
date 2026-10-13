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
    previous_size = None
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    @wraps(original_log)
    def log(*args, **kwargs):
        nonlocal previous_rows, previous_size
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                return original_log(*args, **kwargs)
        finally:
            text = output.getvalue()
            if text:
                size = shutil.get_terminal_size()
                # Leave a column spare to avoid terminal-dependent autowrap,
                # and a row spare for the cursor after the final newline.
                columns = max(1, size.columns - 1)
                available_rows = max(1, size.lines - 1)
                lines = []
                for line in ansi.sub("", text).expandtabs().splitlines():
                    if line:
                        lines.extend(line[index : index + columns] for index in range(0, len(line), columns))
                    else:
                        lines.append("")
                if len(lines) > available_rows:
                    # Preserve the iteration heading and footer on small screens.
                    omitted = len(lines) - available_rows + 1
                    notice = f"... {omitted} rows hidden; enlarge terminal to show all metrics ..."[:columns]
                    footer_rows = min(4, max(0, available_rows - 2))
                    heading_rows = available_rows - footer_rows - 1
                    lines = lines[:heading_rows] + [notice] + (lines[-footer_rows:] if footer_rows else [])

                if previous_rows and size != previous_size:
                    # Resizing can reflow old lines, invalidating cursor offsets.
                    # Reset the visible screen; startup scrollback stays intact.
                    stream.write("\x1b[H\x1b[J")
                elif previous_rows:
                    stream.write(f"\x1b[{previous_rows}A\r\x1b[J")
                stream.write("\n".join(lines) + "\n")
                stream.flush()
                previous_rows = len(lines)
                previous_size = size

    owner.log = log
    try:
        yield
    finally:
        owner.log = original_log
