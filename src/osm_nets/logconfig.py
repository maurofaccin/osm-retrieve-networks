"""Configure the logs."""

import logging
import logging.config
import sys

# ANSI color codes for terminals
_COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors to the levelname and keeps a clear single-line output:
    [YYYY-MM-DD HH:MM:SS.mmm] LEVEL module:line - message
    """

    def format(self, record):
        levelname = record.levelname
        color = _COLORS.get(levelname, "")
        record.levelname = f"{color}{levelname}{_RESET}" if color else levelname
        # Include module and line number for quick tracing
        if not hasattr(record, "module_line"):
            record.module_line = f"{record.module}:{record.lineno}"
        return super().format(record)


def setup_logging(level="INFO"):
    """
    Configure root logger for console output.
    Call setup_logging() early in your program.
    """
    log_level = getattr(logging, str(level).upper(), logging.INFO)

    fmt = "[%(asctime)s.%(msecs)03d] %(levelname)s %(module_line)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
    root = logging.getLogger()
    root.handlers[:] = []  # remove existing handlers
    root.addHandler(handler)
    root.setLevel(log_level)
