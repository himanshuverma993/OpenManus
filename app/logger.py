import sys
import tempfile
from datetime import datetime
from pathlib import Path

from loguru import logger as _logger

from app.config import PROJECT_ROOT


_print_level = "INFO"


def _log_file_path(log_name: str):
    """Return a writable log file path, falling back to the temp directory.

    The project directory can be read-only (e.g. a mounted dataset on Kaggle or
    a system package directory). Loguru raises at import time when it cannot
    create its file sink, which would take down the whole application.
    """
    preferred = PROJECT_ROOT / "logs" / f"{log_name}.log"
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        with open(preferred, "a"):
            pass
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "openmanus-logs" / f"{log_name}.log"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    try:
        _logger.add(_log_file_path(log_name), level=logfile_level)
    except OSError:
        # Never let logging setup break the application.
        pass
    return _logger


logger = define_log_level()


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
