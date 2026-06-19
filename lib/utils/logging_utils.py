"""Logging setup and timing helpers."""

import logging
import time
from contextlib import contextmanager


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, fmt: str = DEFAULT_LOG_FORMAT, force: bool = False):
    """Configure project logging output."""
    logging.basicConfig(level=level, format=fmt, force=force)


@contextmanager
def log_timed(logger: logging.Logger, message: str, level: int = logging.INFO):
    """Log elapsed time for a code block."""
    start_time = time.perf_counter()
    logger.log(level, "%s: started", message)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.log(level, "%s: finished in %.2fs", message, elapsed)

