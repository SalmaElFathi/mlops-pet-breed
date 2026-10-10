from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log the execution time of the wrapped function."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s took %.4fs", func.__name__, elapsed)

    return wrapper