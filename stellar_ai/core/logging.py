"""Minimal, dependency-free logging setup.

A single configured root logger ("stellar_ai") is reused across modules. No
external packages required.
"""

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the stellar_ai logger once with a clean stream handler."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger = logging.getLogger("stellar_ai")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s.%(module)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the stellar_ai namespace."""
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("stellar_ai"):
        name = "stellar_ai." + name
    return logging.getLogger(name)
